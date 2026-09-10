"""
uavscenes_dataset.py — PyTorch Dataset for PMNet on UAVScenes.

Optimized for high GPU utilization, minimal CPU/multiprocessing overhead,
and high performance (IMG_SIZE=256, spatial voxel-balanced point sampling).
"""

import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from data_utils import (
    get_calibration,
    map_labels_26_to_19,
    project_lidar_to_image,
    get_frame_list,
    IGNORE_INDEX,
    lidar_xyz_to_bev_coords,
)


class UAVScenesDataset(Dataset):
    """Frame-level dataset for UAVScenes that outputs data in PMNet format.

    Args:
        sequences:  list of sequence directory names
                    (e.g. ``['interval5_AMtown01', ...]``).
        num_point:  number of points to sample for the FUSION branch
                    (camera-FOV-cropped) per frame (default 4096).
        num_point_full: number of points to sample for the FULL,
                    uncropped point cloud per frame (default: same as
                    num_point).
        img_size:   full image is resized to ``(img_size, img_size)``
                    (default 256).
        augment:    if True, apply random horizontal flip (train mode). If
                    False, deterministic sampling and no flip.
        return_full_cloud: if True, returns `points_full`, `labels_full`,
                    `bev_coords_fused`, and `bev_coords_full`. Keep False
                    during training to avoid IPC/multiprocessing bottlenecks.
        cache_data: if True, caches raw frame assets in RAM to eliminate
                    disk I/O during training.
        expected_raw_dims: assert raw LiDAR .npy files have exactly
                    this many columns (default 3).
    """

    def __init__(self, sequences, num_point=4096, num_point_full=None,
                 img_size=256, augment=False, return_full_cloud=False,
                 cache_data=False, expected_raw_dims=3):
        super().__init__()
        self.num_point = num_point
        self.num_point_full = num_point_full or num_point
        self.img_size = img_size
        self.augment = augment
        self.return_full_cloud = return_full_cloud
        self.cache_data = cache_data
        self.expected_raw_dims = expected_raw_dims

        self.frames = []
        for seq in sequences:
            self.frames.extend(get_frame_list(seq))

        self.calibrations = {}
        for seq in sequences:
            self.calibrations[seq] = get_calibration(seq)

        self._fallback_count = 0
        self._checked_count = 0
        self._cache = {}

        print(f"[UAVScenesDataset] {len(self.frames)} frames "
              f"from {len(sequences)} sequences  "
              f"(num_point={num_point}, num_point_full={self.num_point_full}, "
              f"img_size={img_size}, augment={augment}, cache_data={cache_data})")

    def __len__(self):
        return len(self.frames)

    def fallback_rate(self):
        """Fraction of __getitem__ calls so far that hit the dummy-center
        projection fallback."""
        if self._checked_count == 0:
            return 0.0
        return self._fallback_count / self._checked_count

    def _sample_indices(self, rng, pool_size, k):
        """Sample k indices from range(pool_size), with replacement only
        if pool_size < k."""
        if pool_size >= k:
            return rng.choice(pool_size, k, replace=False)
        return rng.choice(pool_size, k, replace=True)

    def _sample_spatial_grid_indices(self, pts_raw, valid_idx, num_samples, rng):
        """Sample points with 3D spatial voxel grid balancing to prevent flat ground dominance."""
        if len(valid_idx) <= num_samples:
            if len(valid_idx) == 0:
                return np.zeros(num_samples, dtype=np.int64)
            return rng.choice(valid_idx, num_samples, replace=True)

        pts_fov = pts_raw[valid_idx]
        p_min = pts_fov.min(axis=0)
        p_max = pts_fov.max(axis=0)
        p_range = np.maximum(p_max - p_min, 1e-4)

        # 3D spatial voxel binning (32 x 32 x 16 grid)
        gx = np.floor((pts_fov[:, 0] - p_min[0]) / p_range[0] * 31.99).astype(np.int64)
        gy = np.floor((pts_fov[:, 1] - p_min[1]) / p_range[1] * 31.99).astype(np.int64)
        gz = np.floor((pts_fov[:, 2] - p_min[2]) / p_range[2] * 15.99).astype(np.int64)
        voxel_ids = gx * 512 + gy * 16 + gz

        unique_voxels, inv_idx = np.unique(voxel_ids, return_inverse=True)
        n_vox = len(unique_voxels)

        if n_vox >= num_samples:
            chosen_v = rng.choice(n_vox, num_samples, replace=False)
            perm = rng.permutation(len(valid_idx))
            first_in_vox = np.empty(n_vox, dtype=np.int64)
            first_in_vox[inv_idx[perm]] = perm
            sel = first_in_vox[chosen_v]
            return valid_idx[sel]
        else:
            perm = rng.permutation(len(valid_idx))
            first_in_vox = np.empty(n_vox, dtype=np.int64)
            first_in_vox[inv_idx[perm]] = perm
            sel = list(first_in_vox)

            rem_needed = num_samples - len(sel)
            rem_pool = np.setdiff1d(np.arange(len(valid_idx)), sel)
            if len(rem_pool) > 0:
                extra = rng.choice(rem_pool, rem_needed, replace=(len(rem_pool) < rem_needed))
                sel.extend(extra)
            else:
                extra = rng.choice(len(valid_idx), rem_needed, replace=True)
                sel.extend(extra)
            return valid_idx[np.array(sel)]

    def _load_raw_frame(self, idx):
        """Load and preprocess a single frame."""
        lidar_path, cam_path, label_path, seq_name = self.frames[idx]

        # Load raw LiDAR coordinates in float64 first for precision
        pts_raw = np.load(lidar_path).astype(np.float64)
        if pts_raw.ndim != 2 or pts_raw.shape[1] < 3:
            raise ValueError(f"{lidar_path}: expected (N, >=3) array, got {pts_raw.shape}")
        if pts_raw.shape[1] != self.expected_raw_dims:
            raise ValueError(
                f"{lidar_path}: expected {self.expected_raw_dims} columns, got {pts_raw.shape[1]}"
            )
        pts = pts_raw[:, :3]

        labels_26 = np.load(label_path).astype(np.int64)
        labels = map_labels_26_to_19(labels_26)

        # Fast direct PIL open and resize (stored as uint8 in RAM to save 75% memory)
        with Image.open(cam_path) as img_pil:
            img_w, img_h = img_pil.size
            img_resized = np.array(img_pil.resize((self.img_size, self.img_size), Image.BILINEAR))
        img_uint8 = img_resized.transpose(2, 0, 1)  # (3, H, W) uint8 (only 196 KB!)

        # Projection to camera image
        calib = self.calibrations[seq_name]
        u, v, valid_front = project_lidar_to_image(pts, calib)
        in_fov = (valid_front & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h))

        # STRICT IN-FOV FILTER: Discard all out-of-FOV points completely
        if np.any(in_fov):
            pts = pts[in_fov]
            labels = labels[in_fov]
            u = u[in_fov]
            v = v[in_fov]

        # Mapped pixel coordinates on the resized image
        col_mapped = (u / float(img_w) * (self.img_size - 1)).astype(np.float32)
        row_mapped = (v / float(img_h) * (self.img_size - 1)).astype(np.float32)

        # Coordinate normalization strictly over the In-FOV point cloud
        scene_min = pts.min(axis=0)
        scene_max = pts.max(axis=0)
        scene_range = scene_max - scene_min
        scene_range = np.where(scene_range < 1e-6, 1.0, scene_range)
        pts_norm = ((pts - scene_min) / scene_range).astype(np.float32)

        return {
            "pts": pts.astype(np.float32),
            "pts_norm": pts_norm,
            "labels": labels.astype(np.int8),
            "row_mapped": row_mapped,
            "col_mapped": col_mapped,
            "img_uint8": img_uint8,
            "seq_name": seq_name,
            "lidar_path": lidar_path,
            "label_path": label_path,
        }

    def preload_cache(self, max_frames=None, num_threads=16, verbose=True):
        """Pre-load and store frame assets in RAM upfront using multi-threading."""
        total_to_load = len(self.frames) if max_frames is None else min(max_frames, len(self.frames))
        if verbose:
            print(f"[UAVScenesDataset] Pre-loading {total_to_load} in-FOV frames into RAM using {num_threads} threads...")

        def _worker(idx):
            return idx, self._load_raw_frame(idx)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx, data in executor.map(_worker, range(total_to_load)):
                self._cache[idx] = data

        self.cache_data = True
        if verbose:
            print(f"[UAVScenesDataset] Cached {len(self._cache)} in-FOV frames in RAM successfully!")

    def __getitem__(self, idx):
        if self.augment:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed=idx)

        data = self._cache.get(idx) if self.cache_data else None
        if data is None:
            data = self._load_raw_frame(idx)
            if self.cache_data:
                self._cache[idx] = data

        # All points in data["pts_norm"] are 100% strictly in-FOV points
        n_infov_pts = len(data["pts_norm"])
        if self.augment and n_infov_pts > self.num_point:
            # 3D spatial voxel-balanced sampling to prevent flat ground dominance
            fusion_idx = self._sample_spatial_grid_indices(data["pts"], np.arange(n_infov_pts), self.num_point, rng)
        else:
            fusion_idx = self._sample_indices(rng, n_infov_pts, self.num_point)

        pts_sel = data["pts_norm"][fusion_idx].copy()
        labels_sel = data["labels"][fusion_idx].astype(np.int64)
        row_sel = data["row_mapped"][fusion_idx]
        col_sel = data["col_mapped"][fusion_idx]
        proj_indices = np.stack([row_sel, col_sel], axis=-1).astype(np.float32)
        img_norm = (data["img_uint8"].astype(np.float32) / 255.0)

        # Train augmentation (horizontal flip)
        if self.augment and np.random.random() > 0.5:
            pts_sel[:, 1] = 1.0 - pts_sel[:, 1]
            img_norm = np.ascontiguousarray(img_norm[:, :, ::-1])
            proj_indices[:, 1] = (self.img_size - 1) - proj_indices[:, 1]

        item = {
            "points_fused": torch.from_numpy(pts_sel),
            "labels_fused": torch.from_numpy(labels_sel).long(),
            "image": torch.from_numpy(img_norm),
            "proj_indices": torch.from_numpy(proj_indices),
            "used_fallback": torch.tensor(False, dtype=torch.bool),
            "seq_name": data["seq_name"],
            "lidar_path": data["lidar_path"],
            "label_path": data["label_path"],
        }

        # Optional full-cloud branch for evaluation / ablation
        if self.return_full_cloud:
            full_idx = self._sample_indices(rng, len(data["pts_norm"]), self.num_point_full)
            pts_full_norm = data["pts_norm"][full_idx].copy()
            labels_full = data["labels"][full_idx]

            bev_coords_sel = lidar_xyz_to_bev_coords(data["pts"][sel])
            bev_coords_full_sel = lidar_xyz_to_bev_coords(data["pts"][full_idx])

            if self.augment and np.random.random() > 0.5:
                pts_full_norm[:, 1] = 1.0 - pts_full_norm[:, 1]

            item["points_full"] = torch.from_numpy(pts_full_norm)
            item["labels_full"] = torch.from_numpy(labels_full).long()
            item["bev_coords_fused"] = torch.from_numpy(bev_coords_sel)
            item["bev_coords_full"] = torch.from_numpy(bev_coords_full_sel)

        return item

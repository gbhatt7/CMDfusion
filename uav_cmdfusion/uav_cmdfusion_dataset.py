"""
uav_cmdfusion_dataset.py — PyTorch Dataset for CMDFusion on UAVScenes.

Produces the data_dict format expected by CMDFusion:
  - Variable-length points (all volume-filtered points, NOT fixed N sampling)
  - batch_idx for flat-tensor batching across the batch
  - img_indices (pixel coords of camera-visible points)
  - point2img_index (which points are visible in the camera)

Uses the SAME underlying data loading as the PMNet pipeline:
  - get_frame_list(), get_calibration(), map_labels_26_to_19(),
    project_lidar_to_image() — all from data_utils.py
"""

import os
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
)


class UAVScenesCMDFusionDataset(Dataset):
    """Frame-level dataset for UAVScenes that outputs data in CMDFusion format.

    Key differences from the PMNet UAVScenesDataset:
      - Returns ALL points after volume-space filtering (variable length)
      - Returns img_indices: (M, 2) pixel coordinates of visible points
      - Returns point2img_index: (M,) indices into point array for visible points
      - Custom collate_fn_uav() batches variable-length point clouds

    Args:
        sequences:  list of sequence directory names.
        img_h:      target image height for feature map correspondence (default 360).
        img_w:      target image width for feature map correspondence (default 480).
        augment:    if True, apply 3D augmentation (rotation, flip, scale) and 2D flip.
        min_volume_space: [x_min, y_min, z_min] for volume filtering.
        max_volume_space: [x_max, y_max, z_max] for volume filtering.
        image_normalizer: optional (mean, std) for image normalization.
        max_dropout_ratio: max fraction of points to randomly drop (train only).
    """

    def __init__(self, sequences, img_h=360, img_w=480, augment=False,
                 min_volume_space=None, max_volume_space=None,
                 image_normalizer=None, max_dropout_ratio=0.2):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment
        self.min_volume_space = min_volume_space or [-100, -100, -50]
        self.max_volume_space = max_volume_space or [100, 100, 150]
        self.image_normalizer = image_normalizer
        self.max_dropout_ratio = max_dropout_ratio

        self.frames = []
        for seq in sequences:
            self.frames.extend(get_frame_list(seq))

        self.calibrations = {}
        for seq in sequences:
            self.calibrations[seq] = get_calibration(seq)

        print(f"[UAVScenesCMDFusionDataset] {len(self.frames)} frames "
              f"from {len(sequences)} sequences  "
              f"(img={img_h}x{img_w}, augment={augment})")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        lidar_path, cam_path, label_path, seq_name = self.frames[idx]

        # ============ Load raw LiDAR ============
        pts_raw = np.load(lidar_path).astype(np.float64)
        if pts_raw.ndim != 2 or pts_raw.shape[1] < 3:
            raise ValueError(f"{lidar_path}: expected (N, >=3), got {pts_raw.shape}")
        xyz = pts_raw[:, :3].astype(np.float32)

        # ============ Load labels ============
        labels_26 = np.load(label_path).astype(np.int64)
        labels = map_labels_26_to_19(labels_26).reshape(-1, 1)  # (N, 1)

        origin_len = len(xyz)
        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(xyz))

        # ============ Volume space filtering ============
        mask_x = (xyz[:, 0] > self.min_volume_space[0]) & (xyz[:, 0] < self.max_volume_space[0])
        mask_y = (xyz[:, 1] > self.min_volume_space[1]) & (xyz[:, 1] < self.max_volume_space[1])
        mask_z = (xyz[:, 2] > self.min_volume_space[2]) & (xyz[:, 2] < self.max_volume_space[2])
        mask = mask_x & mask_y & mask_z

        xyz = xyz[mask]
        labels = labels[mask]
        ref_index = ref_index[mask]
        point_num = len(xyz)

        # ============ Point dropout (train only) ============
        if self.augment and self.max_dropout_ratio > 0:
            dropout_ratio = np.random.random() * self.max_dropout_ratio
            drop_idx = np.where(np.random.random(xyz.shape[0]) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                xyz[drop_idx, :] = xyz[0, :]
                labels[drop_idx, :] = labels[0, :]
                ref_index[drop_idx] = ref_index[0]

        # ============ Load and resize image ============
        with Image.open(cam_path) as img_pil:
            img_w_orig, img_h_orig = img_pil.size
            image = img_pil.resize((self.img_w, self.img_h), Image.BILINEAR)

        # ============ Project LiDAR to camera image ============
        calib = self.calibrations[seq_name]
        u, v, valid_front = project_lidar_to_image(xyz.astype(np.float64), calib)

        # Points visible in camera FOV
        keep_idx = (valid_front
                    & (u >= 0) & (u < img_w_orig)
                    & (v >= 0) & (v < img_h_orig))

        # Map pixel coordinates to the target (resized) image space
        # img_indices are (row, col) in the [img_h, img_w] feature map
        row_mapped = (v[keep_idx] / float(img_h_orig) * (self.img_h - 1))
        col_mapped = (u[keep_idx] / float(img_w_orig) * (self.img_w - 1))
        points_img = np.stack([row_mapped, col_mapped], axis=-1).astype(np.float32)

        # ============ 3D Augmentation ============
        if self.augment:
            # Random rotation around Z axis
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.array([[c, s], [-s, c]], dtype=np.float32)
            xyz[:, :2] = (xyz[:, :2] @ j)

            # Random flip
            flip_type = np.random.choice(4)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

            # Random scale
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, :2] = (xyz[:, :2] * noise_scale).astype(np.float32)

            # Random translation
            noise_translate = np.array([
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
            ], dtype=np.float32)
            xyz += noise_translate

        # ============ Image label and point-to-image index ============
        img_label = labels[keep_idx]
        # Indices into the xyz array that are visible in the camera
        point2img_index = np.arange(len(xyz))[keep_idx]

        # Point features: just XYZ for UAVScenes (no intensity)
        feat = xyz.copy()  # (N, 3)

        # Clamp img_indices to valid range
        img_indices = points_img.copy()
        img_indices[:, 0] = np.clip(img_indices[:, 0], 0, self.img_h - 1)
        img_indices[:, 1] = np.clip(img_indices[:, 1], 0, self.img_w - 1)
        img_indices = img_indices.astype(np.int64)

        # ============ Image preprocessing ============
        image = image.convert("RGB")
        image = np.array(image, dtype=np.float32, copy=False) / 255.0

        # 2D augmentation: random horizontal flip
        if self.augment and np.random.rand() < 0.5:
            image = np.ascontiguousarray(np.fliplr(image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]

        # Optional image normalization (ImageNet stats)
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std

        # ============ Build output dict ============
        data_dict = {
            'point_feat': feat,             # (N, 3) xyz features
            'point_label': labels,          # (N, 1) labels
            'ref_xyz': ref_pc,              # (N_orig, 3) reference xyz
            'ref_label': ref_labels,        # (N_orig, 1) reference labels
            'ref_label_o': ref_labels,      # (N_orig, 1) original labels
            'ref_index': ref_index,         # (N,) indices into original
            'mask': mask,                   # (N_orig,) volume mask
            'point_num': point_num,         # int
            'origin_len': origin_len,       # int
            'root': lidar_path,             # str path for reference
            'img': image,                   # (H, W, 3) normalized image
            'img_indices': img_indices,     # (M, 2) pixel coords of visible pts
            'img_label': img_label,         # (M, 1) labels of visible pts
            'point2img_index': point2img_index,  # (M,) indices into N
        }

        return data_dict


def collate_fn_uav(data):
    """Collate function for CMDFusion with UAVScenes data.

    Batches variable-length point clouds into a single flat tensor with
    batch_idx, matching the format expected by CMDFusion's voxelization.

    Returns:
        dict with:
          - points:          (total_N, 3) concatenated point features
          - batch_idx:       (total_N,) batch index per point
          - labels:          (total_N,) concatenated labels
          - batch_size:      int
          - img:             (B, 3, H, W) stacked images
          - img_indices:     list of (M_i, 2) arrays per batch
          - point2img_index: list of (M_i,) tensors per batch
          - raw_labels, origin_len, indices: for validation
    """
    point_num = [d['point_num'] for d in data]
    batch_size = len(point_num)
    ref_labels = data[0]['ref_label']
    ref_labels_o = data[0]['ref_label_o']
    origin_len = data[0]['origin_len']
    ref_indices = [torch.from_numpy(d['ref_index']) for d in data]
    point2img_index = [torch.from_numpy(d['point2img_index']).long() for d in data]
    path = [d['root'] for d in data]

    img = [torch.from_numpy(d['img']) for d in data]
    img_indices = [d['img_indices'] for d in data]
    img_label = [torch.from_numpy(d['img_label']) for d in data]

    # Batch index: each point gets its batch number
    b_idx = []
    for i in range(batch_size):
        b_idx.append(torch.ones(point_num[i]) * i)

    points = [torch.from_numpy(d['point_feat']) for d in data]
    ref_xyz = [torch.from_numpy(d['ref_xyz']) for d in data]
    labels = [torch.from_numpy(d['point_label']) for d in data]

    return {
        'points': torch.cat(points).float(),                    # (total_N, 3)
        'ref_xyz': torch.cat(ref_xyz).float(),                  # (total_ref, 3)
        'batch_idx': torch.cat(b_idx).long(),                   # (total_N,)
        'batch_size': batch_size,                                # int
        'labels': torch.cat(labels).long().squeeze(1),           # (total_N,)
        'raw_labels': torch.from_numpy(ref_labels).long(),       # (N_orig, 1)
        'raw_labels_o': torch.from_numpy(ref_labels_o).long(),   # (N_orig, 1)
        'origin_len': origin_len,                                # int
        'indices': torch.cat(ref_indices).long(),                # (total_N,)
        'point2img_index': point2img_index,                      # list of tensors
        'img': torch.stack(img, 0).permute(0, 3, 1, 2).float(), # (B, 3, H, W)
        'img_indices': img_indices,                              # list of arrays
        'img_label': torch.cat(img_label, 0).squeeze(1).long(), # (total_M,)
        'path': path,                                            # list of str
    }

"""
data_utils.py — Data utilities for PMNet x UAVScenes pipeline.

CHANGES FROM THE PREVIOUS VERSION (read this before trusting any number
this pipeline produces):

  1. Train/val/test split rewritten. The old split put ALL of val and ALL
     of test in Hong Kong sequences (HKairport/HKisland) while claiming
     "geographic diversity is preserved in every split" -- that claim was
     false. AMtown/AMvalley never appeared in val or test. The new split
     puts one run from EACH of the four locations (AMtown, AMvalley,
     HKairport, HKisland) into every split.

  2. Sequences whose calibration maps to `_Featureless_GNSS` are excluded
     from all splits by default. The UAVScenes paper distinguishes 20
     fully-annotated sequences from a separate "Featureless_GNSS" category
     described as carrying only instance annotations for dynamic objects,
     not full semantic segmentation. "HKisland_GNSS_Evening" was being
     used as a TEST sequence in the old split, and its calibration entry
     names it into that exact category. Until you have personally opened
     its label folder and confirmed dense per-point semantic labels exist,
     it stays excluded. See UNVERIFIED_LABEL_SEQUENCES below.

  3. `verify_sequence_labels()` added -- call this on every sequence
     before training. It fails loudly (raises) if a label directory is
     missing or empty, instead of silently producing an empty frame list
     that quietly shrinks your dataset.

  4. Class 0 (background), raw id 15 (solar_board), raw id 16 (umbrella)
     are ALL mapped to IGNORE_INDEX (-1). "Background" is intentionally
     excluded from training and evaluation entirely -- it is not a
     meaningful semantic target for aerial 3-D segmentation.
     The 8 TRAINABLE classes are:
       0 = Building      (roof, transparent_roof)
       1 = Road          (dirt_motor_road, paved_motor_road, airstrip,
                          car_park, paved_walk)
       2 = Water         (river, pool)
       3 = Vegetation    (green_field, wild_field)
       4 = Bridge        (bridge)
       5 = Container     (container)
       6 = Traffic Barrier (traffic_barrier)
       7 = Vehicle       (sedan, truck)
     Any remaining unnamed raw classes (ids 7, 8, 12, 21, 22, 23, 25)
     also map to IGNORE_INDEX so they do not silently corrupt metrics.

  5. compute_metrics() excludes ignored labels (-1) from the
     confusion matrix / IoU / accuracy instead of silently counting them
     as a real class.
"""

import os
import numpy as np

# ================================================================
# Paths -- everything is relative to the Multimodal_segmentation root
# ================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
UAVSCENES_ROOT = os.path.join(_PROJECT_ROOT, "UAVScenes")

CAM_LIDAR_DIR = os.path.join(UAVSCENES_ROOT, "interval5_CAM_LIDAR")
CAM_LABEL_DIR = os.path.join(UAVSCENES_ROOT, "interval5_CAM_label")
LIDAR_LABEL_DIR = os.path.join(UAVSCENES_ROOT, "interval5_LIDAR_label")

# ================================================================
# 26-class -> 19-class Label Mapping  (Background = class 0)
# ================================================================
#
#  Target class index -> class name -> raw UAVScenes class IDs
#  ----------------------------------------------------------------
#   0 = Background      raw ids: 0 (background), and unnamed (7, 8, 12, 21, 22, 23, 25)
#   1 = Roof            raw id:  1 (roof)
#   2 = Dirt Road       raw id:  2 (dirt_motor_road)
#   3 = Paved Road      raw id:  3 (paved_motor_road)
#   4 = River           raw id:  4 (river)
#   5 = Pool            raw id:  5 (pool)
#   6 = Bridge          raw id:  6 (bridge)
#   7 = Conta.          raw id:  9 (container)
#   8 = Airstrip        raw id:  10 (airstrip)
#   9 = Traffic Barrier raw id:  11 (traffic_barrier)
#  10 = Green Field     raw id:  13 (green_field)
#  11 = Wild Field      raw id:  14 (wild_field)
#  12 = Solar Panel     raw id:  15 (solar_board)
#  13 = Umbre.          raw id:  16 (umbrella)
#  14 = Transp. Roof    raw id:  17 (transparent_roof)
#  15 = Car Park        raw id:  18 (car_park)
#  16 = Paved Walk      raw id:  19 (paved_walk)
#  17 = Sedan           raw id:  20 (sedan)
#  18 = Truck           raw id:  24 (truck)
#
# Background (class 0) is a trainable class included in loss and confusion matrix,
# but EXCLUDED from final mIoU calculation (evaluated across classes 1..18).
NUM_CLASSES = 19
CLASS_NAMES = [
    "Background",      # 0
    "Roof",            # 1
    "Dirt Road",       # 2
    "Paved Road",      # 3
    "River",           # 4
    "Pool",            # 5
    "Bridge",          # 6
    "Conta.",          # 7
    "Airstrip",        # 8
    "Traffic Barrier", # 9
    "Green Field",     # 10
    "Wild Field",      # 11
    "Solar Panel",     # 12
    "Umbre.",          # 13
    "Transp. Roof",    # 14
    "Car Park",        # 15
    "Paved Walk",      # 16
    "Sedan",           # 17
    "Truck",           # 18
]
IGNORE_INDEX = -1

# Official 19-class RGB palette matching UAVScenes
PALETTE = np.array([
    [ 80/255,  80/255,  80/255],  # 0 Background
    [119/255,  11/255,  32/255],  # 1 Roof
    [180/255, 165/255, 180/255],  # 2 Dirt Road
    [128/255,  64/255, 128/255],  # 3 Paved Road
    [173/255, 216/255, 230/255],  # 4 River
    [  0/255,  80/255, 100/255],  # 5 Pool
    [150/255, 100/255, 100/255],  # 6 Bridge
    [250/255, 170/255,  30/255],  # 7 Conta.
    [ 81/255,   0/255,  81/255],  # 8 Airstrip
    [102/255, 102/255, 156/255],  # 9 Traffic Barrier
    [107/255, 142/255,  35/255],  # 10 Green Field
    [210/255, 180/255, 140/255],  # 11 Wild Field
    [220/255, 220/255,   0/255],  # 12 Solar Panel
    [153/255, 153/255, 153/255],  # 13 Umbre.
    [  0/255,   0/255,  90/255],  # 14 Transp. Roof
    [250/255, 170/255, 160/255],  # 15 Car Park
    [244/255,  35/255, 232/255],  # 16 Paved Walk
    [  0/255,   0/255, 142/255],  # 17 Sedan
    [  0/255,   0/255,  70/255],  # 18 Truck
], dtype=np.float32)
IGNORE_COLOR = np.array([0.0, 0.0, 0.0], dtype=np.float32)

# ----------------------------------------------------------------
# Build the 26-entry lookup table.
# Default is class 0 (Background) for raw 0 and all unnamed IDs.
# ----------------------------------------------------------------
_LABEL_MAP_26_TO_19 = np.zeros(26, dtype=np.int64)

# 0 = Background
_LABEL_MAP_26_TO_19[0]  = 0
_LABEL_MAP_26_TO_19[7]  = 0
_LABEL_MAP_26_TO_19[8]  = 0
_LABEL_MAP_26_TO_19[12] = 0
_LABEL_MAP_26_TO_19[21] = 0
_LABEL_MAP_26_TO_19[22] = 0
_LABEL_MAP_26_TO_19[23] = 0
_LABEL_MAP_26_TO_19[25] = 0

# 1 = Roof
_LABEL_MAP_26_TO_19[1]  = 1
# 2 = Dirt Road
_LABEL_MAP_26_TO_19[2]  = 2
# 3 = Paved Road
_LABEL_MAP_26_TO_19[3]  = 3
# 4 = River
_LABEL_MAP_26_TO_19[4]  = 4
# 5 = Pool
_LABEL_MAP_26_TO_19[5]  = 5
# 6 = Bridge
_LABEL_MAP_26_TO_19[6]  = 6
# 7 = Conta.
_LABEL_MAP_26_TO_19[9]  = 7
# 8 = Airstrip
_LABEL_MAP_26_TO_19[10] = 8
# 9 = Traffic Barrier
_LABEL_MAP_26_TO_19[11] = 9
# 10 = Green Field
_LABEL_MAP_26_TO_19[13] = 10
# 11 = Wild Field
_LABEL_MAP_26_TO_19[14] = 11
# 12 = Solar Panel
_LABEL_MAP_26_TO_19[15] = 12
# 13 = Umbre.
_LABEL_MAP_26_TO_19[16] = 13
# 14 = Transp. Roof
_LABEL_MAP_26_TO_19[17] = 14
# 15 = Car Park
_LABEL_MAP_26_TO_19[18] = 15
# 16 = Paved Walk
_LABEL_MAP_26_TO_19[19] = 16
# 17 = Sedan
_LABEL_MAP_26_TO_19[20] = 17
# 18 = Truck
_LABEL_MAP_26_TO_19[24] = 18

_LABEL_MAP_26_TO_8 = _LABEL_MAP_26_TO_19  # backward compatibility alias


def map_labels_26_to_19(labels):
    """Vectorised mapping from 26-class UAVScenes labels to 19-class labels.

    Background (raw id 0, plus unnamed raw ids 7, 8, 12, 21, 22, 23, 25) is class 0.
    The 18 semantic classes (Roof, Dirt Road, Paved Road, River, Pool, Bridge,
    Conta., Airstrip, Traffic Barrier, Green Field, Wild Field, Solar Panel,
    Umbre., Transp. Roof, Car Park, Paved Walk, Sedan, Truck) map to 1..18.

    Args:
        labels: np.ndarray of integer labels (values 0-25).

    Returns:
        np.ndarray of the same shape with values in [0, 18].
    """
    labels = np.clip(labels, 0, 25).astype(np.int64)
    return _LABEL_MAP_26_TO_19[labels]


# Backward compatibility alias
map_labels_26_to_8 = map_labels_26_to_19


# ================================================================
# Calibration Data (embedded from UAVScenes/calibration_results.py)
# ================================================================
_AMtown = {
    "camera_intrinsic": [1453.72, 0.0, 1172.18,
                         0.0, 1453.28, 1041.78,
                         0.0, 0.0, 1.0],
    "camera_dist_coeffs": [-0.121, 0.1113, 0.0016, 0.00013, -0.062353],
    "camera_ext_R": [0.00298088, -0.999728, -0.0231416,
                     -0.00504636, 0.0231263, -0.99972,
                     0.999983, 0.00309683, -0.00497605],
    "camera_ext_t": [0.0025563, 0.0567484, -0.0512149],
}
_AMvalley = {
    "camera_intrinsic": [1453.88, 0.0, 1182.53,
                         0.0, 1452.85, 1045.82,
                         0.0, 0.0, 1.0],
    "camera_dist_coeffs": [-0.052, 0.1168, 0.0015, 0.00013, -0.068564],
    "camera_ext_R": [0.00298068, -0.999735, -0.0231428,
                     -0.00504595, 0.023132, -0.99974,
                     0.999985, 0.00309701, -0.00497598],
    "camera_ext_t": [-0.0025563, 0.0567484, -0.0512149],
}
_HKisland = {
    "camera_intrinsic": [1444.43, 0.0, 1177.8,
                         0.0, 1444.34, 1043.6,
                         0.0, 0.0, 1.0],
    "camera_dist_coeffs": [-0.053, 0.121, 0.00127, 0.00043, -0.06495],
    "camera_ext_R": [0.00352762, -0.999765, -0.0213775,
                     -0.0111803, 0.0213369, -0.99971,
                     0.999931, 0.00376561, -0.0111025],
    "camera_ext_t": [-0.0025563, 0.0470454, -0.0513375],
}
_HK_GNSS = {
    "camera_intrinsic": [1444.43, 0.0, 1179.50,
                         0.0, 1444.34, 1044.90,
                         0.0, 0.0, 1.0],
    "camera_dist_coeffs": [-0.0560, 0.1180, 0.00122, 0.00064, -0.0627],
    "camera_ext_R": [0.00363212, -0.999819, -0.0213618,
                     -0.0111679, 0.0214512, -0.999591,
                     0.999879, 0.00375613, -0.0111134],
    "camera_ext_t": [-0.0021928, 0.0470312, -0.0513126],
}
# NOTE: this calibration block's name is the tell. In the UAVScenes
# paper, "Featureless_GNSS" sequences are a distinct category with only
# instance annotations for dynamic objects -- NOT full semantic
# segmentation labels. Any sequence mapped to this calibration block
# should be treated as unverified for semantic segmentation until you
# have personally confirmed its label folder contains dense per-point
# class labels.
_Featureless_GNSS = {
    "camera_intrinsic": [1450.09, 0, 1177.5,
                         0.0, 1450.09, 1044.5,
                         0.0, 0.0, 1.0],
    "camera_dist_coeffs": [-0.0558, 0.1247, 0.00126, 0.0007, -0.06762],
    "camera_ext_R": [0.00941312, -0.999586, -0.0271775,
                     -0.00737429, 0.0271086, -0.999605,
                     0.999929, 0.00960988, -0.00711613],
    "camera_ext_t": [-0.0025563, 0.0567484, -0.0512149],
}

# Scene name (without 'interval5_' prefix) -> calibration dict
_SCENE_TO_CALIB = {
    "AMtown01": _AMtown, "AMtown02": _AMtown, "AMtown03": _AMtown,
    "AMvalley01": _AMvalley, "AMvalley02": _AMvalley, "AMvalley03": _AMvalley,
    "HKairport01": _HK_GNSS, "HKairport02": _HK_GNSS, "HKairport03": _HK_GNSS,
    "HKisland01": _HKisland, "HKisland02": _HKisland, "HKisland03": _HKisland,
    "HKairport_GNSS_Evening": _HK_GNSS,
    "HKisland_GNSS_Evening": _Featureless_GNSS,
    "HKairport_GNSS01": _HK_GNSS, "HKairport_GNSS02": _HK_GNSS, "HKairport_GNSS03": _HK_GNSS,
    "HKisland_GNSS01": _HK_GNSS, "HKisland_GNSS02": _HK_GNSS, "HKisland_GNSS03": _HK_GNSS,
}

# Sequences excluded from every split until you've personally verified
# their label folders contain real dense semantic labels. Built
# automatically from any sequence mapped to the _Featureless_GNSS block,
# plus anything else you want to flag by name below.
UNVERIFIED_LABEL_SEQUENCES = {
    name for name, calib in _SCENE_TO_CALIB.items() if calib is _Featureless_GNSS
}


def get_calibration(seq_name):
    """Return calibration dict for a given sequence directory name.

    Args:
        seq_name: e.g. ``'interval5_AMtown01'``

    Returns:
        dict with keys ``camera_intrinsic``, ``camera_dist_coeffs``,
        ``camera_ext_R``, ``camera_ext_t``.
    """
    scene = seq_name.replace("interval5_", "")
    return _SCENE_TO_CALIB[scene]


# ================================================================
# Perspective Projection
# ================================================================
def project_lidar_to_image(pts, calib):
    """Project 3-D LiDAR points onto the camera's 2-D image plane.

    Args:
        pts:   (N, 3) array -- XYZ in LiDAR frame.
        calib: calibration dict.

    Returns:
        u:     (N,) horizontal pixel coordinates.
        v:     (N,) vertical pixel coordinates.
        valid: (N,) boolean mask -- True for points in front of the camera.
    """
    R = np.array(calib["camera_ext_R"]).reshape(3, 3)
    t = np.array(calib["camera_ext_t"]).reshape(3, 1)
    K = np.array(calib["camera_intrinsic"]).reshape(3, 3)
    dist = np.array(calib["camera_dist_coeffs"])

    pts_cam = (R @ pts[:, :3].T + t).T          # (N, 3)

    valid = pts_cam[:, 2] > 0.1

    z_safe = np.where(valid, pts_cam[:, 2], 1.0)
    x = pts_cam[:, 0] / z_safe
    y = pts_cam[:, 1] / z_safe

    r2 = x ** 2 + y ** 2
    radial = 1.0 + dist[0] * r2 + dist[1] * r2 ** 2 + dist[4] * r2 ** 3
    x_d = x * radial + 2 * dist[2] * x * y + dist[3] * (r2 + 2 * x ** 2)
    y_d = y * radial + dist[2] * (r2 + 2 * y ** 2) + 2 * dist[3] * x * y

    u = K[0, 0] * x_d + K[0, 2]
    v = K[1, 1] * y_d + K[1, 2]

    return u, v, valid


def lidar_xyz_to_bev_coords(pts):
    """Convert XYZ LiDAR coordinates to top-down BEV coordinates.

    The convention used here is the same one used by the notebook
    visualization: lateral is encoded as -Y and depth as Z. This is a
    geometry conversion only; it does not change the original XYZ point
    features used for the model.
    """
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[-1] < 3:
        raise ValueError(
            f"Expected LiDAR coordinates with at least 3 values per point, "
            f"got shape {pts.shape}."
        )
    return np.stack((-pts[..., 1], pts[..., 2]), axis=-1).astype(np.float32)


# ================================================================
# Automatic "up axis" detection for a genuine top-down BEV
# ================================================================
def detect_vertical_axis(pts):
    """Heuristically pick which of X/Y/Z is the vertical axis for a frame.

    HEURISTIC, NOT A GUARANTEE: assumes the vertical extent of a single
    LiDAR frame's points is smaller than the horizontal extent (true for
    most aerial nadir/oblique captures, but can fail over very steep
    terrain, e.g. AMvalley). Sanity-check this visually per-sequence
    before trusting it blindly -- that's exactly what the visualization
    section of the pipeline notebook is for.

    Returns:
        vertical_axis: int in {0, 1, 2} (X, Y, or Z)
        horizontal_axes: tuple of the other two axis indices, in order
    """
    ranges = pts[:, :3].max(axis=0) - pts[:, :3].min(axis=0)
    vertical_axis = int(np.argmin(ranges))
    horizontal_axes = tuple(i for i in range(3) if i != vertical_axis)
    return vertical_axis, horizontal_axes


# ================================================================
# Sequence label verification (fail loud, not silent)
# ================================================================
def verify_sequence_labels(seq_name, min_frames=1):
    """Raise if a sequence's label directory is missing/empty/suspect.

    Call this on EVERY sequence before you build a split from it. This
    is intentionally strict: an empty or missing label directory should
    stop execution, not silently shrink your dataset.
    """
    scene = seq_name.replace("interval5_", "")
    if scene in UNVERIFIED_LABEL_SEQUENCES:
        raise ValueError(
            f"Sequence '{seq_name}' is mapped to the _Featureless_GNSS "
            f"calibration block. Per the UAVScenes paper, sequences in "
            f"that category may carry only instance annotations for "
            f"dynamic objects, not full semantic segmentation labels. "
            f"Refusing to include it in a split automatically. If you "
            f"have personally confirmed this sequence's label folder "
            f"contains real dense semantic labels, remove it from "
            f"UNVERIFIED_LABEL_SEQUENCES explicitly -- don't just "
            f"catch this exception."
        )

    label_dir = os.path.join(LIDAR_LABEL_DIR, seq_name, "interval5_LIDAR_label_id")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(
            f"Label directory does not exist for '{seq_name}': {label_dir}"
        )
    n_files = len([f for f in os.listdir(label_dir) if f.endswith(".npy")])
    if n_files < min_frames:
        raise ValueError(
            f"Label directory for '{seq_name}' exists but has only "
            f"{n_files} label files (< {min_frames} required): {label_dir}"
        )
    return n_files


# ================================================================
# Train / Val / Test Split (by sequence, genuinely geo-diverse)
# ================================================================
def get_train_val_test_split(verify=True):
    """Return three lists of sequence directory names.

    Every split (train, val, AND test) contains at least one run from
    each of the four locations (AMtown, AMvalley, HKairport, HKisland).
    The previous split put every val/test sequence in Hong Kong and
    claimed "geographic diversity is preserved" -- it wasn't. This
    version actually is diverse, at the cost of a smaller train set.

    Sequences with unverified/likely-absent semantic labels
    (UNVERIFIED_LABEL_SEQUENCES) are excluded entirely by default.

    Args:
        verify: if True (default), calls ``verify_sequence_labels`` on
            every sequence in every split before returning, and raises
            if any sequence's labels are missing, empty, or unverified.
            Set to False only for quick offline sanity checks where you
            know the label directories aren't mounted.
    """
    train = [
        "interval5_AMtown01", "interval5_AMvalley01",
        "interval5_HKairport01", "interval5_HKisland01",
        "interval5_HKairport_GNSS01", "interval5_HKisland_GNSS01",
        "interval5_HKairport_GNSS02", "interval5_HKisland_GNSS02",
    ]
    val = [
        "interval5_AMtown02", "interval5_AMvalley02",
        "interval5_HKairport02", "interval5_HKisland02",
    ]
    test = [
        "interval5_AMtown03", "interval5_AMvalley03",
        "interval5_HKairport03", "interval5_HKisland03",
    ]

    if verify:
        for split_name, seqs in [("train", train), ("val", val), ("test", test)]:
            for seq in seqs:
                verify_sequence_labels(seq)

    return train, val, test


# ================================================================
# Frame Discovery
# ================================================================
def get_frame_list(seq_name):
    """Discover all valid frames for one sequence.

    Each frame is a tuple:
        ``(lidar_npy_path, cam_jpg_path, label_npy_path, seq_name)``

    Only frames where *all three* files exist are returned.
    """
    lidar_dir = os.path.join(CAM_LIDAR_DIR, seq_name, "interval5_LIDAR")
    cam_dir = os.path.join(CAM_LIDAR_DIR, seq_name, "interval5_CAM")
    label_dir = os.path.join(LIDAR_LABEL_DIR, seq_name, "interval5_LIDAR_label_id")

    lidar_files = sorted(f for f in os.listdir(lidar_dir) if f.endswith(".npy"))
    cam_set = set(os.listdir(cam_dir)) if os.path.isdir(cam_dir) else set()
    label_set = set(os.listdir(label_dir)) if os.path.isdir(label_dir) else set()

    frames = []
    for fname in lidar_files:
        cam_ts = os.path.splitext(fname)[0].split("_")[0].replace("image", "", 1)
        cam_fname = f"{cam_ts}.jpg"

        if cam_fname in cam_set and fname in label_set:
            lidar_path = os.path.join(lidar_dir, fname)
            cam_path = os.path.join(cam_dir, cam_fname)
            label_path = os.path.join(label_dir, fname)
            frames.append((lidar_path, cam_path, label_path, seq_name))

    if len(frames) == 0:
        raise ValueError(
            f"get_frame_list('{seq_name}') found ZERO matching frames. "
            f"Either the sequence name is wrong, the directory layout "
            f"doesn't match what this code expects, or the filename "
            f"timestamp-matching heuristic silently failed for every "
            f"frame. Don't let this pass silently."
        )

    return frames


def build_frame_list(sequences):
    """Build the combined frame list for a list of sequence names."""
    frames = []
    for seq in sequences:
        frames.extend(get_frame_list(seq))
    return frames


# ================================================================
# Class Weight Computation
# ================================================================
def compute_class_weights(frame_list, num_classes=NUM_CLASSES, max_frames=500,
                          weight_cap=5.0):
    """Compute per-class weights using smooth logarithmic inverse-frequency balancing
    strictly over In-FOV points matching the training distribution.
    """
    from PIL import Image
    rng = np.random.default_rng(42)
    n = min(max_frames, len(frame_list))
    indices = rng.choice(len(frame_list), n, replace=False)

    counts = np.zeros(num_classes, dtype=np.int64)
    for idx in indices:
        lidar_path, cam_path, label_path, seq_name = frame_list[idx]
        pts_raw = np.load(lidar_path).astype(np.float64)[:, :3]
        labels = map_labels_26_to_19(np.load(label_path))

        # Strict in-FOV filtering
        calib = get_calibration(seq_name)
        with Image.open(cam_path) as img_pil:
            img_w, img_h = img_pil.size
        u, v, valid_front = project_lidar_to_image(pts_raw, calib)
        in_fov = (valid_front & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h))

        if np.any(in_fov):
            labels = labels[in_fov]

        labels = labels[labels != IGNORE_INDEX]
        uniq, cnts = np.unique(labels, return_counts=True)
        for cls_id, cnt in zip(uniq, cnts):
            if 0 <= cls_id < num_classes:
                counts[cls_id] += cnt

    total = float(counts.sum())
    freqs = counts.astype(np.float64) / max(total, 1.0)
    
    # Smooth logarithmic inverse frequency: 1.0 / log(1.05 + freq)
    weights = 1.0 / np.log(1.05 + freqs)
    weights = weights / max(weights.mean(), 1e-4)
    weights = np.clip(weights, 0.2, weight_cap)

    return weights.astype(np.float32)


# ================================================================
# Focal Loss (for highly imbalanced 19-class aerial distributions)
# ================================================================
class FocalLoss:
    """Focal Loss with class weights and ignore_index support.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy background/ground points and focuses gradient updates
    on hard boundaries and under-represented foreground classes.
    """

    def __init__(self, weight=None, gamma=2.0, ignore_index=IGNORE_INDEX, reduction="mean"):
        import torch
        import torch.nn as nn
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def __call__(self, pred, target):
        import torch
        import torch.nn.functional as F

        # pred: (B, C, N), target: (B, N)
        ce_loss = F.cross_entropy(
            pred, target, weight=self.weight, ignore_index=self.ignore_index, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        valid_mask = target != self.ignore_index
        if self.reduction == "mean":
            return focal_loss[valid_mask].mean() if valid_mask.any() else focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss[valid_mask].sum()
        return focal_loss


# ================================================================
# Evaluation Metrics
# ================================================================
def compute_metrics(all_preds, all_targets, num_classes=NUM_CLASSES,
                     ignore_index=IGNORE_INDEX):
    """Compute per-class IoU, mIoU, and overall accuracy.

    Points/pixels with target == ignore_index are excluded entirely --
    they no longer silently count as a real class the way they did
    before.

    Args:
        all_preds:   1-D array of predicted labels.
        all_targets: 1-D array of ground-truth labels.
        num_classes: number of classes.
        ignore_index: label value to exclude from every computation.

    Returns:
        dict with ``per_class_iou`` (array), ``miou`` (float),
        ``accuracy`` (float), ``confusion_matrix`` (2-D array),
        ``n_ignored`` (int, how many points were excluded).
    """
    all_preds = np.asarray(all_preds, dtype=np.int64)
    all_targets = np.asarray(all_targets, dtype=np.int64)

    keep = all_targets != ignore_index
    n_ignored = int((~keep).sum())
    all_preds = all_preds[keep]
    all_targets = all_targets[keep]

    flat = all_targets * num_classes + all_preds
    cm = np.bincount(flat, minlength=num_classes * num_classes)
    cm = cm.reshape(num_classes, num_classes)

    per_class_iou = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        per_class_iou[c] = (tp / denom) if denom > 0 else 0.0

    miou = float(np.mean(per_class_iou[1:]))
    accuracy = float(np.trace(cm)) / float(cm.sum()) if cm.sum() > 0 else 0.0

    return {
        "per_class_iou": per_class_iou,
        "miou": miou,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "n_ignored": n_ignored,
    }


def validate(model, loader, criterion, device, num_classes=NUM_CLASSES, use_amp=True):
    """Run validation/testing over a DataLoader for PMNet.

    Args:
        model: trained PMNet model (in eval mode).
        loader: DataLoader over UAVScenesDataset.
        criterion: loss function (e.g. CrossEntropyLoss with class weights).
        device: torch device (cuda / cpu).
        num_classes: number of target classes (default 19).
        use_amp: whether to use automatic mixed precision (FP16).

    Returns:
        total_loss: average loss over the dataset.
        metrics: dict with 'miou', 'accuracy', 'per_class_iou', 'confusion_matrix'.
    """
    import torch
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            pc = batch['points_fused'].to(device, non_blocking=True)
            img = batch['image'].to(device, non_blocking=True)
            proj = batch['proj_indices'].to(device, non_blocking=True)
            target = batch['labels_fused'].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred, _ = model(pc, img, proj)
                loss = criterion(pred, target)
            total_loss += loss.item() * pc.size(0)
            all_preds.append(pred.argmax(1).cpu().numpy().flatten())
            all_targets.append(target.cpu().numpy().flatten())
    metrics = compute_metrics(
        np.concatenate(all_preds), np.concatenate(all_targets), num_classes
    )
    return total_loss / max(len(loader.dataset), 1), metrics


# ================================================================
# Full-Cloud Dense Inference & Evaluation (100% Raw LiDAR Points)
# ================================================================
def predict_full_pointcloud(
    model,
    pts_full,
    img_tensor,
    calib,
    device,
    chunk_size=4096,
    img_orig_size=(1920, 1080),
    img_target_size=256,
    in_fov_only=True,
    num_passes=3,
    seed=None,
):
    """Run full-density inference across raw LiDAR points for PMNet using
    Multi-Pass Monte Carlo Random Permutation Voting.

    In each pass, all in-FOV points are randomly permuted and partitioned into
    chunks of size `chunk_size` (default 4096). Because points in each chunk are
    sampled across the full scene rather than localized sequential scanlines,
    PointNet's global max-pooling maintains full scene context. Softmax
    probabilities across all passes are accumulated and voted for final predictions.

    Args:
        model: trained PMNet model (in eval mode).
        pts_full: (M, 3) raw XYZ numpy array.
        img_tensor: (3, H, W) normalized float tensor (256x256).
        calib: calibration dict.
        device: torch device.
        chunk_size: chunk size for point cloud inference (default 4096).
        img_orig_size: (img_w, img_h) native camera resolution tuple.
        img_target_size: image size passed to model (default 256).
        in_fov_only: if True (default), normalizes strictly over the In-FOV frustum
            and evaluates In-FOV points (matching the training pipeline).
        num_passes: number of independent Monte Carlo voting passes (default 3).
        seed: optional random seed for reproducibility.

    Returns:
        pred_labels: (M,) predicted class labels for raw points.
        all_logits: (M, NUM_CLASSES) class logits/probabilities.
        valid_fov: (M,) bool mask of points in camera FOV.
    """
    import torch
    M = len(pts_full)
    img_w, img_h = img_orig_size

    # 1. Perspective projection
    u, v, valid_front = project_lidar_to_image(pts_full, calib)
    valid_fov = (
        valid_front
        & (u >= 0) & (u < img_w)
        & (v >= 0) & (v < img_h)
    )

    all_logits = np.zeros((M, NUM_CLASSES), dtype=np.float32)

    if img_tensor.ndim == 3:
        img_batch = img_tensor.unsqueeze(0).to(device)
    else:
        img_batch = img_tensor.to(device)

    rng = np.random.default_rng(seed)
    n_passes = max(1, int(num_passes))

    if in_fov_only and np.any(valid_fov):
        infov_idx = np.where(valid_fov)[0]
        pts_infov = pts_full[valid_fov]
        u_infov = u[valid_fov]
        v_infov = v[valid_fov]
        M_infov = len(pts_infov)

        col_mapped = (u_infov / float(img_w) * (img_target_size - 1)).astype(np.float32)
        row_mapped = (v_infov / float(img_h) * (img_target_size - 1)).astype(np.float32)
        proj_infov = np.stack([row_mapped, col_mapped], axis=-1).astype(np.float32)

        # In-FOV Frustum Normalization (exact same as uavscenes_dataset.py)
        scene_min = pts_infov.min(axis=0)
        scene_max = pts_infov.max(axis=0)
        scene_range = scene_max - scene_min
        scene_range = np.where(scene_range < 1e-6, 1.0, scene_range)
        pts_norm_infov = ((pts_infov - scene_min) / scene_range).astype(np.float32)

        accum_probs = np.zeros((M_infov, NUM_CLASSES), dtype=np.float32)
        hit_counts = np.zeros(M_infov, dtype=np.int32)

        n_chunks = int(np.ceil(M_infov / float(chunk_size)))
        total_len = n_chunks * chunk_size

        model.eval()
        with torch.no_grad():
            for _ in range(n_passes):
                perm = rng.permutation(M_infov)
                if total_len > M_infov:
                    pad = rng.choice(M_infov, total_len - M_infov, replace=True)
                    eval_indices = np.concatenate([perm, pad])
                else:
                    eval_indices = perm

                eval_indices_chunks = eval_indices.reshape(n_chunks, chunk_size)

                for chunk_idx in eval_indices_chunks:
                    pc_chunk = torch.from_numpy(pts_norm_infov[chunk_idx]).unsqueeze(0).to(device)
                    proj_chunk = torch.from_numpy(proj_infov[chunk_idx]).unsqueeze(0).to(device)

                    logits, _ = model(pc_chunk, img_batch, proj_chunk)
                    probs_np = torch.softmax(logits.squeeze(0), dim=0).permute(1, 0).cpu().numpy()

                    for i_local, idx_orig in enumerate(chunk_idx):
                        accum_probs[idx_orig] += probs_np[i_local]
                        hit_counts[idx_orig] += 1

        infov_logits = accum_probs / np.maximum(hit_counts[:, None], 1)
        all_logits[infov_idx] = infov_logits
    else:
        col_mapped = u / float(img_w) * (img_target_size - 1)
        row_mapped = v / float(img_h) * (img_target_size - 1)
        proj_all = np.stack([row_mapped, col_mapped], axis=-1).astype(np.float32)

        scene_min = pts_full.min(axis=0)
        scene_max = pts_full.max(axis=0)
        scene_range = scene_max - scene_min
        scene_range = np.where(scene_range < 1e-6, 1.0, scene_range)
        pts_norm_all = ((pts_full - scene_min) / scene_range).astype(np.float32)

        accum_probs = np.zeros((M, NUM_CLASSES), dtype=np.float32)
        hit_counts = np.zeros(M, dtype=np.int32)

        n_chunks = int(np.ceil(M / float(chunk_size)))
        total_len = n_chunks * chunk_size

        model.eval()
        with torch.no_grad():
            for _ in range(n_passes):
                perm = rng.permutation(M)
                if total_len > M:
                    pad = rng.choice(M, total_len - M, replace=True)
                    eval_indices = np.concatenate([perm, pad])
                else:
                    eval_indices = perm

                eval_indices_chunks = eval_indices.reshape(n_chunks, chunk_size)

                for chunk_idx in eval_indices_chunks:
                    pc_chunk = torch.from_numpy(pts_norm_all[chunk_idx]).unsqueeze(0).to(device)
                    proj_chunk = torch.from_numpy(proj_all[chunk_idx]).unsqueeze(0).to(device)

                    logits, _ = model(pc_chunk, img_batch, proj_chunk)
                    probs_np = torch.softmax(logits.squeeze(0), dim=0).permute(1, 0).cpu().numpy()

                    for i_local, idx_orig in enumerate(chunk_idx):
                        accum_probs[idx_orig] += probs_np[i_local]
                        hit_counts[idx_orig] += 1

        all_logits = accum_probs / np.maximum(hit_counts[:, None], 1)

    pred_labels = np.argmax(all_logits, axis=-1).astype(np.int64)
    return pred_labels, all_logits, valid_fov


def predict_full_pointcloud_knn(
    sampled_pts,
    sampled_logits,
    pts_full,
    k_neighbors=3,
):
    """Fast KNN Inverse-Distance Interpolation of predicted class probabilities
    from sampled working points to the full raw point cloud.

    Args:
        sampled_pts: (N, 3) XYZ coordinates of sampled points.
        sampled_logits: (N, C) logits of sampled points.
        pts_full: (M, 3) full raw XYZ coordinates.
        k_neighbors: number of nearest neighbors (default 3).

    Returns:
        pred_labels: (M,) predicted class labels for all M raw points.
        full_logits: (M, C) interpolated class logits.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(sampled_pts[:, :3])
    dists, indices = tree.query(pts_full[:, :3], k=k_neighbors)

    if k_neighbors == 1 or dists.ndim == 1:
        full_logits = sampled_logits[indices]
    else:
        weights = 1.0 / np.maximum(dists, 1e-6)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        full_logits = np.sum(weights[:, :, None] * sampled_logits[indices], axis=1)

    pred_labels = np.argmax(full_logits, axis=-1).astype(np.int64)
    return pred_labels, full_logits


def evaluate_full_pointcloud_dataset(
    model,
    dataset,
    device,
    num_classes=NUM_CLASSES,
    chunk_size=4096,
    max_frames=None,
    verbose=True,
    in_fov_only=True,
    num_passes=3,
):
    """Evaluates raw LiDAR points across all frames in a dataset split for PMNet.

    Args:
        model: trained PMNet model (in eval mode).
        dataset: UAVScenesDataset instance.
        device: torch device.
        num_classes: number of target classes (9).
        chunk_size: chunk size for point cloud inference (4096).
        max_frames: optional limit on frames evaluated (None for full split).
        verbose: whether to print progress.
        in_fov_only: if True (default), evaluates only in-camera-FOV points with
            proper frustum-aligned normalization and chunking.

    Returns:
        metrics: dict with 'miou', 'accuracy', 'per_class_iou', 'confusion_matrix', 'total_points_evaluated'.
    """
    import time
    import torch
    from PIL import Image

    model.eval()
    total_pts = 0
    t0 = time.time()

    frames = dataset.frames
    if max_frames is not None:
        frames = frames[:max_frames]

    n_frames = len(frames)
    target_str = "In-Camera-FOV" if in_fov_only else "100% full-cloud"
    if verbose:
        print(f">> Starting PMNet Dense Inference on {n_frames} frames ({target_str} raw LiDAR points)...")

    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    cm_infov = np.zeros((num_classes, num_classes), dtype=np.int64)
    cm_outfov = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i, (lidar_path, cam_path, label_path, seq_name) in enumerate(frames):
        pts_raw = np.load(lidar_path).astype(np.float32)
        pts_full = pts_raw[:, :3]
        labels_26 = np.load(label_path).astype(np.int64)
        labels_full = map_labels_26_to_19(labels_26)

        with Image.open(cam_path) as img_pil:
            img_w, img_h = img_pil.size
            img_resized = np.array(img_pil.resize((dataset.img_size, dataset.img_size), Image.BILINEAR))
        img_norm = torch.from_numpy(img_resized.astype(np.float32) / 255.0).permute(2, 0, 1)

        calib = dataset.calibrations[seq_name]

        preds_full, _, valid_fov = predict_full_pointcloud(
            model=model,
            pts_full=pts_full,
            img_tensor=img_norm,
            calib=calib,
            device=device,
            chunk_size=chunk_size,
            img_orig_size=(img_w, img_h),
            img_target_size=dataset.img_size,
            in_fov_only=in_fov_only,
            num_passes=num_passes,
            seed=i,
        )

        keep = labels_full != IGNORE_INDEX
        p_valid = preds_full[keep]
        t_valid = labels_full[keep]
        v_mask = valid_fov[keep]

        # Overall
        flat = t_valid * num_classes + p_valid
        cm_total += np.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

        # In-FOV
        if np.any(v_mask):
            flat_in = t_valid[v_mask] * num_classes + p_valid[v_mask]
            cm_infov += np.bincount(flat_in, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

        # Out-of-FOV
        if np.any(~v_mask):
            flat_out = t_valid[~v_mask] * num_classes + p_valid[~v_mask]
            cm_outfov += np.bincount(flat_out, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

        total_pts += int(v_mask.sum()) if in_fov_only else len(p_valid)

        if verbose and ((i + 1) % 100 == 0 or (i + 1) == n_frames):
            elapsed = time.time() - t0
            pts_per_sec = total_pts / max(elapsed, 1e-4)
            print(f"   [{i+1:4d}/{n_frames:4d}] frames evaluated | {total_pts:,} {target_str} points | {pts_per_sec:.0f} pts/sec")

    def _calc_metrics(cm):
        ious = np.zeros(num_classes, dtype=np.float64)
        for c in range(num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            denom = tp + fp + fn
            ious[c] = (tp / denom) if denom > 0 else 0.0
        m = float(np.mean(ious[1:]))
        a = float(np.trace(cm)) / float(cm.sum()) if cm.sum() > 0 else 0.0
        return m, a, ious, int(cm.sum())

    miou_all, acc_all, per_class_iou_all, n_all = _calc_metrics(cm_total)
    miou_in, acc_in, per_class_iou_in, n_in = _calc_metrics(cm_infov)
    miou_out, acc_out, per_class_iou_out, n_out = _calc_metrics(cm_outfov)

    if in_fov_only:
        primary_miou = miou_in
        primary_acc = acc_in
        primary_iou = per_class_iou_in
        primary_cm = cm_infov
        primary_pts = n_in
    else:
        primary_miou = miou_all
        primary_acc = acc_all
        primary_iou = per_class_iou_all
        primary_cm = cm_total
        primary_pts = n_all

    return {
        "miou": primary_miou,
        "accuracy": primary_acc,
        "per_class_iou": primary_iou,
        "confusion_matrix": primary_cm,
        "total_points_evaluated": primary_pts,
        "elapsed_seconds": time.time() - t0,
        "in_fov": {
            "miou": miou_in,
            "accuracy": acc_in,
            "per_class_iou": per_class_iou_in,
            "total_points": n_in,
            "ratio": n_in / max(n_all, 1),
        },
        "out_fov": {
            "miou": miou_out,
            "accuracy": acc_out,
            "per_class_iou": per_class_iou_out,
            "total_points": n_out,
            "ratio": n_out / max(n_all, 1),
        },
    }

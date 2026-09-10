"""
model_pmnet_torch.py — Faithful PyTorch port of PMNet.

CHANGE FROM THE PREVIOUS VERSION: `forward()` now asserts the incoming
point cloud's feature dimension matches `feature_num` before running any
layers. Previously, a raw-column mismatch upstream (see
uavscenes_dataset.py's fix #4) would have surfaced as a cryptic shape
error somewhere inside the Conv1d stack instead of a clear message at
the model boundary.

Original: model_pmnet.py (TensorFlow 1.x)
Paper:    "A Point-Wise LiDAR and Image Multimodal Fusion Network (PMNet)
           for Aerial Point Cloud 3D Semantic Segmentation"
           — Remote Sensing, 2019.

Architecture overview:
  +------------+   +----------------+
  | 2-D Image  |   | 3-D Point Cloud|
  +-----+------+   +------+---------+
        |                 |
   U-Net CNN         PointNet-like
   (ReLU act.)       (ELU act.)
        |                 |
   feature_img       local+global
   [B,128,H,W]       [B,128,N]
        |                 |
        +--> get_point_   |
             corres_      |
             features ----+
                  |
             Concat (256)
                  |
           Conv1d -> 256 -> 128 -> Dropout -> num_cls
                  |
             Predictions
             [B, C, N]
"""

import torch
import torch.nn as nn


class PMNet(nn.Module):
    """PMNet: Point-wise Multimodal fusion Network.

    Args:
        num_classes: number of semantic classes (default 19 for the
                     19-class UAVScenes scheme with Background as class 0).
        feature_num: per-point input features (default 3 for XYZ).
        dropout_rate: probability of dropout (default 0.4,
                      matching original keep_prob=0.6).
    """

    def __init__(self, num_classes=19, feature_num=3, dropout_rate=0.4):
        super().__init__()
        self.num_classes = num_classes
        self.feature_num = feature_num

        # =============================================================
        # Image Stream — U-Net-like encoder-decoder (ReLU activations)
        # =============================================================
        self.img_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.img_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.img_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.img_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.img_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.img_conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        self.img_up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.img_dconv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.img_dconv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        self.img_up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.img_dconv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.img_feature = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        # =============================================================
        # Point Stream — PointNet-like (ELU activations)
        # =============================================================
        self.pt_conv1 = nn.Sequential(
            nn.Conv1d(feature_num, 64, 1), nn.ELU(inplace=True))
        self.pt_conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.ELU(inplace=True))
        self.pt_conv3 = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.ELU(inplace=True))
        self.pt_conv4 = nn.Sequential(
            nn.Conv1d(128, 1024, 1), nn.ELU(inplace=True))
        self.pt_conv5 = nn.Sequential(
            nn.Conv1d(1152, 128, 1), nn.ELU(inplace=True))

        # =============================================================
        # Fusion + Classification Head (ELU activations)
        # =============================================================
        self.fuse_conv1 = nn.Sequential(
            nn.Conv1d(256, 256, 1), nn.ELU(inplace=True))
        self.fuse_conv2 = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.ELU(inplace=True))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fuse_conv3 = nn.Conv1d(128, num_classes, 1)

    # ----------------------------------------------------------------- #
    @staticmethod
    def get_point_corres_features(feature_img, proj_indices):
        """Look up 2-D image features for each 3-D point."""
        B, C, H, W = feature_img.shape
        N = proj_indices.shape[1]

        row = proj_indices[:, :, 0].long().clamp(0, H - 1)
        col = proj_indices[:, :, 1].long().clamp(0, W - 1)

        feat_flat = feature_img.reshape(B, C, H * W)
        linear_idx = row * W + col
        linear_idx = linear_idx.unsqueeze(1).expand(B, C, N)

        return torch.gather(feat_flat, 2, linear_idx)

    # ----------------------------------------------------------------- #
    def forward(self, point_cloud, img, proj_indices):
        """Full forward pass.

        Args:
            point_cloud:  (B, N, feature_num) — per-point features (XYZ …).
            img:          (B, 3, H, W)        — camera image crop.
            proj_indices: (B, N, 2)           — (row, col) in the image.

        Returns:
            pred:         (B, num_classes, N) — per-point class logits.
            feat_256:     (B, 256, N)         — intermediate features.
        """
        if point_cloud.shape[-1] != self.feature_num:
            raise ValueError(
                f"PMNet was built with feature_num={self.feature_num}, but "
                f"received a point cloud with {point_cloud.shape[-1]} "
                f"features. This is exactly the failure mode that used to "
                f"happen silently deeper in the Conv1d stack -- fix the "
                f"upstream dataset/feature_num mismatch rather than "
                f"changing this check."
            )

        # ============ Image Stream ====================================
        c1 = self.img_conv1(img)
        p1 = self.img_pool1(c1)
        c2 = self.img_conv2(p1)
        p2 = self.img_pool2(c2)
        c3 = self.img_conv3(p2)
        c4 = self.img_conv4(c3)

        u1 = self.img_up1(c4)
        d1 = self.img_dconv1(u1)
        d2 = self.img_dconv2(torch.cat([d1, c2], 1))
        u2 = self.img_up2(d2)
        d3 = self.img_dconv3(u2)
        feature_img = self.img_feature(torch.cat([d3, c1], 1))

        # ============ Point Stream ====================================
        pc = point_cloud.permute(0, 2, 1)

        net = self.pt_conv1(pc)
        net = self.pt_conv2(net)
        local_feat = self.pt_conv3(net)
        global_feat = self.pt_conv4(local_feat)

        global_feat = global_feat.max(dim=2, keepdim=True).values
        global_feat = global_feat.expand(-1, -1, local_feat.size(2))

        combined = torch.cat([local_feat, global_feat], dim=1)
        pt_net = self.pt_conv5(combined)

        # ============ Fusion ==========================================
        img_feats = self.get_point_corres_features(feature_img, proj_indices)

        fused = torch.cat([img_feats, pt_net], dim=1)

        # ============ Classification ==================================
        feat_256 = self.fuse_conv1(fused)
        out = self.fuse_conv2(feat_256)
        out = self.dropout(out)
        pred = self.fuse_conv3(out)

        return pred, feat_256

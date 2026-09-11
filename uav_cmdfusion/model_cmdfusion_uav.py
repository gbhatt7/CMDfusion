"""
model_cmdfusion_uav.py — Self-contained CMDFusion model for UAVScenes.

All CMDFusion architecture components are included inline to avoid the
PyTorch Lightning dependency chain in the original network/ folder.

Architecture:
  3D Branch (SPVCNN-G):
    voxelization → voxel_3d_generator_g → multi-scale SPVBlock (L + C branches)
  2D Branch (ResNet-50 FCN):
    ResNet-50 encoder → multi-scale feature maps
  UpsampleNet:
    Upsamples 2D feature maps to target image resolution for point-pixel lookup
  CMDFuse:
    Bidirectional cross-modal fusion + cross-modal knowledge distillation

Original CMDFusion paper:
  "CMDFusion: Bidirectional Fusion Network with Cross-modality Knowledge
   Distillation for LiDAR Semantic Segmentation" — ICRA 2024
  Authors: Jun Cen et al.

Components adapted from:
  network/voxel_fea_generator.py, network/spvcnn_g.py,
  network/basic_block.py, network/arch_cmd_fusion.py
"""

import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from torchvision.models.resnet import resnet34, resnet50
from torchvision import transforms

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int = None) -> torch.Tensor:
    """PyTorch native implementation of torch_scatter.scatter_mean"""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    
    index_expanded = index
    for _ in range(src.dim() - index.dim()):
        index_expanded = index_expanded.unsqueeze(-1)
    index_expanded = index_expanded.expand_as(src)
    
    out = src.new_zeros(out_shape)
    out.scatter_add_(dim, index_expanded, src)
    
    count = src.new_zeros(out_shape)
    count.scatter_add_(dim, index_expanded, torch.ones_like(src))
    count.clamp_(min=1)
    
    return out / count

# Import lovasz_softmax locally
from lovasz_loss import lovasz_softmax


# ====================================================================
# Sparse 3D Convolution Block (from basic_block.py)
# ====================================================================
class SparseBasicBlock(spconv.SparseModule):
    """Residual sparse 3D convolution block."""
    def __init__(self, in_channels, out_channels, indice_key):
        super(SparseBasicBlock, self).__init__()
        self.layers_in = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 1,
                              indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3,
                              indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1),
            spconv.SubMConv3d(out_channels, out_channels, 3,
                              indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        identity = self.layers_in(x)
        output = self.layers(x)
        return output.replace_feature(
            F.leaky_relu(output.features + identity.features, 0.1)
        )


# ====================================================================
# Lovasz Loss (from basic_block.py)
# ====================================================================
class Lovasz_loss(nn.Module):
    def __init__(self, ignore=None):
        super(Lovasz_loss, self).__init__()
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, ignore=self.ignore)


# ====================================================================
# Voxelization (from voxel_fea_generator.py)
# ====================================================================
class Voxelization(nn.Module):
    """Quantizes 3D points into multi-scale voxel grids."""
    def __init__(self, coors_range_xyz, spatial_shape, scale_list):
        super(Voxelization, self).__init__()
        self.spatial_shape = spatial_shape
        self.scale_list = scale_list + [1]
        self.coors_range_xyz = coors_range_xyz

    @staticmethod
    def sparse_quantize(pc, coors_range, spatial_shape):
        idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
        return idx.long()

    def forward(self, data_dict):
        pc = data_dict['points'][:, :3]
        for idx, scale in enumerate(self.scale_list):
            xidx = self.sparse_quantize(
                pc[:, 0], self.coors_range_xyz[0],
                np.ceil(self.spatial_shape[0] / scale))
            yidx = self.sparse_quantize(
                pc[:, 1], self.coors_range_xyz[1],
                np.ceil(self.spatial_shape[1] / scale))
            zidx = self.sparse_quantize(
                pc[:, 2], self.coors_range_xyz[2],
                np.ceil(self.spatial_shape[2] / scale))

            bxyz_indx = torch.stack(
                [data_dict['batch_idx'], xidx, yidx, zidx], dim=-1
            ).long()
            unq, unq_inv, unq_cnt = torch.unique(
                bxyz_indx, return_inverse=True, return_counts=True, dim=0
            )
            unq = torch.cat([unq[:, 0:1], unq[:, [3, 2, 1]]], dim=1)
            data_dict['scale_{}'.format(scale)] = {
                'full_coors': bxyz_indx,
                'coors_inv': unq_inv,
                'coors': unq.type(torch.int32)
            }
        return data_dict


# ====================================================================
# Voxel 3D Feature Generator — dual-branch (from voxel_fea_generator.py)
# ====================================================================
class Voxel3DGeneratorG(nn.Module):
    """Generates initial sparse 3D features for L and C branches."""
    def __init__(self, in_channels, out_channels, coors_range_xyz, spatial_shape):
        super(Voxel3DGeneratorG, self).__init__()
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = coors_range_xyz
        self.PPmodel = nn.Sequential(
            nn.Linear(in_channels + 6, out_channels),
            nn.ReLU(True),
            nn.Linear(out_channels, out_channels)
        )

    def prepare_input(self, point, grid_ind, inv_idx):
        pc_mean = scatter_mean(
            point[:, :3], inv_idx, dim=0
        )[inv_idx]
        nor_pc = point[:, :3] - pc_mean

        coors_range_xyz = torch.Tensor(self.coors_range_xyz)
        cur_grid_size = torch.Tensor(self.spatial_shape)
        crop_range = coors_range_xyz[:, 1] - coors_range_xyz[:, 0]
        intervals = (crop_range / cur_grid_size).to(point.device)
        voxel_centers = grid_ind * intervals + coors_range_xyz[:, 0].to(point.device)
        center_to_point = point[:, :3] - voxel_centers

        pc_feature = torch.cat((point, nor_pc, center_to_point), dim=1)
        return pc_feature

    def forward(self, data_dict):
        pt_fea = self.prepare_input(
            data_dict['points'],
            data_dict['scale_1']['full_coors'][:, 1:],
            data_dict['scale_1']['coors_inv']
        )
        pt_fea = self.PPmodel(pt_fea)

        features = scatter_mean(
            pt_fea, data_dict['scale_1']['coors_inv'], dim=0
        )
        spatial_shape_rev = np.int32(self.spatial_shape)[::-1].tolist()

        # L branch (LiDAR)
        data_dict['sparse_tensor_L'] = spconv.SparseConvTensor(
            features=features,
            indices=data_dict['scale_1']['coors'].int(),
            spatial_shape=spatial_shape_rev,
            batch_size=data_dict['batch_size']
        )
        # C branch (Camera/cross-modal)
        data_dict['sparse_tensor_C'] = spconv.SparseConvTensor(
            features=features.clone(),
            indices=data_dict['scale_1']['coors'].int(),
            spatial_shape=spatial_shape_rev,
            batch_size=data_dict['batch_size']
        )

        data_dict['coors_L'] = data_dict['scale_1']['coors'].clone()
        data_dict['coors_inv_L'] = data_dict['scale_1']['coors_inv'].clone()
        data_dict['full_coors_L'] = data_dict['scale_1']['full_coors'].clone()

        data_dict['coors_C'] = data_dict['scale_1']['coors'].clone()
        data_dict['coors_inv_C'] = data_dict['scale_1']['coors_inv'].clone()
        data_dict['full_coors_C'] = data_dict['scale_1']['full_coors'].clone()

        return data_dict


# ====================================================================
# Point Encoder (from spvcnn_g.py)
# ====================================================================
class PointEncoder(nn.Module):
    """Point-wise feature encoder with downsampling and identity skip."""
    def __init__(self, in_channels, out_channels, scale, modality):
        super(PointEncoder, self).__init__()
        self.scale = scale
        self.modality = modality
        self.layer_in = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(0.1, True),
        )
        self.PPmodel = nn.Sequential(
            nn.Linear(in_channels, out_channels // 2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(out_channels // 2),
            nn.Linear(out_channels // 2, out_channels // 2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(out_channels // 2),
            nn.Linear(out_channels // 2, out_channels),
            nn.LeakyReLU(0.1, True),
        )
        self.layer_out = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.LeakyReLU(0.1, True),
            nn.Linear(out_channels, out_channels)
        )

    @staticmethod
    def downsample(coors, p_fea, scale=2):
        batch = coors[:, 0:1]
        coors = coors[:, 1:] // scale
        inv = torch.unique(
            torch.cat([batch, coors], 1), return_inverse=True, dim=0
        )[1]
        return scatter_mean(p_fea, inv, dim=0), inv

    def forward(self, features, data_dict):
        output, inv = self.downsample(
            data_dict['coors_{}'.format(self.modality)], features
        )
        identity = self.layer_in(features)
        output = self.PPmodel(output)[inv]
        output = torch.cat([identity, output], dim=1)

        v_feat = scatter_mean(
            self.layer_out(
                output[data_dict['coors_inv_{}'.format(self.modality)]]
            ),
            data_dict['scale_{}'.format(self.scale)]['coors_inv'],
            dim=0
        )
        data_dict['coors_{}'.format(self.modality)] = \
            data_dict['scale_{}'.format(self.scale)]['coors']
        data_dict['coors_inv_{}'.format(self.modality)] = \
            data_dict['scale_{}'.format(self.scale)]['coors_inv']
        data_dict['full_coors_{}'.format(self.modality)] = \
            data_dict['scale_{}'.format(self.scale)]['full_coors']

        return v_feat


# ====================================================================
# SPV Block — Sparse Point-Voxel block (from spvcnn_g.py)
# ====================================================================
class SPVBlock(nn.Module):
    """Sparse Point-Voxel encoding block with dual voxel + point encoders."""
    def __init__(self, in_channels, out_channels, indice_key,
                 scale, last_scale, spatial_shape, modality):
        super(SPVBlock, self).__init__()
        self.scale = scale
        self.indice_key = indice_key
        self.layer_id = indice_key.split('_')[1]
        self.last_scale = last_scale
        self.spatial_shape = spatial_shape
        self.modality = modality
        self.v_enc = spconv.SparseSequential(
            SparseBasicBlock(in_channels, out_channels, self.indice_key),
            SparseBasicBlock(out_channels, out_channels, self.indice_key),
        )
        self.p_enc = PointEncoder(in_channels, out_channels, scale, modality)

    def forward(self, data_dict):
        coors_inv_last = data_dict['scale_{}'.format(self.last_scale)]['coors_inv']
        coors_inv = data_dict['scale_{}'.format(self.scale)]['coors_inv']

        # Voxel encoder
        v_fea = self.v_enc(
            data_dict['sparse_tensor_{}'.format(self.modality)]
        )
        layer_key = 'layer_{}_{}'.format(self.layer_id, self.modality)
        data_dict[layer_key] = {}
        data_dict[layer_key]['pts_feat'] = v_fea.features
        data_dict[layer_key]['full_coors'] = \
            data_dict['full_coors_{}'.format(self.modality)]
        v_fea_inv = scatter_mean(
            v_fea.features[coors_inv_last], coors_inv, dim=0
        )

        # Point encoder
        p_fea = self.p_enc(
            features=(
                data_dict['sparse_tensor_{}'.format(self.modality)].features
                + v_fea.features
            ),
            data_dict=data_dict
        )
        data_dict[layer_key]['pts_feat_f'] = p_fea[coors_inv]

        # Fusion and update sparse tensor
        data_dict['sparse_tensor_{}'.format(self.modality)] = \
            spconv.SparseConvTensor(
                features=p_fea + v_fea_inv,
                indices=data_dict['coors_{}'.format(self.modality)],
                spatial_shape=self.spatial_shape,
                batch_size=data_dict['batch_size']
            )

        return p_fea[coors_inv]


# ====================================================================
# ResNet FCN — 2D image feature extractor (from basic_block.py)
# ====================================================================
class ResNetFCN(nn.Module):
    """ResNet-based 2D feature extractor. Returns multi-scale features."""
    def __init__(self, backbone="resnet50", pretrained=True):
        super(ResNetFCN, self).__init__()

        if backbone == "resnet34":
            net = resnet34(pretrained=pretrained)
        elif backbone == "resnet50":
            net = resnet50(pretrained=pretrained)
        else:
            raise NotImplementedError(
                "invalid backbone: {}".format(backbone)
            )

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.conv1.weight.data = net.conv1.weight.data
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, data_dict):
        x = data_dict['img']
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        layer1_out = self.layer1(self.maxpool(conv1_out))
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        return layer1_out, layer2_out, layer3_out, layer4_out


# ====================================================================
# UpsampleNet for UAVScenes (adapted from basic_block.py UpsampleNet_2)
# ====================================================================
class UpsampleNetUAV(nn.Module):
    """Upsamples multi-scale 2D features to the target image size for
    point-pixel feature correspondence.

    For ResNet-50 backbone, input channels per layer are:
      layer1=256, layer2=512, layer3=1024, layer4=2048
    """
    def __init__(self, target_size, hiden_size=128):
        super(UpsampleNetUAV, self).__init__()
        # target_size = [H, W] for the feature map (must match img_indices)
        self.up = nn.Upsample(size=target_size, mode='bilinear',
                               align_corners=False)
        # Channel reduction from ResNet-50 layer sizes
        self.down0 = nn.Conv2d(256, hiden_size, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.down1 = nn.Conv2d(512, hiden_size, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.down2 = nn.Conv2d(1024, hiden_size, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.down3 = nn.Conv2d(2048, hiden_size, kernel_size=1, stride=1,
                               padding=0, bias=False)

    def forward(self, x, data_dict):
        fea_down0 = self.down0(x[0])
        fea_down1 = self.down1(x[1])
        fea_down2 = self.down2(x[2])
        fea_down3 = self.down3(x[3])

        fea_up0 = self.up(fea_down0)
        fea_up1 = self.up(fea_down1)
        fea_up2 = self.up(fea_down2)
        fea_up3 = self.up(fea_down3)

        data_dict['img_scale2'] = fea_up0
        data_dict['img_scale4'] = fea_up1
        data_dict['img_scale8'] = fea_up2
        data_dict['img_scale16'] = fea_up3

        return data_dict


# ====================================================================
# CMDFuse — Bidirectional Cross-Modal Fusion (from arch_cmd_fusion.py)
# ====================================================================
class CMDFuse(nn.Module):
    """CMDFusion bidirectional fusion module.

    For each scale:
      Direction 1 (2D → 3D): Image knowledge enhances 3D features
      Direction 2 (3D → 2D): 3D features enhance image knowledge
      + Cross-modal knowledge distillation (MSE loss)
    """
    def __init__(self, hiden_size, scale_list, num_classes,
                 lambda_xm=0.05, lambda_seg2d=4.0,
                 seg_labelweights=None, ignore_label=0):
        super(CMDFuse, self).__init__()
        self.hiden_size = hiden_size
        self.scale_list = scale_list
        self.num_classes = num_classes
        self.lambda_xm = lambda_xm
        self.lambda_seg2d = lambda_seg2d
        self.num_scales = len(scale_list)

        # Per-scale 3D classifiers
        self.multihead_3d_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_3d_classifier.append(nn.Sequential(
                nn.Linear(hiden_size, 128),
                nn.ReLU(True),
                nn.Linear(128, num_classes)
            ))

        # Per-scale fusion classifiers (direction 1: img-side)
        self.multihead_fuse_classifier = nn.ModuleList()
        # Per-scale fusion classifiers (direction 2: pts-side)
        self.multihead_fuse_classifier_2 = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_fuse_classifier.append(nn.Sequential(
                nn.Linear(hiden_size, 128),
                nn.ReLU(True),
                nn.Linear(128, num_classes)
            ))
            self.multihead_fuse_classifier_2.append(nn.Sequential(
                nn.Linear(hiden_size, 128),
                nn.ReLU(True),
                nn.Linear(128, num_classes)
            ))

        # Learnable fusion layers
        self.leaners = nn.ModuleList()
        self.leaners_2 = nn.ModuleList()
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()
        self.fcs1_2 = nn.ModuleList()
        self.fcs2_2 = nn.ModuleList()
        for i in range(self.num_scales):
            self.leaners.append(nn.Sequential(
                nn.Linear(hiden_size, hiden_size)))
            self.leaners_2.append(nn.Sequential(
                nn.Linear(hiden_size, hiden_size)))
            self.fcs1.append(nn.Sequential(
                nn.Linear(hiden_size * 2, hiden_size)))
            self.fcs2.append(nn.Sequential(
                nn.Linear(hiden_size * 2, hiden_size)))
            self.fcs1_2.append(nn.Sequential(
                nn.Linear(hiden_size * 2, hiden_size)))
            self.fcs2_2.append(nn.Sequential(
                nn.Linear(hiden_size * 2, hiden_size)))

        # Final multi-scale classifiers
        self.classifier = nn.Sequential(
            nn.Linear(hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes),
        )
        self.classifier_2 = nn.Sequential(
            nn.Linear(hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes),
        )

        # Loss functions
        if seg_labelweights is not None:
            seg_labelweights = torch.Tensor(seg_labelweights)
        self.ce_loss = nn.CrossEntropyLoss(
            weight=seg_labelweights, ignore_index=ignore_label
        )
        self.lovasz_loss = Lovasz_loss(ignore=ignore_label)

    @staticmethod
    def p2img_mapping(pts_fea, p2img_idx, batch_idx):
        """Map point features to image-visible subset per batch."""
        img_feat = []
        for b in range(batch_idx.max() + 1):
            img_feat.append(pts_fea[batch_idx == b][p2img_idx[b]])
        return torch.cat(img_feat, 0)

    @staticmethod
    def voxelize_labels(labels, full_coors):
        """Voxelize labels by majority vote."""
        lbxyz = torch.cat([labels.reshape(-1, 1), full_coors], dim=-1)
        unq_lbxyz, count = torch.unique(lbxyz, return_counts=True, dim=0)
        inv_ind = torch.unique(
            unq_lbxyz[:, 1:], return_inverse=True, dim=0
        )[1]
        
        if len(count) == 0:
            return labels.new_empty(0)

        max_count = count.max().item()
        combined_key = inv_ind.to(torch.int64) * (max_count + 1) + count.to(torch.int64)
        sort_idx = torch.argsort(combined_key)
        sorted_inv_ind = inv_ind[sort_idx]
        
        is_last = torch.ones(len(sorted_inv_ind), dtype=torch.bool, device=count.device)
        is_last[:-1] = sorted_inv_ind[1:] != sorted_inv_ind[:-1]
        label_ind = sort_idx[is_last]
        
        labels = unq_lbxyz[:, 0][label_ind]
        return labels

    def seg_loss(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        lovasz_loss = self.lovasz_loss(
            F.softmax(logits, dim=1), labels
        )
        return ce_loss + lovasz_loss

    def forward(self, data_dict):
        loss = 0
        img_seg_feat = []
        pts_seg_feat = []
        batch_idx = data_dict['batch_idx']
        point2img_index = data_dict['point2img_index']

        for idx, scale in enumerate(self.scale_list):
            img_feat = data_dict['img_scale{}'.format(scale)]
            pts_feat = data_dict['layer_{}_L'.format(idx)]['pts_feat']
            pts_feat_f = data_dict['layer_{}_L'.format(idx)]['pts_feat_f']
            coors_inv = data_dict['scale_{}'.format(scale)]['coors_inv']
            g_img_feat = data_dict['layer_{}_C'.format(idx)]['pts_feat_f']
            g_img_feat_p = self.p2img_mapping(
                g_img_feat, point2img_index, batch_idx
            )

            # 3D prediction at this scale
            pts_pred_full = self.multihead_3d_classifier[idx](pts_feat)

            # Voxelize labels for 3D loss
            pts_label_full = self.voxelize_labels(
                data_dict['labels'],
                data_dict['layer_{}_L'.format(idx)]['full_coors']
            )
            pts_feat_mapped = self.p2img_mapping(
                pts_feat[coors_inv], point2img_index, batch_idx
            )
            pts_feat_f_p = self.p2img_mapping(
                pts_feat_f, point2img_index, batch_idx
            )

            # ---- Direction 1: 2D → 3D (image teaches 3D) ----
            feat_learner = F.relu(self.leaners[idx](pts_feat_f))
            feat_cat = torch.cat([g_img_feat, feat_learner], 1)
            feat_cat = self.fcs1[idx](feat_cat)
            feat_weight = torch.sigmoid(
                self.fcs2[idx](torch.cat(
                    (feat_cat,
                     feat_cat.mean(0).unsqueeze(0).expand(feat_cat.shape)),
                    -1
                ))
            )
            feat_cat = F.relu(feat_cat * feat_weight)
            fuse_pred = feat_cat + g_img_feat
            img_seg_feat.append(fuse_pred)
            fuse_pred = self.multihead_fuse_classifier[idx](fuse_pred)
            data_dict['fuse_g_img_scale{}'.format(scale)] = fuse_pred

            # ---- Direction 2: 3D → 2D (3D teaches image) ----
            feat_learner_2 = F.relu(self.leaners_2[idx](g_img_feat))
            feat_cat_2 = torch.cat([pts_feat_f, feat_learner_2], 1)
            feat_cat_2 = self.fcs1_2[idx](feat_cat_2)
            feat_weight_2 = torch.sigmoid(
                self.fcs2_2[idx](torch.cat(
                    (feat_cat_2,
                     feat_cat_2.mean(0).unsqueeze(0).expand(feat_cat.shape)),
                    -1
                ))
            )
            feat_cat_2 = F.relu(feat_cat_2 * feat_weight_2)
            fuse_pred_2 = feat_cat_2 + pts_feat_f
            pts_seg_feat.append(fuse_pred_2)
            fuse_pred_2 = self.multihead_fuse_classifier_2[idx](fuse_pred_2)
            data_dict['fuse_pts_scale{}'.format(scale)] = fuse_pred_2

            # ---- Cross-modal knowledge distillation loss ----
            mse_loss = nn.MSELoss()
            g_loss = mse_loss(g_img_feat_p, img_feat)
            data_dict['g_loss'] = g_loss
            loss += g_loss * self.lambda_seg2d / self.num_scales

        # Multi-scale aggregated predictions
        img_seg_logits = self.classifier(torch.cat(img_seg_feat, 1))
        data_dict['fuse_img_scale_all'] = img_seg_logits

        pts_seg_logits = self.classifier_2(torch.cat(pts_seg_feat, 1))
        data_dict['fuse_pts_scale_all'] = pts_seg_logits
        loss += self.seg_loss(pts_seg_logits, data_dict['labels'])

        data_dict['loss'] += loss
        return data_dict


# ====================================================================
# SPVCNN3D — Standalone 3D backbone (from spvcnn_g.py, no Lightning)
# ====================================================================
class SPVCNN3D(nn.Module):
    """Standalone SPVCNN-G 3D backbone for CMDFusion.

    Dual-branch (L for LiDAR, C for Camera/cross-modal) multi-scale
    sparse point-voxel convolution network.
    """
    def __init__(self, input_dims, hiden_size, num_classes, scale_list,
                 coors_range_xyz, spatial_shape):
        super(SPVCNN3D, self).__init__()
        self.input_dims = input_dims
        self.hiden_size = hiden_size
        self.num_classes = num_classes
        self.scale_list = scale_list
        self.num_scales = len(scale_list)
        self.coors_range_xyz = coors_range_xyz
        self.spatial_shape = np.array(spatial_shape)
        self.strides = [int(s / scale_list[0]) for s in scale_list]

        # Voxelization
        self.voxelizer = Voxelization(
            coors_range_xyz=coors_range_xyz,
            spatial_shape=self.spatial_shape,
            scale_list=scale_list
        )

        # Input processing
        self.voxel_3d_generator = Voxel3DGeneratorG(
            in_channels=input_dims,
            out_channels=hiden_size,
            coors_range_xyz=coors_range_xyz,
            spatial_shape=self.spatial_shape
        )

        # L branch encoder layers
        self.spv_enc_L = nn.ModuleList()
        for i in range(self.num_scales):
            self.spv_enc_L.append(SPVBlock(
                in_channels=hiden_size,
                out_channels=hiden_size,
                indice_key='spv_' + str(i),
                scale=scale_list[i],
                last_scale=scale_list[i - 1] if i > 0 else 1,
                spatial_shape=np.int32(
                    self.spatial_shape // self.strides[i]
                )[::-1].tolist(),
                modality='L'
            ))

        # C branch encoder layers
        self.spv_enc_C = nn.ModuleList()
        for i in range(self.num_scales):
            self.spv_enc_C.append(SPVBlock(
                in_channels=hiden_size,
                out_channels=hiden_size,
                indice_key='spv_' + str(i),
                scale=scale_list[i],
                last_scale=scale_list[i - 1] if i > 0 else 1,
                spatial_shape=np.int32(
                    self.spatial_shape // self.strides[i]
                )[::-1].tolist(),
                modality='C'
            ))

        # Classifier for baseline (3D-only) logits
        self.classifier = nn.Sequential(
            nn.Linear(hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes),
        )

    def forward(self, data_dict):
        with torch.no_grad():
            data_dict = self.voxelizer(data_dict)

        data_dict = self.voxel_3d_generator(data_dict)

        enc_feats = []
        for i in range(self.num_scales):
            enc_feats.append(self.spv_enc_L[i](data_dict))
        for i in range(self.num_scales):
            self.spv_enc_C[i](data_dict)

        output = torch.cat(enc_feats, dim=1)
        data_dict['logits'] = self.classifier(output)
        data_dict['loss'] = 0.
        return data_dict


# ====================================================================
# CMDFusionUAV — Full model for UAVScenes
# ====================================================================
class CMDFusionUAV(nn.Module):
    """Complete CMDFusion model adapted for UAVScenes.

    Combines:
      1. SPVCNN-G 3D backbone (dual L+C branches)
      2. ResNet-50 2D backbone (pretrained on ImageNet)
      3. UpsampleNet (feature map → target image size)
      4. CMDFuse (bidirectional cross-modal fusion + KD)

    Args:
        num_classes:   Number of semantic classes (default 19 for UAVScenes).
        input_dims:    Per-point input feature dims (default 3 for XYZ).
        hiden_size:    Hidden feature dimension (default 128).
        scale_list:    Multi-scale voxel scales (default [2, 4, 8, 16]).
        min_volume_space: [x_min, y_min, z_min] volume bounds.
        max_volume_space: [x_max, y_max, z_max] volume bounds.
        spatial_shape: Voxel grid resolution [X, Y, Z].
        img_target_size: [H, W] for 2D feature correspondence.
        backbone_2d:   ResNet variant ('resnet50' or 'resnet34').
        pretrained_2d: Use ImageNet pretrained weights.
        lambda_seg2d:  Weight for KD loss (default 4.0).
        lambda_xm:     Weight for cross-modal loss (default 0.05).
        ignore_label:  Label to ignore in loss (default 0 = Background).
        seg_labelweights: Optional per-class weights for CE loss.
    """
    def __init__(
        self,
        num_classes=19,
        input_dims=3,
        hiden_size=128,
        scale_list=None,
        min_volume_space=None,
        max_volume_space=None,
        spatial_shape=None,
        img_target_size=None,
        backbone_2d='resnet50',
        pretrained_2d=True,
        lambda_seg2d=4.0,
        lambda_xm=0.05,
        ignore_label=0,
        seg_labelweights=None,
    ):
        super(CMDFusionUAV, self).__init__()
        self.num_classes = num_classes
        self.hiden_size = hiden_size

        if scale_list is None:
            scale_list = [2, 4, 8, 16]
        if min_volume_space is None:
            min_volume_space = [-100, -100, -50]
        if max_volume_space is None:
            max_volume_space = [100, 100, 150]
        if spatial_shape is None:
            spatial_shape = [400, 400, 40]
        if img_target_size is None:
            img_target_size = [360, 480]

        self.scale_list = scale_list
        self.img_target_size = img_target_size

        coors_range_xyz = [
            [min_volume_space[0], max_volume_space[0]],
            [min_volume_space[1], max_volume_space[1]],
            [min_volume_space[2], max_volume_space[2]],
        ]

        # 3D backbone
        self.model_3d = SPVCNN3D(
            input_dims=input_dims,
            hiden_size=hiden_size,
            num_classes=num_classes,
            scale_list=scale_list,
            coors_range_xyz=coors_range_xyz,
            spatial_shape=spatial_shape,
        )

        # 2D backbone
        self.model_2d = ResNetFCN(
            backbone=backbone_2d,
            pretrained=pretrained_2d,
        )

        # Upsample 2D features to target image size
        self.upsamplenet = UpsampleNetUAV(
            target_size=img_target_size,
            hiden_size=hiden_size,
        )

        # Bidirectional fusion
        self.fusion = CMDFuse(
            hiden_size=hiden_size,
            scale_list=scale_list,
            num_classes=num_classes,
            lambda_xm=lambda_xm,
            lambda_seg2d=lambda_seg2d,
            seg_labelweights=seg_labelweights,
            ignore_label=ignore_label,
        )

    def forward(self, data_dict):
        """Full CMDFusion forward pass.

        Args:
            data_dict: dict from collate_fn_uav() with keys:
              - points:          (total_N, 3) flat point features
              - batch_idx:       (total_N,) batch index
              - labels:          (total_N,) ground truth labels
              - batch_size:      int
              - img:             (B, 3, H, W) images
              - img_indices:     list of (M_i, 2) pixel coords per batch
              - point2img_index: list of (M_i,) visible point indices

        Returns:
            data_dict with added keys:
              - logits:           (total_N, C) 3D-only logits
              - fuse_pts_scale_all: (total_N, C) fused logits (primary)
              - loss:             scalar training loss
        """
        # ============ Freeze 2D backbone during training ============
        self.model_2d.eval()
        for p in self.model_2d.parameters():
            p.detach_()

        # ============ 3D backbone ============
        data_dict = self.model_3d(data_dict)

        # ============ 2D backbone ============
        # Resize image to [256, 512] for ResNet (same as original CMDFusion)
        data_dict['img'] = transforms.Resize([256, 512])(data_dict['img'])
        output_image = self.model_2d(data_dict)

        # ============ Upsample 2D features ============
        data_dict = self.upsamplenet(output_image, data_dict)

        # ============ Point-to-image feature lookup ============
        # Extract 2D features at each point's projected pixel location
        process_keys = [
            k for k in data_dict.keys() if k.find('img_scale') != -1
        ]
        img_indices = data_dict['img_indices']

        temp = {k: [] for k in process_keys}
        for i in range(data_dict['batch_size']):
            for k in process_keys:
                # data_dict[k] is (B, C, H, W) → permute to (B, H, W, C)
                # then index by (row, col) to get (M_i, C) features
                temp[k].append(
                    data_dict[k].permute(0, 2, 3, 1)[i][
                        img_indices[i][:, 0], img_indices[i][:, 1]
                    ]
                )
        for k in process_keys:
            data_dict[k] = torch.cat(temp[k], 0)

        # ============ Bidirectional fusion ============
        data_dict = self.fusion(data_dict)

        return data_dict


# ====================================================================
# Utility: build model from config dict
# ====================================================================
def build_cmdfusion_uav(config):
    """Build CMDFusionUAV model from a config dictionary.

    Expected config keys (all optional with defaults):
        num_classes, input_dims, hiden_size, scale_list,
        min_volume_space, max_volume_space, spatial_shape,
        img_target_size, backbone_2d, pretrained_2d,
        lambda_seg2d, lambda_xm, ignore_label, seg_labelweights
    """
    return CMDFusionUAV(
        num_classes=config.get('num_classes', 19),
        input_dims=config.get('input_dims', 3),
        hiden_size=config.get('hiden_size', 128),
        scale_list=config.get('scale_list', [2, 4, 8, 16]),
        min_volume_space=config.get('min_volume_space', [-100, -100, -50]),
        max_volume_space=config.get('max_volume_space', [100, 100, 150]),
        spatial_shape=config.get('spatial_shape', [400, 400, 40]),
        img_target_size=config.get('img_target_size', [360, 480]),
        backbone_2d=config.get('backbone_2d', 'resnet50'),
        pretrained_2d=config.get('pretrained_2d', True),
        lambda_seg2d=config.get('lambda_seg2d', 4.0),
        lambda_xm=config.get('lambda_xm', 0.05),
        ignore_label=config.get('ignore_label', 0),
        seg_labelweights=config.get('seg_labelweights', None),
    )

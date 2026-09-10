"""
train_cmdfusion_uav.py — Training / evaluation script for CMDFusion on UAVScenes.

Mirrors the PMNet UAVScenes pipeline but drives the CMDFusion model.
Uses standard PyTorch training loop (no Lightning).

Usage:
    python train_cmdfusion_uav.py                    # train with defaults
    python train_cmdfusion_uav.py --epochs 100       # custom epochs
    python train_cmdfusion_uav.py --eval_only --checkpoint best_model.pth
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- Ensure project root is on path ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_utils import (
    get_train_val_test_split,
    compute_class_weights,
    compute_metrics,
    build_frame_list,
    NUM_CLASSES,
    CLASS_NAMES,
    IGNORE_INDEX,
)
from uav_cmdfusion_dataset import UAVScenesCMDFusionDataset, collate_fn_uav
from model_cmdfusion_uav import CMDFusionUAV, build_cmdfusion_uav


# ====================================================================
# Default config
# ====================================================================
DEFAULT_CONFIG = {
    # Model
    'num_classes': NUM_CLASSES,
    'input_dims': 3,
    'hiden_size': 128,
    'scale_list': [2, 4, 8, 16],
    'min_volume_space': [-100, -100, -50],
    'max_volume_space': [100, 100, 150],
    'spatial_shape': [400, 400, 40],
    'img_target_size': [360, 480],
    'backbone_2d': 'resnet50',
    'pretrained_2d': True,
    'lambda_seg2d': 4.0,
    'lambda_xm': 0.05,
    'ignore_label': 0,

    # Dataset
    'img_h': 360,
    'img_w': 480,
    'image_normalizer': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],

    # Training
    'batch_size': 2,
    'num_workers': 4,
    'epochs': 64,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'lr_scheduler': 'cosine',
    'gradient_clip': 1.0,
    'use_amp': True,
    'val_every_n_epoch': 1,
    'save_dir': os.path.join(_THIS_DIR, 'checkpoints'),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='CMDFusion on UAVScenes — Training Script'
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (overrides defaults)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='Run evaluation only (requires --checkpoint)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_verify', action='store_true', default=False,
                        help='Skip sequence label verification')
    return parser.parse_args()


def load_config(args):
    """Build config from defaults + YAML file + CLI overrides."""
    config = DEFAULT_CONFIG.copy()

    # Load YAML config if provided
    if args.config is not None:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        config.update(yaml_config)

    # CLI overrides
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['learning_rate'] = args.lr
    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
    if args.save_dir is not None:
        config['save_dir'] = args.save_dir

    return config


# ====================================================================
# Validation
# ====================================================================
def validate(model, loader, device, num_classes=NUM_CLASSES):
    """Run validation over a DataLoader for CMDFusion.

    Returns:
        avg_loss: average loss over the dataset
        metrics: dict with 'miou', 'accuracy', 'per_class_iou', etc.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            # Move tensors to device
            batch = move_to_device(batch, device)

            with torch.cuda.amp.autocast(enabled=True):
                data_dict = model(batch)

            loss = data_dict['loss']
            n_pts = len(data_dict['labels'])
            total_loss += loss.item() * n_pts
            total_samples += n_pts

            # Use fused 3D logits (primary output)
            if 'fuse_pts_scale_all' in data_dict:
                preds = data_dict['fuse_pts_scale_all'].argmax(1)
            else:
                preds = data_dict['logits'].argmax(1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(data_dict['labels'].cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metrics = compute_metrics(all_preds, all_targets, num_classes)
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, metrics


def move_to_device(batch, device):
    """Move batch tensors to device, handling mixed types."""
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            batch[k] = [t.to(device, non_blocking=True) for t in v]
    return batch


# ====================================================================
# Training
# ====================================================================
def train(config, args):
    """Main training loop."""
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    # ============ Data splits ============
    verify = not args.skip_verify
    train_seqs, val_seqs, test_seqs = get_train_val_test_split(verify=verify)
    print(f"Train: {len(train_seqs)} sequences")
    print(f"Val:   {len(val_seqs)} sequences")
    print(f"Test:  {len(test_seqs)} sequences")

    # ============ Datasets ============
    train_dataset = UAVScenesCMDFusionDataset(
        sequences=train_seqs,
        img_h=config['img_h'],
        img_w=config['img_w'],
        augment=True,
        min_volume_space=config['min_volume_space'],
        max_volume_space=config['max_volume_space'],
        image_normalizer=config.get('image_normalizer'),
    )
    val_dataset = UAVScenesCMDFusionDataset(
        sequences=val_seqs,
        img_h=config['img_h'],
        img_w=config['img_w'],
        augment=False,
        min_volume_space=config['min_volume_space'],
        max_volume_space=config['max_volume_space'],
        image_normalizer=config.get('image_normalizer'),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn_uav,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn_uav,
        pin_memory=True,
    )

    # ============ Class weights ============
    print("Computing class weights...")
    frame_list = build_frame_list(train_seqs)
    class_weights_np = compute_class_weights(frame_list, num_classes=NUM_CLASSES)
    config['seg_labelweights'] = class_weights_np.tolist()
    print(f"Class weights: {class_weights_np}")

    # ============ Model ============
    model = build_cmdfusion_uav(config)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    # ============ Optimizer & Scheduler ============
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    if config['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'] - 4,
            eta_min=1e-6,
        )
    elif config['lr_scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.5
        )
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])

    # ============ Resume from checkpoint ============
    start_epoch = 0
    best_miou = 0.0
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        if 'optimizer_state_dict' in ckpt and not args.eval_only:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        if 'best_miou' in ckpt:
            best_miou = ckpt['best_miou']
        print(f"Resumed from epoch {start_epoch}, best_miou={best_miou:.4f}")

    # ============ Eval only ============
    if args.eval_only:
        print("\n" + "=" * 60)
        print("EVALUATION ONLY")
        print("=" * 60)
        val_loss, metrics = validate(model, val_loader, device)
        print(f"\nVal Loss: {val_loss:.4f}")
        print(f"Val mIoU: {metrics['miou'] * 100:.2f}%")
        print(f"Val Acc:  {metrics['accuracy'] * 100:.2f}%")
        print("\nPer-class IoU:")
        for i, (name, iou) in enumerate(
            zip(CLASS_NAMES, metrics['per_class_iou'])
        ):
            print(f"  {i:2d} {name:18s}: {iou * 100:6.2f}%")
        return

    # ============ Save directory ============
    os.makedirs(config['save_dir'], exist_ok=True)
    print(f"Checkpoints will be saved to: {config['save_dir']}")

    # ============ Training loop ============
    print("\n" + "=" * 60)
    print(f"TRAINING: {config['epochs']} epochs, "
          f"batch_size={config['batch_size']}, "
          f"lr={config['learning_rate']}")
    print("=" * 60)

    for epoch in range(start_epoch, config['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_pts = 0
        t_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = move_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=config['use_amp']):
                data_dict = model(batch)
                loss = data_dict['loss']

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARNING] NaN/Inf loss at batch {batch_idx}, skipping")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            # Gradient clipping
            if config['gradient_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config['gradient_clip']
                )

            scaler.step(optimizer)
            scaler.update()

            n_pts = len(data_dict['labels'])
            epoch_loss += loss.item() * n_pts
            epoch_pts += n_pts

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                lr_now = optimizer.param_groups[0]['lr']
                print(
                    f"  Epoch {epoch+1}/{config['epochs']} "
                    f"[{batch_idx+1}/{len(train_loader)}] "
                    f"loss={loss.item():.4f}  lr={lr_now:.6f}"
                )

        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / max(epoch_pts, 1)
        elapsed = time.time() - t_start
        print(
            f"\nEpoch {epoch+1}/{config['epochs']} — "
            f"train_loss={avg_loss:.4f}  "
            f"time={elapsed:.1f}s"
        )

        # ============ Validation ============
        if (epoch + 1) % config['val_every_n_epoch'] == 0:
            print("Running validation...")
            val_loss, metrics = validate(model, val_loader, device)
            miou = metrics['miou']
            acc = metrics['accuracy']
            print(
                f"  val_loss={val_loss:.4f}  "
                f"val_mIoU={miou * 100:.2f}%  "
                f"val_acc={acc * 100:.2f}%"
            )

            # Per-class IoU summary
            for i, (name, iou_val) in enumerate(
                zip(CLASS_NAMES, metrics['per_class_iou'])
            ):
                print(f"    {i:2d} {name:18s}: {iou_val * 100:6.2f}%")

            # Save best
            if miou > best_miou:
                best_miou = miou
                save_path = os.path.join(
                    config['save_dir'], 'best_model.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_miou': best_miou,
                    'config': config,
                }, save_path)
                print(f"  ★ New best mIoU: {best_miou * 100:.2f}% "
                      f"→ saved {save_path}")

        # Save latest
        save_path_latest = os.path.join(
            config['save_dir'], 'latest_model.pth'
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_miou': best_miou,
            'config': config,
        }, save_path_latest)

    print("\n" + "=" * 60)
    print(f"Training complete! Best mIoU: {best_miou * 100:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    train(config, args)

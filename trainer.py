import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

from config.config import PTGConfig
from models.losses.ptg_loss import PTGLoss
from training.metrics import compute_segmentation_metrics
from training.early_stopping import EarlyStopping
from training.postprocess import apply_postprocess, normalize_postprocess_modes


class Trainer:

    def __init__(self, model, config: PTGConfig, run_name=None):
        self.config = config
        self.device = torch.device(config.DEVICE)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

        if run_name is None:
            self.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.run_name = run_name

        self.model = model.to(self.device)
        self.criterion = PTGLoss(config).to(self.device)
        self.use_amp = bool(getattr(config, 'USE_AMP', False)) and self.device.type == 'cuda'
        self.scaler = self._make_grad_scaler()
        self.pass_seismic_to_physics = bool(getattr(config, 'PASS_SEISMIC_TO_PHYSICS', True))
        self.checkpoint_save_interval = int(getattr(config, 'CHECKPOINT_SAVE_INTERVAL', 10))
        self.use_ema = bool(getattr(config, 'USE_EMA', False))
        self.ema_decay = float(getattr(config, 'EMA_DECAY', 0.995))
        self.ema_state = None
        self.eval_postprocess_modes = normalize_postprocess_modes(
            getattr(config, 'EVAL_POSTPROCESS_MODES', ['none'])
        )

        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        self.scheduler = None
        if config.USE_SCHEDULER:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=config.SCHEDULER_FACTOR,
                patience=config.SCHEDULER_PATIENCE,
                verbose=True
            )

        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
            mode='max'
        )

        self.checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, self.run_name)
        self.log_dir = os.path.join(config.LOG_DIR, self.run_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'best_threshold': [],
            'best_postprocess': [],
            'learning_rates': []
        }

        self.best_val_f1 = 0.0
        self.best_threshold = 0.5
        self.best_postprocess = 'none'
        self.current_epoch = 0
        if self.use_ema:
            self.ema_state = self._copy_model_state()

    def _copy_model_state(self):
        return {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }

    def _update_ema(self):
        if self.ema_state is None:
            return
        with torch.no_grad():
            model_state = self.model.state_dict()
            for name, value in model_state.items():
                if name not in self.ema_state:
                    self.ema_state[name] = value.detach().clone()
                    continue
                if torch.is_floating_point(value):
                    self.ema_state[name].mul_(self.ema_decay).add_(value.detach(), alpha=1.0 - self.ema_decay)
                else:
                    self.ema_state[name].copy_(value)

    def _apply_ema_state(self):
        if self.ema_state is None:
            return None
        current_state = self._copy_model_state()
        ema_state = self._copy_model_state()
        ema_state.update(self.ema_state)
        self.model.load_state_dict(ema_state, strict=False)
        return current_state

    def _restore_model_state(self, state):
        if state is not None:
            self.model.load_state_dict(state, strict=True)

    def _make_grad_scaler(self):
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            try:
                return torch.amp.GradScaler('cuda', enabled=self.use_amp)
            except TypeError:
                return torch.amp.GradScaler(enabled=self.use_amp)
        return torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _autocast(self):
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            try:
                return torch.amp.autocast('cuda', enabled=self.use_amp)
            except TypeError:
                return torch.amp.autocast(enabled=self.use_amp)
        return torch.cuda.amp.autocast(enabled=self.use_amp)

    def _device_transfer_kwargs(self):
        return {'non_blocking': self.device.type == 'cuda'}

    def _compute_loss(self, pred, labels, seismic):
        if self.pass_seismic_to_physics:
            return self.criterion(pred, labels, seismic=seismic)
        return self.criterion(pred, labels)

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch} [Train]')
        for batch_idx, (seismic, labels) in enumerate(pbar):
            transfer_kwargs = self._device_transfer_kwargs()
            seismic = seismic.to(self.device, **transfer_kwargs)
            labels = labels.to(self.device, **transfer_kwargs)

            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast():
                pred = self.model(seismic)
                loss, loss_dict = self._compute_loss(pred, labels, seismic)

            if self.config.GRADIENT_CLIP > 0:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.GRADIENT_CLIP
                )
            else:
                self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._update_ema()
            total_loss += loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.6f}'
            })

        return total_loss / num_batches

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)

        thresholds = getattr(self.config, 'EVAL_THRESHOLDS', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        metric_counts = {
            (thresh, mode): {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
            for thresh in thresholds
            for mode in self.eval_postprocess_modes
        }

        backup_state = self._apply_ema_state() if self.use_ema else None
        try:
            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch} [Val]')
                for seismic, labels in pbar:
                    transfer_kwargs = self._device_transfer_kwargs()
                    seismic = seismic.to(self.device, **transfer_kwargs)
                    labels = labels.to(self.device, **transfer_kwargs)

                    with self._autocast():
                        pred = self.model(seismic)
                        loss, loss_dict = self._compute_loss(pred, labels, seismic)
                    total_loss += loss.item()

                    labels_bool = labels > 0.5
                    labels_neg = ~labels_bool
                    for thresh in thresholds:
                        raw_pred_bool = pred > thresh
                        for mode in self.eval_postprocess_modes:
                            pred_bool = apply_postprocess(raw_pred_bool, mode)
                            pred_neg = ~pred_bool
                            counts = metric_counts[(thresh, mode)]
                            counts['tp'] += torch.logical_and(pred_bool, labels_bool).sum().item()
                            counts['tn'] += torch.logical_and(pred_neg, labels_neg).sum().item()
                            counts['fp'] += torch.logical_and(pred_bool, labels_neg).sum().item()
                            counts['fn'] += torch.logical_and(pred_neg, labels_bool).sum().item()
        finally:
            self._restore_model_state(backup_state)

        best_f1 = -1.0
        best_metrics = None
        best_threshold = 0.5
        best_postprocess = 'none'

        for thresh in thresholds:
            for mode in self.eval_postprocess_modes:
                counts = metric_counts[(thresh, mode)]
                tp = counts['tp']
                tn = counts['tn']
                fp = counts['fp']
                fn = counts['fn']
                total = tp + tn + fp + fn
                accuracy = (tp + tn) / total if total > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_metrics = metrics
                    best_threshold = thresh
                    best_postprocess = mode

        return total_loss / num_batches, best_metrics, best_threshold, best_postprocess

    def fit(self, train_loader, val_loader):
        print("=" * 60)
        print(f"Starting training run: {self.run_name}")
        print(f"Device: {self.device}")
        print(f"AMP enabled: {self.use_amp}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"PTG Loss weights: lambda_data=1.0, lambda_phys={self.config.PHYSICS_LOSS_WEIGHT}, "
              f"lambda_topo={self.config.TOPOLOGY_LOSS_WEIGHT}, lambda_geo={self.config.GEOLOGY_LOSS_WEIGHT}")
        print(f"Balanced BCE: {getattr(self.config, 'USE_BALANCED_BCE', True)}")
        print(f"Pass seismic to physics loss: {self.pass_seismic_to_physics}")
        print(f"EMA enabled: {self.use_ema} (decay={self.ema_decay})")
        print(f"Eval postprocess modes: {self.eval_postprocess_modes}")
        print("=" * 60)

        start_epoch = self.current_epoch + 1
        for epoch in range(start_epoch, self.config.NUM_EPOCHS + 1):
            self.current_epoch = epoch

            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_metrics, best_threshold, best_postprocess = self.validate(val_loader)

            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['best_threshold'].append(best_threshold)
            self.history['best_postprocess'].append(best_postprocess)
            self.history['learning_rates'].append(current_lr)

            print(f"\nEpoch {epoch}/{self.config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Best threshold for this epoch: {best_threshold:.3f}")
            print(f"Best postprocess for this epoch: {best_postprocess}")
            print(f"Val Metrics: Acc={val_metrics['accuracy']:.4f}, "
                  f"Prec={val_metrics['precision']:.4f}, "
                  f"Rec={val_metrics['recall']:.4f}, "
                  f"F1={val_metrics['f1']:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")

            if hasattr(self.criterion, 'get_loss_components'):
                loss_components = self.criterion.get_loss_components()
                print(f"Loss components: data={loss_components.get('data', 0):.4f}, "
                      f"physics={loss_components.get('physics', 0):.4f}, "
                      f"topology={loss_components.get('topology', 0):.4f}, "
                      f"geology={loss_components.get('geology', 0):.4f}")

            if self.scheduler:
                self.scheduler.step(val_metrics['f1'])

            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_threshold = best_threshold
                self.best_postprocess = best_postprocess
                self._save_checkpoint(epoch, val_metrics, best_threshold, best_postprocess, is_best=True)
                print(f"New best model saved with F1={val_metrics['f1']:.4f} "
                      f"(threshold={best_threshold:.3f}, postprocess={best_postprocess})")

            self._save_checkpoint(epoch, val_metrics, best_threshold, best_postprocess, filename='last_model.pth')

            if self.early_stopping(val_metrics['f1']):
                print(f"Early stopping triggered at epoch {epoch}")
                break

            if self.checkpoint_save_interval > 0 and epoch % self.checkpoint_save_interval == 0:
                self._save_checkpoint(epoch, val_metrics, best_threshold, best_postprocess)
                self._save_history()

        self._save_history()
        print("=" * 60)
        print("Training completed!")
        print(f"Best validation F1: {self.best_val_f1:.4f} "
              f"(threshold={self.best_threshold:.3f}, postprocess={self.best_postprocess})")
        print("=" * 60)

        return self.history

    def _config_dict(self):
        config_dict = {}
        for name in dir(self.config):
            if not name.isupper():
                continue
            value = getattr(self.config, name)
            if isinstance(value, tuple):
                value = list(value)
            if isinstance(value, (str, int, float, bool, list)) or value is None:
                config_dict[name] = value
        return config_dict

    def _save_checkpoint(self, epoch, metrics, best_threshold, best_postprocess='none', is_best=False, filename=None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'best_val_f1': self.best_val_f1,
            'best_threshold': best_threshold,
            'best_postprocess': best_postprocess,
            'config': self._config_dict(),
            'run_name': self.run_name,
            'history': self.history
        }

        if filename is not None:
            filename = os.path.join(self.checkpoint_dir, filename)
        elif is_best:
            filename = os.path.join(self.checkpoint_dir, 'best_model.pth')
        else:
            filename = os.path.join(self.checkpoint_dir, f'checkpoint_epoch{epoch}.pth')

        torch.save(checkpoint, filename)

    def _save_history(self):
        history_file = os.path.join(self.log_dir, 'history.json')
        with open(history_file, 'w') as f:
            json_history = {}
            for key, values in self.history.items():
                json_history[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v
                                     for v in values]
            json.dump(json_history, f, indent=2)

    def load_checkpoint(self, checkpoint_path, model_only=False, strict=True):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        if model_only:
            if self.use_ema:
                self.ema_state = self._copy_model_state()
            print(f"Loaded model weights only from checkpoint: {checkpoint_path}")
            return
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint.get('scaler_state_dict') is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if self.use_ema:
            if checkpoint.get('ema_state_dict') is not None:
                self.ema_state = checkpoint['ema_state_dict']
            else:
                self.ema_state = self._copy_model_state()
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        self.best_threshold = checkpoint.get('best_threshold', 0.5)
        self.best_postprocess = checkpoint.get('best_postprocess', 'none')
        self.history = checkpoint.get('history', self.history)
        self.current_epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation F1: {self.best_val_f1:.4f} "
              f"(threshold={self.best_threshold:.3f}, postprocess={self.best_postprocess})")

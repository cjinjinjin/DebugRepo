"""
Fix for PyTorch 2.8+ / DeepSpeed LR scheduler param_group mismatch.

ms-swift bug #8299: scheduler is created before accelerator.prepare(),
but DeepSpeed reorganizes optimizer param_groups during prepare, causing
len(scheduler.base_lrs) != len(optimizer.param_groups).
PyTorch 2.8+ strict zip in _update_lr raises ValueError.

This callback truncates scheduler attributes to match optimizer after
preparation is complete.

Usage: --external_plugins ./fix_lr_scheduler.py --callbacks fix_lr
"""

from swift.callbacks import callbacks_map
from swift.callbacks.base import TrainerCallback


class FixLRSchedulerCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self.trainer
        scheduler = trainer.lr_scheduler
        optimizer = trainer.optimizer
        if scheduler is None or optimizer is None:
            return

        n_groups = len(optimizer.param_groups)
        n_lrs = len(getattr(scheduler, "base_lrs", []))
        if n_lrs == n_groups:
            return

        print(f"[FixLRScheduler] Truncating scheduler attrs: "
              f"{n_lrs} base_lrs -> {n_groups} param_groups")

        scheduler.base_lrs = scheduler.base_lrs[:n_groups]
        if hasattr(scheduler, "lr_lambdas"):
            scheduler.lr_lambdas = scheduler.lr_lambdas[:n_groups]
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = scheduler._last_lr[:n_groups]


callbacks_map["fix_lr"] = FixLRSchedulerCallback

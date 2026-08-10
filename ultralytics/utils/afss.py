# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
AFSS: Anti-Forgetting Sampling Strategy for efficient YOLO training.

AFSS tracks per-image learning sufficiency S_i = min(precision_i, recall_i),
classifies training images into Easy/Moderate/Hard groups, and samples each
group at different rates after an initial full-coverage warmup.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

from ultralytics.utils import LOGGER, plt_settings


class AFSSManager:
    """Manage per-image AFSS states and sample active training images for each epoch."""

    def __init__(
        self,
        num_images: int,
        easy_thresh: float = 0.8,
        hard_thresh: float = 0.3,
        easy_ratio: float = 0.02,
        moderate_ratio: float = 0.4,
        update_interval: int = 5,
        easy_review_interval: int = 10,
        moderate_review_interval: int = 3,
        warmup_epochs: int = 10,
        num_classes: int = 0,
        class_aware: bool = False,
    ) -> None:
        """Initialize the AFSS manager."""
        if num_images <= 0:
            raise ValueError("AFSS requires num_images > 0")
        if not 0 <= hard_thresh < easy_thresh <= 1:
            raise ValueError("AFSS thresholds must satisfy 0 <= hard_thresh < easy_thresh <= 1")

        self.num_images = int(num_images)
        self.easy_thresh = float(easy_thresh)
        self.hard_thresh = float(hard_thresh)
        self.easy_ratio = float(easy_ratio)
        self.moderate_ratio = float(moderate_ratio)
        self.update_interval = max(1, int(update_interval))
        self.easy_review_interval = max(1, int(easy_review_interval))
        self.moderate_review_interval = max(1, int(moderate_review_interval))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.num_classes = max(0, int(num_classes))
        self.class_aware = bool(class_aware and self.num_classes > 0)

        self.precision = np.zeros(self.num_images, dtype=np.float32)
        self.recall = np.zeros(self.num_images, dtype=np.float32)
        self.sufficiency = np.zeros(self.num_images, dtype=np.float32)
        self.last_used_epoch = np.full(self.num_images, -1, dtype=np.int32)
        if self.class_aware:
            self.class_precision = np.ones((self.num_images, self.num_classes), dtype=np.float32)
            self.class_recall = np.ones((self.num_images, self.num_classes), dtype=np.float32)
            self.class_sufficiency = np.ones((self.num_images, self.num_classes), dtype=np.float32)
            self.class_present = np.zeros((self.num_images, self.num_classes), dtype=bool)
            self.class_aware_sufficiency = np.zeros(self.num_images, dtype=np.float32)
        else:
            self.class_precision = None
            self.class_recall = None
            self.class_sufficiency = None
            self.class_present = None
            self.class_aware_sufficiency = self.sufficiency

        self.current_epoch = 0
        self._classify_dirty = True
        self._easy_indices: List[int] = []
        self._moderate_indices: List[int] = []
        self._hard_indices: List[int] = []
        self._adapted = False
        self.history: List[dict] = []

    def update_metrics(
        self,
        image_indices: np.ndarray,
        precisions: np.ndarray,
        recalls: np.ndarray,
        current_epoch: int,
        class_precisions: np.ndarray | None = None,
        class_recalls: np.ndarray | None = None,
        class_present: np.ndarray | None = None,
    ) -> None:
        """Update precision/recall values and recompute image sufficiency."""
        image_indices = np.asarray(image_indices, dtype=np.int64)
        precisions = np.asarray(precisions, dtype=np.float32)
        recalls = np.asarray(recalls, dtype=np.float32)

        valid = (image_indices >= 0) & (image_indices < self.num_images)
        image_indices = image_indices[valid]
        precisions = precisions[valid]
        recalls = recalls[valid]
        if not len(image_indices):
            LOGGER.warning("AFSS: no valid image metrics were provided at epoch %s.", current_epoch)
            return

        self.precision[image_indices] = np.clip(precisions, 0.0, 1.0)
        self.recall[image_indices] = np.clip(recalls, 0.0, 1.0)
        self.sufficiency[image_indices] = np.minimum(self.precision[image_indices], self.recall[image_indices])
        if self.class_aware and class_precisions is not None and class_recalls is not None and class_present is not None:
            class_precisions = np.asarray(class_precisions, dtype=np.float32)[valid]
            class_recalls = np.asarray(class_recalls, dtype=np.float32)[valid]
            class_present = np.asarray(class_present, dtype=bool)[valid]
            if class_precisions.shape == (len(image_indices), self.num_classes):
                self.class_precision[image_indices] = np.clip(class_precisions, 0.0, 1.0)
                self.class_recall[image_indices] = np.clip(class_recalls, 0.0, 1.0)
                self.class_sufficiency[image_indices] = np.minimum(
                    self.class_precision[image_indices], self.class_recall[image_indices]
                )
                self.class_present[image_indices] = class_present
                self.class_aware_sufficiency[image_indices] = self.sufficiency[image_indices]
                rows_with_classes = class_present.any(axis=1)
                if rows_with_classes.any():
                    present_scores = np.where(
                        class_present[rows_with_classes],
                        self.class_sufficiency[image_indices[rows_with_classes]],
                        1.0,
                    )
                    self.class_aware_sufficiency[image_indices[rows_with_classes]] = np.minimum(
                        self.sufficiency[image_indices[rows_with_classes]], present_scores.min(axis=1)
                    )
            else:
                LOGGER.warning(
                    "AFSS: class-aware metric shape mismatch, expected (%d, %d), got %s.",
                    len(image_indices),
                    self.num_classes,
                    class_precisions.shape,
                )
        self._classify_dirty = True

    def classify_images(self) -> tuple[List[int], List[int], List[int]]:
        """Return Easy, Moderate, and Hard image indices."""
        if not self._classify_dirty:
            return self._easy_indices, self._moderate_indices, self._hard_indices

        easy: List[int] = []
        moderate: List[int] = []
        hard: List[int] = []
        scores = self.class_aware_sufficiency if self.class_aware else self.sufficiency
        for i, sufficiency in enumerate(scores):
            if sufficiency >= self.easy_thresh:
                easy.append(i)
            elif sufficiency < self.hard_thresh:
                hard.append(i)
            else:
                moderate.append(i)

        self._easy_indices = easy
        self._moderate_indices = moderate
        self._hard_indices = hard
        self._classify_dirty = False
        return easy, moderate, hard

    def sample_indices(self, current_epoch: int) -> List[int]:
        """Sample active image indices for a training epoch."""
        self.current_epoch = int(current_epoch)

        if current_epoch < self.warmup_epochs:
            all_indices = list(range(self.num_images))
            self.last_used_epoch[:] = current_epoch
            return all_indices

        easy, moderate, hard = self.classify_images()
        selected: List[int] = []

        selected.extend(hard)
        for idx in hard:
            self.last_used_epoch[idx] = current_epoch

        if moderate:
            n_target = max(1, int(len(moderate) * self.moderate_ratio))
            forced = [
                idx
                for idx in moderate
                if (current_epoch - self.last_used_epoch[idx]) >= self.moderate_review_interval
            ]
            if len(forced) > n_target:
                forced = random.sample(forced, n_target)
            selected.extend(forced)
            for idx in forced:
                self.last_used_epoch[idx] = current_epoch

            forced_set = set(forced)
            remaining = [idx for idx in moderate if idx not in forced_set]
            n_random = min(max(0, n_target - len(forced)), len(remaining))
            if n_random:
                random_fill = random.sample(remaining, n_random)
                selected.extend(random_fill)
                for idx in random_fill:
                    self.last_used_epoch[idx] = current_epoch

        if easy:
            n_target = max(1, int(len(easy) * self.easy_ratio))
            n_forced = max(1, n_target // 2)
            forced = [
                idx for idx in easy if (current_epoch - self.last_used_epoch[idx]) >= self.easy_review_interval
            ]
            if len(forced) > n_forced:
                forced = random.sample(forced, n_forced)
            selected.extend(forced)
            for idx in forced:
                self.last_used_epoch[idx] = current_epoch

            forced_set = set(forced)
            remaining = [idx for idx in easy if idx not in forced_set]
            n_random = min(max(0, n_target - len(forced)), len(remaining))
            if n_random:
                random_fill = random.sample(remaining, n_random)
                selected.extend(random_fill)
                for idx in random_fill:
                    self.last_used_epoch[idx] = current_epoch

        return selected or list(range(self.num_images))

    def should_update(self, current_epoch: int) -> bool:
        """Return True when per-image metrics should be recomputed."""
        current_epoch = int(current_epoch)
        if current_epoch < self.warmup_epochs:
            return False
        return (current_epoch - self.warmup_epochs) % self.update_interval == 0

    def get_stats(self) -> dict:
        """Return difficulty distribution statistics."""
        easy, moderate, hard = self.classify_images()
        total = self.num_images
        active = len(hard) + int(len(moderate) * self.moderate_ratio) + int(len(easy) * self.easy_ratio)
        return {
            "easy": len(easy),
            "moderate": len(moderate),
            "hard": len(hard),
            "easy_pct": 100.0 * len(easy) / total,
            "moderate_pct": 100.0 * len(moderate) / total,
            "hard_pct": 100.0 * len(hard) / total,
            "mean_sufficiency": float((self.class_aware_sufficiency if self.class_aware else self.sufficiency).mean()),
            "active": active,
            "active_pct": 100.0 * active / total,
            "class_aware": self.class_aware,
        }

    def get_class_stats(self, names: dict | list | None = None) -> list[dict]:
        """Return class-aware difficulty distribution for images where each class appears."""
        if not self.class_aware:
            return []
        rows = []
        for cls_id in range(self.num_classes):
            present = self.class_present[:, cls_id]
            total = int(present.sum())
            if total == 0:
                continue
            scores = self.class_sufficiency[present, cls_id]
            easy = int((scores >= self.easy_thresh).sum())
            hard = int((scores < self.hard_thresh).sum())
            moderate = total - easy - hard
            if isinstance(names, dict):
                name = names.get(cls_id, str(cls_id))
            elif isinstance(names, list) and cls_id < len(names):
                name = names[cls_id]
            else:
                name = str(cls_id)
            rows.append(
                {
                    "class_id": cls_id,
                    "name": name,
                    "total": total,
                    "easy": easy,
                    "moderate": moderate,
                    "hard": hard,
                    "mean_sufficiency": float(scores.mean()) if total else 0.0,
                }
            )
        return rows

    def record_history(self, epoch: int, stats: dict | None = None) -> dict:
        """Append one AFSS state snapshot to history."""
        stats = stats or self.get_stats()
        row = {
            "epoch": int(epoch),
            "easy": int(stats["easy"]),
            "moderate": int(stats["moderate"]),
            "hard": int(stats["hard"]),
            "easy_pct": float(stats["easy_pct"]),
            "moderate_pct": float(stats["moderate_pct"]),
            "hard_pct": float(stats["hard_pct"]),
            "mean_sufficiency": float(stats["mean_sufficiency"]),
            "active": int(stats["active"]),
            "active_pct": float(stats["active_pct"]),
            "easy_ratio": float(self.easy_ratio),
            "moderate_ratio": float(self.moderate_ratio),
            "class_aware": int(self.class_aware),
        }
        self.history.append(row)
        return row

    def save_history_csv(self, path: str | Path) -> None:
        """Save AFSS history to a CSV file."""
        if not self.history:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.history[0].keys())
        lines = [",".join(keys)]
        lines.extend(",".join(str(row[k]) for k in keys) for row in self.history)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @plt_settings()
    def plot_history(self, path: str | Path) -> None:
        """Plot Easy/Moderate/Hard image counts over AFSS state updates."""
        if not self.history:
            return
        import matplotlib.pyplot as plt

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        epochs = [row["epoch"] for row in self.history]
        easy = np.array([row["easy"] for row in self.history], dtype=np.float32)
        moderate = np.array([row["moderate"] for row in self.history], dtype=np.float32)
        hard = np.array([row["hard"] for row in self.history], dtype=np.float32)

        scale = 10000.0 if max(easy.max(), moderate.max(), hard.max()) >= 10000 else 1.0
        ylabel = "Number of samples (x10^4)" if scale > 1 else "Number of samples"

        fig, ax = plt.subplots(figsize=(8, 4.8), tight_layout=True)
        ax.plot(epochs, easy / scale, color="green", marker="o", linewidth=2, markersize=4, label="Easy")
        ax.plot(epochs, moderate / scale, color="blue", marker="s", linewidth=2, markersize=4, label="Moderate")
        ax.plot(epochs, hard / scale, color="red", marker="^", linewidth=2, markersize=4, label="Hard")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title("AFSS Difficulty Distribution")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", frameon=True)
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def adapt_ratios(self, batch_size: int = 16) -> dict | None:
        """Adjust sampling ratios once based on the first observed difficulty distribution."""
        if self._adapted:
            return None

        easy, moderate, hard = self.classify_images()
        n_easy, n_moderate, n_hard = len(easy), len(moderate), len(hard)
        n_total = n_easy + n_moderate + n_hard
        if n_total == 0:
            return None

        old_easy_ratio = self.easy_ratio
        old_moderate_ratio = self.moderate_ratio
        active = n_hard + int(n_moderate * self.moderate_ratio) + int(n_easy * self.easy_ratio)
        active_ratio = active / n_total

        min_active_ratio = 0.25
        min_active_count = max(int(batch_size) * 16, int(batch_size) * 5)
        for _ in range(50):
            if active_ratio >= min_active_ratio and active >= min_active_count:
                break
            bumped = False
            if self.easy_ratio < 0.80:
                self.easy_ratio = min(0.80, self.easy_ratio * 1.15)
                bumped = True
            if self.moderate_ratio < 0.95:
                self.moderate_ratio = min(0.95, self.moderate_ratio * 1.10)
                bumped = True
            active = n_hard + int(n_moderate * self.moderate_ratio) + int(n_easy * self.easy_ratio)
            active_ratio = active / n_total
            if not bumped:
                break

        self._adapted = True
        if abs(self.easy_ratio - old_easy_ratio) <= 0.001 and abs(self.moderate_ratio - old_moderate_ratio) <= 0.001:
            return None

        return {
            "old_easy_ratio": old_easy_ratio,
            "old_moderate_ratio": old_moderate_ratio,
            "new_easy_ratio": self.easy_ratio,
            "new_moderate_ratio": self.moderate_ratio,
            "n_easy": n_easy,
            "n_moderate": n_moderate,
            "n_hard": n_hard,
            "active": active,
            "active_ratio": active_ratio,
        }


class AFSSBatchSampler:
    """Batch sampler that switches its active indices when AFSS epoch changes."""

    def __init__(self, afss_manager: AFSSManager, batch_size: int, drop_last: bool = False) -> None:
        """Initialize the AFSS batch sampler."""
        self.afss = afss_manager
        self.batch_size = max(1, int(batch_size))
        self.drop_last = bool(drop_last)
        self._last_epoch = -1
        self._last_active_indices: List[int] = []

    def _compute_active(self) -> List[int]:
        epoch = self.afss.current_epoch
        if epoch != self._last_epoch:
            self._last_epoch = epoch
            self._last_active_indices = self.afss.sample_indices(epoch)
        return self._last_active_indices

    def __iter__(self):
        """Yield batches indefinitely for compatibility with InfiniteDataLoader."""
        while True:
            active_indices = list(self._compute_active())
            random.shuffle(active_indices)
            if self.drop_last:
                n_batches = len(active_indices) // self.batch_size
                for i in range(n_batches):
                    start = i * self.batch_size
                    yield active_indices[start : start + self.batch_size]
            else:
                for i in range(0, len(active_indices), self.batch_size):
                    yield active_indices[i : i + self.batch_size]

    def __len__(self) -> int:
        active = self._compute_active()
        if self.drop_last:
            return len(active) // self.batch_size
        return math.ceil(len(active) / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch."""
        self.afss.current_epoch = int(epoch)


def suggest_afss_params(
    num_images: int,
    batch_size: int = 16,
    epochs: int = 300,
    num_classes: int = 10,
    easy_thresh: float = 0.8,
    hard_thresh: float = 0.3,
) -> Dict:
    """Suggest conservative AFSS hyperparameters from dataset scale."""
    if num_images < 500:
        LOGGER.warning(
            "AFSS auto-tune: dataset too small (%d images). AFSS is not recommended for this run.",
            num_images,
        )
        return {"afss": False, "reason": "dataset_too_small"}

    warmup_epochs = max(10, min(50, round(num_images / 40)))
    scale = min(math.sqrt(118000 / max(num_images, 1)), 8.0)
    easy_ratio = min(0.50, 0.02 * scale)
    moderate_ratio = min(0.90, 0.40 * math.sqrt(scale))

    n_easy = int(num_images * 0.70)
    n_moderate = int(num_images * 0.20)
    n_hard = int(num_images * 0.10)
    active = n_hard + int(n_moderate * moderate_ratio) + int(n_easy * easy_ratio)
    active_ratio = active / num_images if num_images else 0.0

    min_active_ratio = 0.25
    min_active_batches = max(5, int(batch_size))
    while active_ratio < min_active_ratio or active < int(batch_size) * min_active_batches:
        easy_ratio = min(0.80, easy_ratio * 1.15)
        moderate_ratio = min(0.95, moderate_ratio * 1.10)
        active = n_hard + int(n_moderate * moderate_ratio) + int(n_easy * easy_ratio)
        active_ratio = active / num_images if num_images else 0.0
        if easy_ratio >= 0.80 and moderate_ratio >= 0.95:
            break

    update_interval = max(3, min(20, round(num_images / 200)))
    easy_ratio = round(easy_ratio, 3)
    moderate_ratio = round(moderate_ratio, 3)
    easy_thresh = round(float(easy_thresh), 3)
    hard_thresh = round(float(hard_thresh), 3)
    speedup = 1.0 / active_ratio if active_ratio > 0 else 1.0
    n_updates = max(1, (int(epochs) - warmup_epochs) // update_interval) if int(epochs) > warmup_epochs else 0

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("  AFSS Auto-Tune Recommendations")
    LOGGER.info("=" * 60)
    LOGGER.info(f"  Dataset        : {num_images:,} images, {num_classes} classes")
    LOGGER.info(f"  Batch size     : {batch_size}")
    LOGGER.info(f"  Epochs         : {epochs}")
    LOGGER.info("-" * 60)
    LOGGER.info(f"  easy_thresh    : {easy_thresh}")
    LOGGER.info(f"  hard_thresh    : {hard_thresh}")
    LOGGER.info(f"  easy_ratio     : {easy_ratio:.1%}")
    LOGGER.info(f"  moderate_ratio : {moderate_ratio:.1%}")
    LOGGER.info(f"  update_interval: {update_interval} epochs")
    LOGGER.info(f"  warmup_epochs  : {warmup_epochs} epochs")
    LOGGER.info("-" * 60)
    LOGGER.info(f"  Est. active set      : ~{active:,} / {num_images:,} ({active_ratio:.0%})")
    LOGGER.info(f"  Est. training speedup: ~{speedup:.1f}x per epoch")
    LOGGER.info(f"  Total AFSS updates   : ~{n_updates} times during training")
    LOGGER.info("=" * 60 + "\n")

    return {
        "afss": True,
        "afss_easy_thresh": easy_thresh,
        "afss_hard_thresh": hard_thresh,
        "afss_easy_ratio": easy_ratio,
        "afss_moderate_ratio": moderate_ratio,
        "afss_update_interval": update_interval,
        "afss_warmup_epochs": warmup_epochs,
        "_active_ratio": round(active_ratio * 100, 1),
        "_speedup": round(speedup, 2),
        "_num_updates": n_updates,
    }

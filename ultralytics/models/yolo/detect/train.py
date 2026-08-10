# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
from copy import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, TQDM
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class DetectionTrainer(BaseTrainer):
    """A class extending the BaseTrainer class for training based on a detection model.

    This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models for
    object detection including dataset building, data loading, preprocessing, and model configuration.

    Attributes:
        model (DetectionModel): The YOLO detection model being trained.
        data (dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (tuple): Names of the loss components used in training (box_loss, cls_loss, dfl_loss).

    Methods:
        build_dataset: Build YOLO dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.
        preprocess_batch: Preprocess a batch of images by scaling and converting to float.
        set_model_attributes: Set model attributes based on dataset information.
        get_model: Return a YOLO detection model.
        get_validator: Return a validator for model evaluation.
        label_loss_items: Return a loss dictionary with labeled training loss items.
        progress_string: Return a formatted string of training progress.
        plot_training_samples: Plot training samples with their annotations.
        plot_training_labels: Create a labeled training plot of the YOLO model.
        auto_batch: Calculate optimal batch size based on model memory requirements.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo26n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize a DetectionTrainer object for training YOLO object detection models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (dict, optional): Dictionary of callback functions to be executed during training.
        """
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    max(self.stride, int(self.args.imgsz * (1.0 - self.args.multi_scale))),  # min imgsz
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),  # max imgsz
                )
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def _afss_prediction_tensor(self, preds):
        """Normalize different YOLO detection head outputs to a tensor consumable by NMS."""
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if not isinstance(preds, dict):
            return preds

        preds = preds.get("one2one", preds)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if isinstance(preds, dict):
            head = unwrap_model(self.model).model[-1]
            if {"boxes", "scores", "feats"}.issubset(preds) and hasattr(head, "_inference"):
                return head._inference(preds)
            if "boxes" in preds and "scores" in preds and hasattr(head, "dfl") and hasattr(head, "decode_bboxes"):
                if "feats" in preds and hasattr(head, "_get_decode_boxes"):
                    boxes = head._get_decode_boxes(preds)
                else:
                    boxes = head.decode_bboxes(head.dfl(preds["boxes"]), head.anchors.unsqueeze(0)) * head.strides
                return torch.cat((boxes, preds["scores"].sigmoid()), 1)
        return preds

    def _afss_preprocess_batch(self, batch: dict) -> dict:
        """Preprocess AFSS eval batches without training-time multi-scale resizing."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        return batch

    def compute_per_image_metrics(self, iou_thresh: float = 0.5, conf_thresh: float = 0.25):
        """Compute per-image precision and recall on the training set for AFSS."""
        model = self.ema.ema if self.ema else unwrap_model(self.model)
        was_training = model.training
        model.eval()

        if not hasattr(self, "_afss_eval_loader") or self._afss_eval_loader is None:
            gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
            self._afss_eval_dataset = build_yolo_dataset(
                self.args,
                self.data["train"],
                self.batch_size,
                self.data,
                mode="val",
                rect=False,
                stride=gs,
            )
            eval_workers = max(0, min(self.args.workers, 4))
            self._afss_eval_loader = build_dataloader(
                self._afss_eval_dataset,
                batch=self.batch_size * 2,
                workers=eval_workers,
                shuffle=False,
                rank=-1,
            )

        eval_loader = self._afss_eval_loader
        n_images = len(self._afss_eval_dataset)
        if self.afss_manager and n_images != self.afss_manager.num_images:
            LOGGER.warning(
                "AFSS: eval dataset has %d images but training sampler has %d; metrics will be truncated.",
                n_images,
                self.afss_manager.num_images,
            )
            n_images = min(n_images, self.afss_manager.num_images)

        precisions = np.zeros(n_images, dtype=np.float32)
        recalls = np.zeros(n_images, dtype=np.float32)
        nc = int(self.data["nc"])
        class_precisions = np.ones((n_images, nc), dtype=np.float32)
        class_recalls = np.ones((n_images, nc), dtype=np.float32)
        class_present = np.zeros((n_images, nc), dtype=bool)
        base_model = unwrap_model(self.model)
        end2end = getattr(model, "end2end", getattr(base_model, "end2end", False))

        seen = 0
        with torch.no_grad():
            pbar = TQDM(eval_loader, desc="AFSS: Computing per-image metrics")
            for batch in pbar:
                batch = self._afss_preprocess_batch(batch)
                preds = self._afss_prediction_tensor(model(batch["img"]))
                preds_nms = non_max_suppression(
                    preds,
                    conf_thres=conf_thresh,
                    iou_thres=iou_thresh,
                    multi_label=True,
                    agnostic=self.args.single_cls,
                    max_det=self.args.max_det,
                    nc=self.data["nc"],
                    end2end=end2end,
                )

                for si, pred in enumerate(preds_nms):
                    img_idx = seen + si
                    if img_idx >= n_images:
                        break

                    device = batch["img"].device
                    batch_idx_mask = batch["batch_idx"] == si
                    gt_cls = batch["cls"][batch_idx_mask].squeeze(-1).to(device)
                    gt_bboxes = batch["bboxes"][batch_idx_mask].to(device)
                    n_gt = len(gt_cls)
                    n_pred = len(pred)
                    if n_gt:
                        gt_cls_np = gt_cls.detach().cpu().numpy().astype(int)
                        gt_cls_np = gt_cls_np[(gt_cls_np >= 0) & (gt_cls_np < nc)]
                        class_present[img_idx, gt_cls_np] = True

                    if n_gt == 0 and n_pred == 0:
                        precisions[img_idx] = 1.0
                        recalls[img_idx] = 1.0
                    elif n_gt == 0:
                        precisions[img_idx] = 0.0
                        recalls[img_idx] = 1.0
                    elif n_pred == 0:
                        precisions[img_idx] = 1.0
                        recalls[img_idx] = 0.0
                        for cls_id in np.where(class_present[img_idx])[0]:
                            class_precisions[img_idx, cls_id] = 1.0
                            class_recalls[img_idx, cls_id] = 0.0
                    else:
                        pred_boxes = pred[:, :4]
                        img_h, img_w = batch["img"].shape[2:]
                        gt_xyxy = gt_bboxes.clone()
                        gt_xyxy[:, 0] = (gt_bboxes[:, 0] - gt_bboxes[:, 2] / 2) * img_w
                        gt_xyxy[:, 1] = (gt_bboxes[:, 1] - gt_bboxes[:, 3] / 2) * img_h
                        gt_xyxy[:, 2] = (gt_bboxes[:, 0] + gt_bboxes[:, 2] / 2) * img_w
                        gt_xyxy[:, 3] = (gt_bboxes[:, 1] + gt_bboxes[:, 3] / 2) * img_h

                        iou = box_iou(gt_xyxy, pred_boxes)
                        matched_gt: set[int] = set()
                        matched_pred: set[int] = set()
                        tp = 0
                        tp_per_class = np.zeros(nc, dtype=np.float32)
                        while iou.numel():
                            max_iou, max_idx = iou.flatten().max(0)
                            if max_iou < iou_thresh:
                                break
                            gi = max_idx.item() // iou.shape[1]
                            pi = max_idx.item() % iou.shape[1]
                            if gi in matched_gt or pi in matched_pred:
                                iou[gi, pi] = 0
                                continue
                            matched_gt.add(gi)
                            matched_pred.add(pi)
                            if int(gt_cls[gi]) == int(pred[pi, 5]):
                                tp += 1
                                cls_id = int(gt_cls[gi])
                                if 0 <= cls_id < nc:
                                    tp_per_class[cls_id] += 1
                            iou[gi, pi] = 0

                        fp = n_pred - tp
                        fn = n_gt - tp
                        precisions[img_idx] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recalls[img_idx] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        gt_ids = gt_cls.detach().cpu().numpy().astype(int)
                        gt_ids = gt_ids[(gt_ids >= 0) & (gt_ids < nc)]
                        gt_counts = np.bincount(gt_ids, minlength=nc)[:nc]
                        pred_ids = pred[:, 5].detach().cpu().numpy().astype(int)
                        pred_ids = pred_ids[(pred_ids >= 0) & (pred_ids < nc)]
                        pred_counts = np.bincount(pred_ids, minlength=nc)[:nc]
                        for cls_id in np.where(gt_counts > 0)[0]:
                            cls_tp = tp_per_class[cls_id]
                            cls_fp = pred_counts[cls_id] - cls_tp
                            cls_fn = gt_counts[cls_id] - cls_tp
                            class_precisions[img_idx, cls_id] = (
                                cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 1.0
                            )
                            class_recalls[img_idx, cls_id] = (
                                cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0.0
                            )

                seen += len(preds_nms)
                if seen >= n_images:
                    break

        model.train(was_training)
        return np.arange(n_images), precisions, recalls, class_precisions, class_recalls, class_present

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        if getattr(self.model, "end2end"):
            self.model.set_head_attr(max_det=self.args.max_det)

    def set_class_weights(self):
        """Compute and set class weights for handling class imbalance.

        Class weights are computed based on inverse class frequency in the training dataset,
        raised to the power of cls_pw (0 < cls_pw <= 1 dampens, cls_pw > 1 amplifies).
        Final weights are normalized so their mean equals 1.0.
        """
        assert 0 <= self.args.cls_pw <= 1.0, "cls_pw must be in the range [0, 1]"
        if self.args.cls_pw == 0.0:
            return
        classes = np.concatenate([lb["cls"].flatten() for lb in self.train_loader.dataset.labels], 0)
        class_counts = np.bincount(classes.astype(int), minlength=self.data["nc"]).astype(np.float32)
        class_counts = np.where(class_counts == 0, 1.0, class_counts)

        weights = (1.0 / class_counts) ** self.args.cls_pw  # apply power directly
        weights = weights / weights.mean()  # normalize so mean equals 1.0
        self.model.class_weights = torch.from_numpy(weights).to(self.device)
        LOGGER.info(f"Class weights: {self.model.class_weights.cpu().numpy().round(3)}")

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """Return a loss dict with labeled training loss items tensor.

        Args:
            loss_items (list[float], optional): List of loss values.
            prefix (str): Prefix for keys in the returned dictionary.

        Returns:
            (dict | list): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot training samples with their annotations.

        Args:
            batch (dict[str, Any]): Dictionary containing batch data.
            ni (int): Batch index used for naming the output file.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        n = len(train_dataset)
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj, dataset_size=n)

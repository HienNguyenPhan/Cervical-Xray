from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.ops as ops
from lightning import LightningModule
from src.models.components import ResNet101Backbone
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import MeanMetric

class MaskRCNN(LightningModule):
    def __init__(
        self,
        num_classes: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.backbone = ResNet101Backbone()
        self.rpn = ops.RPNHead(256, 9)
        self.roi_align = ops.RoIAlign(output_size=(7, 7), spatial_scale=1/16, sampling_ratio=2)
        self.mask_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.map_metric = MeanAveragePrecision()

    def forward(self, x, proposals):
        features = self.backbone(x)
        rpn_logits, rpn_bbox = self.rpn(features[-1])
        roi_pooled = self.roi_align(features[-1], proposals)
        masks = self.mask_head(roi_pooled)
        return rpn_logits, rpn_bbox, masks

    def model_step(self, batch):
        images, targets = batch
        proposals = [t["boxes"] for t in targets]
        rpn_logits, rpn_bbox, masks = self.forward(images, proposals)
        
        loss = self.criterion(masks, torch.cat([t["masks"] for t in targets]))
        return loss, masks, targets

    def training_step(self, batch, batch_idx):
        loss, masks, targets = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, masks, targets = self.model_step(batch)
        self.val_loss(loss)
        self.map_metric.update(masks, targets)
        self.log("val/loss", self.val_loss, on_epoch=True, prog_bar=True)
        self.log("val/mAP", self.map_metric.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, masks, targets = self.model_step(batch)
        self.test_loss(loss)
        self.map_metric.update(masks, targets)
        self.log("test/loss", self.test_loss, on_epoch=True, prog_bar=True)
        self.log("test/mAP", self.map_metric.compute(), prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

from typing import Any, Dict, Tuple, Optional

import hydra
import torch
import rootutils
import numpy as np
from omegaconf import DictConfig
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.detection import MeanAveragePrecision
from src.model.mask_rcnn import MaskRCNN  # Import your MaskRCNN class

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class model_module(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        compile: bool = False,
        num_classes: int = 7,  # Specify the number of classes (including background)
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.num_classes = num_classes

        # Loss function (Mask R-CNN handles its own losses)

        # Metric objects for detection and segmentation
        self.train_map = MeanAveragePrecision(num_classes=num_classes)
        self.val_map = MeanAveragePrecision(num_classes=num_classes)
        self.test_map = MeanAveragePrecision(num_classes=num_classes)

        # for averaging loss across batches (we'll track the losses returned by Mask R-CNN)
        self.train_loss_rpn = MeanMetric()
        self.train_loss_classifier = MeanMetric()
        self.train_loss_mask = MeanMetric()
        self.val_loss_rpn = MeanMetric()
        self.val_loss_classifier = MeanMetric()
        self.val_loss_mask = MeanMetric()
        self.test_loss_rpn = MeanMetric()
        self.test_loss_classifier = MeanMetric()
        self.test_loss_mask = MeanMetric()

        # for tracking best so far validation mAP
        self.val_map_best = MinMetric() # Changed to MinMetric for loss, consider MaxMetric for mAP

    def forward(self, x: torch.Tensor, targets: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        return self.net(x, targets)

    def on_train_start(self) -> None:
        self.val_loss_rpn.reset()
        self.val_loss_classifier.reset()
        self.val_loss_mask.reset()
        self.val_map.reset()
        self.val_map_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, List[Dict]]) -> Tuple[Dict[str, torch.Tensor], List[Dict]]:
        images, targets = batch
        outputs = self.forward(images, targets)
        if self.training:
            losses = outputs[0]
            predictions = None
        else:
            losses = {}
            predictions = outputs
        return losses, predictions, targets

    def training_step(
        self, batch: Tuple[torch.Tensor, List[Dict]], batch_idx: int) -> torch.Tensor:
        losses, _, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss_rpn(losses.get("loss_rpn", torch.tensor(0.0, device=self.device)))
        self.train_loss_classifier(losses.get("loss_classifier", torch.tensor(0.0, device=self.device)))
        self.train_loss_mask(losses.get("loss_mask", torch.tensor(0.0, device=self.device)))

        self.log("train/loss_rpn", self.train_loss_rpn, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_classifier", self.train_loss_classifier, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_mask", self.train_loss_mask, on_step=False, on_epoch=True, prog_bar=True)

        total_loss = sum(loss for loss in losses.values())
        self.log("train/loss_total", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return total_loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, List[Dict]], batch_idx: int) -> None:
        losses, predictions, targets = self.model_step(batch)

        # update and log losses
        self.val_loss_rpn(losses.get("loss_rpn", torch.tensor(0.0, device=self.device)))
        self.val_loss_classifier(losses.get("loss_classifier", torch.tensor(0.0, device=self.device)))
        self.val_loss_mask(losses.get("loss_mask", torch.tensor(0.0, device=self.device)))

        self.log("val/loss_rpn", self.val_loss_rpn, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_classifier", self.val_loss_classifier, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_mask", self.val_loss_mask, on_step=False, on_epoch=True, prog_bar=True)

        if predictions:
            self.val_map.update(predictions, targets)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_map.compute()
        map_value = metrics.get("map", torch.tensor(0.0, device=self.device))
        self.val_map_best(map_value)
        self.log("val/map", map_value, sync_dist=True, prog_bar=True)
        self.log("val/map_best", self.val_map_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, List[Dict]], batch_idx: int) -> None:
        losses, predictions, targets = self.model_step(batch)

        # update and log losses
        self.test_loss_rpn(losses.get("loss_rpn", torch.tensor(0.0, device=self.device)))
        self.test_loss_classifier(losses.get("loss_classifier", torch.tensor(0.0, device=self.device)))
        self.test_loss_mask(losses.get("loss_mask", torch.tensor(0.0, device=self.device)))

        self.log("test/loss_rpn", self.test_loss_rpn, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss_classifier", self.test_loss_classifier, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss_mask", self.test_loss_mask, on_step=False, on_epoch=True, prog_bar=True)

        if predictions:
            self.test_map.update(predictions, targets)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_map.compute()
        self.log_dict(metrics, sync_dist=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/map_best", # Changed monitor to mAP
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

@hydra.main(version_base="1.3", config_path="../../configs/model", config_name="model.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    model: LightningModule = hydra.utils.instantiate(cfg)

if __name__ == "__main__":
    main()

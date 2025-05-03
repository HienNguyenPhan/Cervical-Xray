from typing import Any, Dict, Optional, Tuple

import torch
import hydra
import rootutils
import albumentations as A
from omegaconf import DictConfig
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from src.data.components.dataset import BaseDataset  # Assuming your BaseDataset is in this file
from src.data.components.dataset import CervicalDataset  # Import the corrected CervicalDataset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class DataModule(LightningDataModule):
    def __init__(
        self,
        xml_path: str,
        train_test_split: Tuple[float, float] = (0.9, 0.1),
        train_batch_size: int = 32,
        test_batch_size: int = 64,
        num_workers: int = 4,
        train_transforms: Optional[A.Compose] = None,
        test_transforms: Optional[A.Compose] = None,
        pin_memory: bool = False,
        class_names: Tuple[str, ...] = ('C2', 'C2_lower', 'C3', 'C4', 'C5', 'C6', 'C7'),
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.xml_path = xml_path
        self.class_names = class_names

        # data transformations
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.train_batch_size_per_device = train_batch_size
        self.test_batch_size_per_device = test_batch_size

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.train_batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.train_batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.train_batch_size_per_device = self.hparams.train_batch_size // self.trainer.world_size
            self.test_batch_size_per_device = self.hparams.test_batch_size // self.trainer.world_size

        if not self.data_train and not self.data_val and not self.data_test:
            full_dataset = CervicalDataset(
                xml_path=self.xml_path,
                class_names=self.class_names,
                mode='train',  
                transform=None
            )
            train_len = int(len(full_dataset) * self.hparams.train_test_split[0])
            test_len = len(full_dataset) - train_len
            self.data_train, self.data_test = random_split(
                full_dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42)
            )
            self.data_val = self.data_test
            self.data_train.dataset.transform = self.train_transforms
            self.data_val.dataset.transform = self.test_transforms
            self.data_test.dataset.transform = self.test_transforms

    def train_dataloader(self):
        def collate_fn(batch):
            batch = [b for b in batch if b is not None and isinstance(b, tuple) and len(b) == 2]
            if not batch:
                raise ValueError("Empty batch after filtering")

            images = [item[0] for item in batch]
            targets = [item[1] for item in batch]

            try:
                images = torch.stack(images)
            except RuntimeError as e:
                print(f"Error stacking images: {e}")
                # Có thể thêm xử lý fallback nếu cần
                raise e
            return images, targets

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.train_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        def collate_fn(batch):
            images = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            images = torch.stack(images)
            return images, targets

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.test_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        def collate_fn(batch):
            images = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            images = torch.stack(images)
            return images, targets

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.test_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass

@hydra.main(version_base="1.3", config_path="../../configs/data", config_name="data")
def main(cfg: DictConfig) -> Optional[float]:
    datamodule: LightningDataModule = hydra.utils.instantiate(config=cfg)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    images, targets = batch
    print("Image batch shape:", images.shape)
    print("  ber of targets:", len(targets))
    print("Example target keys:", targets[0].keys())
    print("Example target boxes shape:", targets[0]['boxes'].shape)
    print("Example target labels shape:", targets[0]['labels'].shape)
    print("Example target masks shape:", targets[0]['masks'].shape)

if __name__ == "__main__":
    main()
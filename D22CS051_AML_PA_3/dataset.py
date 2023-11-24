import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import config


class GenericDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        datasets.Food101(root=self.data_dir, split="train", download=True)
        datasets.Food101(root=self.data_dir, split="test", download=True)
        # datasets.Flowers102(root=self.data_dir,split="train",download=True)
        # datasets.Flowers102(root=self.data_dir,split="test",download=True)
        # datasets.Flowers102(root=self.data_dir,split="val",download=True)

    def setup(self, stage):
        entire_dataset = datasets.Food101(
            root=self.data_dir,
            split="train",
            transform=transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            download=False,
        )

        # spliting train test val
        self.train_ds, self.val_ds = random_split(entire_dataset, [int(len(entire_dataset)*0.75), len(entire_dataset) - int(len(entire_dataset)*0.75)])
        

        self.test_ds = datasets.Food101(
            root=self.data_dir,
            split="test",
            transform=transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
            ]),
            download=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


if __name__ == "__main__":
    dm = GenericDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    dm.prepare_data()

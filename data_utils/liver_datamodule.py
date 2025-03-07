#%%
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, List
from liver_dataset import LiverDataset

class LiverDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_list_path: Optional[str] = None,
                 data_dir: Optional[str] = None,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 train_folds: List[str] = ["training"],
                 val_folds: List[str] = ["validation"],
                 test_folds: List[str] = ["testing"],
                 num_samples: int = 1):
        super().__init__()
        
        self.data_list_path = data_list_path or "/neodata/open_dataset/ConDistFL/data/Liver/datalist.json"
        self.data_dir = data_dir or "/neodata/open_dataset/ConDistFL/data/Liver"
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_folds = train_folds
        self.val_folds = val_folds
        self.test_folds = test_folds
        self.num_samples = num_samples

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_dataset = LiverDataset(
                folds=self.train_folds,
                mode="train",
                data_list_path=self.data_list_path,
                data_dir=self.data_dir,
                num_samples=self.num_samples
            )
            
            self.val_dataset = LiverDataset(
                folds=self.val_folds,
                mode="validate",
                data_list_path=self.data_list_path,
                data_dir=self.data_dir
            )

        if stage in ("test", None):
            self.test_dataset = LiverDataset(
                folds=self.test_folds,
                mode="infer",
                data_list_path=self.data_list_path,
                data_dir=self.data_dir
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
#%%







#%%
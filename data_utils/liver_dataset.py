#%%
import os
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Literal, Optional

from transforms import get_transforms

class LiverDataset(Dataset):
    def __init__(self,
                 folds: List[str],
                 mode: Literal["train", "validate", "infer"] = "train",
                 data_list_path: Optional[str] = None,
                 data_dir: Optional[str] = None,
                 num_samples: int = 1,
                 transform=None):
        """
        Args:
            folds: List of fold names (e.g., ['training'], ['validation'], ['testing']).
            mode: Operation mode ("train", "validate", "infer").
            data_list_path: Path to JSON file containing dataset paths.
            data_dir: Path to data directory.
            num_samples: Number of augmented samples per original image (for data augmentation).
            transform: Optional custom transform function.
        """
        self.data_list_path = data_list_path or "/neodata/open_dataset/ConDistFL/data/Liver/datalist.json"
        self.data_dir = data_dir or "/neodata/open_dataset/ConDistFL/data/Liver"
        
        with open(self.data_list_path, 'r') as f:
            self.data_list = json.load(f)
        
        self.samples = []
        for fold in folds:
            self.samples.extend(self.data_list[fold])

        # Transform initialization
        self.transform = transform or get_transforms(mode=mode, num_samples=num_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        image_path = os.path.join(self.data_dir, sample["image"])
        label_path = os.path.join(self.data_dir, sample["label"])

        data_item = {
            "image": image_path,
            "label": label_path
        }

        transformed = self.transform(data_item)

        if isinstance(transformed, list):
            transformed = transformed[0]

        return transformed


#%%



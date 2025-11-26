from wilds import get_dataset
from torch.utils.data import Dataset
from torchvision.datasets import Caltech256, ImageFolder
from torch.utils.data import DataLoader
import torch
import os
import pathlib
from PIL import Image

class BaseDataset(Dataset):
    def __init__(self, dataset_name, root_dir, download=True):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.download = download
        self.dataset = get_dataset(dataset=self.dataset_name, root_dir=self.root_dir, download=self.download)
        
    def get_splits(self):
        raise NotImplementedError("Subclasses should implement this method.")

class NoMetaDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.collate = getattr(base_dataset, "collate", None)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y, *_ = self.base_dataset[idx]
        return x, y

def get_custom_loader(
    wilds_dataset,
    split: str,
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    drop_last: bool = False,
):
    subset = wilds_dataset.get_subset(split, transform=transform)

    return DataLoader(
        NoMetaDataset(subset),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=drop_last,
        collate_fn=getattr(subset, "collate", None)
    )
    
# Works but slow
class IWildCamDataset (BaseDataset):
    num_classes = 182
    def __init__(self, root_dir, ood_root_dir=None, download=False):
        super().__init__('iwildcam', root_dir, download)
        # self.num_classes = self.dataset.n_classes

    def get_splits(self, transforms, batch_size, num_workers):
        if transforms is None:
            transforms = {
                'train': None,
                'val': None,
                'test': None
            }
        return {
            "train": get_custom_loader(self.dataset, "train", transforms["train"], batch_size, num_workers, shuffle=True),
            "val": get_custom_loader(self.dataset, "id_val", transforms["val"], batch_size, num_workers),
            "test": get_custom_loader(self.dataset, "id_test", transforms["test"], batch_size, num_workers),
            "ood_test": get_custom_loader(self.dataset, "test", transforms["test"], batch_size, num_workers),
        }

class Caltech256Dataset(Dataset):
    num_classes = 257
    def __init__(self, root_dir, ood_root_dir=None, download=True):
        self.root_dir = root_dir
        self.download = download

    def get_splits(self, transforms, batch_size, num_workers,train_perc = 0.7,val_perc = 0.15):
        if transforms is None:
            transforms = {
                'train': None,
                'val': None,
                'test': None
            }
        full_dataset = Caltech256(root=self.root_dir, transform=None, download=self.download)
        train_size = int(train_perc * len(full_dataset))
        val_size = int(val_perc * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        generator = torch.Generator().manual_seed(42)
        splits = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=generator
        )
        train_dataset, val_dataset, test_dataset = splits

        # Wrap subsets with custom transform behavior
        train_dataset.dataset.transform = transforms["train"]
        val_dataset.dataset.transform = transforms["val"]
        test_dataset.dataset.transform = transforms["test"]

        return {
            "train": DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                pin_memory=True,prefetch_factor=4,
                              persistent_workers=True,drop_last=True,),
            "val": DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                pin_memory=True,prefetch_factor=4,),
            "test": DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                pin_memory=True,prefetch_factor=4,),
            "ood_test": None,
        }
        
# class IntegerLabelImageFolder(ImageFolder):
#     def find_classes(self, directory):
#         classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
#         # No sorting, map folder name (as string) to its int value
#         class_to_idx = {cls_name: int(cls_name) for cls_name in classes}
#         return classes, class_to_idx
    
class IntegerLabelDataset(Dataset):
    """Dataset that assumes folders are named 0, 1, 2, ..., 999."""
    def __init__(self, root, transform=None):
        self.root = pathlib.Path(root)
        self.transform = transform
        # Collect all .jpeg or .jpg files recursively
        self.fnames = list(self.root.rglob("*.jp*g"))
        assert len(self.fnames) > 0, f"No images found under {root}"

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        img_path = self.fnames[i]
        img = Image.open(img_path).convert("RGB")
        label = int(img_path.parent.name)  # folder name is the class number
        if self.transform:
            img = self.transform(img)
        return img, label
    
class ImageNetDataset(Dataset):
    num_classes = 1000
    def __init__(self, root_dir, ood_root_dir=None, download=False):
        self.root_dir = root_dir
        self.download = download
        self.ood_root_dir=ood_root_dir
    def get_splits(self, transforms, batch_size, num_workers,val_perc = 0.05):
        if transforms is None:
            transforms = {
                'train': None,
                'val': None,
                'test': None
            }
        train_dataset = ImageFolder(os.path.join(self.root_dir, "train"), transform=transforms["train"])
        full_dataset = ImageFolder(os.path.join(self.root_dir, "val"), transform=None)
        generator = torch.Generator().manual_seed(42)
        val_size = int(val_perc * len(full_dataset))
        test_size = len(full_dataset) - val_size
        splits = torch.utils.data.random_split(
            full_dataset, [val_size, test_size],
            generator=generator
        )
        val_dataset, test_dataset = splits
        val_dataset.dataset.transform = transforms["val"]
        test_dataset.dataset.transform = transforms["test"]
        
        ood_test_loader = None

        if self.ood_root_dir:
            ood_test_dataset = IntegerLabelDataset(self.ood_root_dir, transform=transforms["test"])
            ood_test_loader = DataLoader(
                ood_test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=4,
                persistent_workers=True,
            )
            
        return {
            "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            prefetch_factor=4, persistent_workers=True),
            "val": DataLoader(val_dataset,batch_size=batch_size,
                              shuffle=False,num_workers=num_workers,
                              pin_memory=True,prefetch_factor=4,
                              persistent_workers=True,),
            "test": DataLoader(test_dataset,batch_size=batch_size,
                               shuffle=False,num_workers=num_workers,
                               pin_memory=True,prefetch_factor=4,
                              persistent_workers=True,),
            "ood_test": ood_test_loader,
        }


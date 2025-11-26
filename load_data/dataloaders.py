# dataloaders.py
from .datasets import IWildCamDataset, Caltech256Dataset, ImageNetDataset

dataset_classes = {
    'iwildcam': IWildCamDataset,
    'caltech256': Caltech256Dataset,
    'imagenet': ImageNetDataset
}

def get_dataloaders(dataset_name, root_dir, batch_size, num_workers, transforms, ood_root_dir=None):
    dataset_name = dataset_name.lower()
    if dataset_name not in dataset_classes:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    dataset = dataset_classes[dataset_name](root_dir=root_dir,ood_root_dir = ood_root_dir)
    return dataset.get_splits(
        transforms=transforms,
        batch_size=batch_size,
        num_workers=num_workers
    )

def get_dataset_class(dataset_name):
    return dataset_classes[dataset_name]
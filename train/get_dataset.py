from dataset import LIDCDataset, DEFAULTDataset, SingleDataGenerator, SingleDataGenerator_real, LNDb, LUNA
from torch.utils.data import WeightedRandomSampler


def get_dataset(cfg):
    if cfg.dataset.name == 'LIDC':
        # train_dataset = LIDCDataset(augmentation=False)
        # val_dataset = LIDCDataset(augmentation=False)
        train_dataset = SingleDataGenerator(mode="train")
        val_dataset = SingleDataGenerator(mode="val")
        visual_dataset = SingleDataGenerator(mode="visual")
        real_dataset = SingleDataGenerator_real(mode="real")
        sampler = None
        return train_dataset, val_dataset, visual_dataset, real_dataset
    if cfg.dataset.name == 'DEFAULT':
        train_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir)
        val_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir)
        sampler = None
    raise ValueError(f'{cfg.dataset.name} Dataset is not available')

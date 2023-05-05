from geotransformer.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)
import torch


# todo : 스파게티 코드 수정
import sys
sys.path.append("/data/wlsgur4011/part_assembly")

from multi_part_assembly.datasets.geometry_data import build_geometry_dataset, build_geometry_dataloader
from util.data import PairBreakingBadDataset
import jhutil

def train_valid_data_loader(cfg, distributed):
    train_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
    )

    valid_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        'val',
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
    )
    valid_dataset = jhutil.GeoTransformerPseudoData()
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg, benchmark):
    train_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
    )
    torch.multiprocessing.set_start_method('spawn')
    cfg2 = jhutil.load_yaml("/data/wlsgur4011/part_assembly/yamls/data_example.yaml")
    train_data, val_data = build_geometry_dataset(cfg2)
    train_dataset = PairBreakingBadDataset(train_data)
    
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    test_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        benchmark,
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
    )
    
    test_dataset = PairBreakingBadDataset(train_data)
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )

    return test_loader, neighbor_limits


def test_dataset(cfg, benchmark):
    test_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        benchmark,
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
    )
    return test_dataset
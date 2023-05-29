import jhutil
from multi_part_assembly.datasets.geometry_data import build_geometry_dataset, build_geometry_dataloader
from geotransformer.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)
import torch
from functools import partial
import numpy as np

# todo : 스파게티 코드 수정
import sys
sys.path.append("/data/wlsgur4011/part_assembly")


def train_valid_data_loader(cfg, distributed, part_assembly=True):

    if part_assembly:
        from part_assembly.stage3_data import Stage3PairDataset

        datafolder = "/data/wlsgur4011/DataCollection/BreakingBad/data_split/"
        artifact_train = f"{datafolder}artifact.train.pth"
        artifact_val = f"{datafolder}artifact.val.pth"
        train_dataset = Stage3PairDataset(artifact_train)
        valid_dataset = Stage3PairDataset(artifact_val)
    else:
        train_dataset = ThreeDMatchPairDataset(
            cfg.data.dataset_root,
            'train',
            point_limit=cfg.train.point_limit,
            use_augmentation=cfg.train.use_augmentation,
            augmentation_noise=cfg.train.augmentation_noise,
            augmentation_rotation=cfg.train.augmentation_rotation,
        )
        valid_dataset = ThreeDMatchPairDataset(
            cfg.data.dataset_root,
            'val',
            point_limit=cfg.test.point_limit,
            use_augmentation=False,
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


def test_data_loader(cfg, benchmark, part_assembly=True):
    if part_assembly:
        from part_assembly.stage3_data import Stage3PairDataset
        # FIXME: 스파게티
        cfg2 = jhutil.load_yaml("/data/wlsgur4011/part_assembly/yamls/data_example.yaml")
        dataname = cfg2.data.data_fn.split(".")[0]

        datafolder = "/data/wlsgur4011/DataCollection/BreakingBad/data_split"
        artifact_train = f"{datafolder}/{dataname}.train.pth"

        jhutil.jhprint(0000, artifact_train)
        train_dataset = Stage3PairDataset(artifact_train)
        test_dataset = train_dataset
    else:
        train_dataset = ThreeDMatchPairDataset(
            cfg.data.dataset_root,
            'train',
            point_limit=cfg.train.point_limit,
            use_augmentation=cfg.train.use_augmentation,
            augmentation_noise=cfg.train.augmentation_noise,
            augmentation_rotation=cfg.train.augmentation_rotation,
        )
        test_dataset = ThreeDMatchPairDataset(
            cfg.data.dataset_root,
            benchmark,
            point_limit=cfg.test.point_limit,
            use_augmentation=False,
        )

    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

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

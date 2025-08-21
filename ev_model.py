# File: model_configs/ev_model.py
# YOLOX-S with P2 feature level for EV dataset (2048x1460)
# Optimized for GTX 1080Ti (11GB VRAM) and class imbalance (80% chips, 20% checks)

_base_ = 'mmdet::yolox/yolox_s_8xb8-300e_coco.py'

# === MODEL ARCHITECTURE (P2 ENABLED) ===
model = dict(
    # Disable batch augmentations that can harm tiny object detection
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=1,  # Allow native image sizes
        batch_augments=None  # Disable Mosaic/MixUp initially
    ),
    
    # Backbone: CSPDarknet with P2 output
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=0.33,  # YOLOX-S depth
        widen_factor=0.5,    # YOLOX-S width
        out_indices=(0, 1, 2, 3),  # KEY: Output C2, C3, C4, C5 (includes stride-4 features)
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    
    # Neck: YOLOX PANet with P2 support
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512, 1024],  # KEY: Accept C2, C3, C4, C5
        out_channels=128,
        num_csp_blocks=1,
        num_outs=4,  # KEY: Generate P2, P3, P4, P5
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    
    # Detection Head: YOLOX head with P2 support and class imbalance handling
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=2,  # chip, check
        in_channels=128,
        feat_channels=128,
        strides=[4, 8, 16, 32],  # KEY: Include stride-4 for P2
        
        # Loss functions optimized for class imbalance
        loss_cls=dict(
            type='FocalLoss',  # Better for imbalanced classes than CrossEntropy
            use_sigmoid=True,
            gamma=2.0,        # Focus on hard examples
            alpha=0.25,       # Weight for rare class (checks)
            reduction='sum',
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0   # Higher weight for bbox regression
        ),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0
        ),
        loss_l1=dict(
            type='L1Loss',
            reduction='sum',
            loss_weight=1.0
        ),
        
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    
    # Test configuration optimized for tiny objects
    test_cfg=dict(
        score_thr=0.01,  # Low threshold to catch small objects
        nms=dict(type='soft_nms', iou_threshold=0.55),  # Soft-NMS for overlapping thin objects
        max_per_img=500  # Allow many detections per image
    )
)

# === DATASET CONFIGURATION (EV ONLY) ===
dataset_type = 'CocoDataset'
data_root = 'EV_dataset_processed/'  # KEY DIFFERENCE: EV dataset
metainfo = dict(classes=('chip', 'check'))

# Training pipeline - minimal augmentation to preserve tiny objects
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # Light augmentations that don't destroy tiny objects
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=16,
        contrast_range=(0.9, 1.1),
        saturation_range=(0.9, 1.1),
        hue_delta=9
    ),
    dict(type='PackDetInputs')
]

# Test pipeline - no augmentation
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape')
    )
]

# Training dataloader - smaller batch size for larger EV images
train_dataloader = dict(
    batch_size=2,  # Smaller batch for larger images (2048x1460)
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty=False),  # Keep clean images for hard negative mining
        pipeline=train_pipeline
    )
)

# Validation dataloader
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader

# Evaluation configuration
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    # Focus on small object performance
    metric_items=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
)
test_evaluator = val_evaluator

# === TRAINING SCHEDULE ===
# Optimizer - YOLOX uses SGD with specific parameters
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.008,  # Slightly lower LR for larger images
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    ),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.)
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=1000  # Warmup iterations
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=150,  # Total epochs
        eta_min=1e-5,
        begin=1000,
        end=150000,  # Approximate total iterations
        by_epoch=False
    )
]

# Training loop configuration
max_epochs = 150  # Sufficient for small dataset
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5  # Validate every 5 epochs
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Custom hooks for YOLOX
custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=15,  # Disable augmentation in last 15 epochs
        priority=48
    ),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49
    )
]

# Checkpoint and logging
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP_s'  # Save best model based on small object mAP
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# Visualization and logging
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Environment settings
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)


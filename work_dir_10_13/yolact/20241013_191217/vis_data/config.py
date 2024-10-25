auto_scale_lr = dict(base_batch_size=8)
backend_args = None
custom_hooks = [
    dict(interval=50, priority='VERY_LOW', type='CheckInvalidLossHook'),
]
data_root = 'data/Cocoon_Morbid_coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_norm_cfg = dict(
    mean=[
        123.68,
        116.78,
        103.94,
    ], std=[
        58.4,
        57.12,
        57.38,
    ], to_rgb=True)
input_size = 550
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 100
metainfo = dict(
    classes=(
        'Crnshed',
        'Imprinted',
        'Mold',
        'Mouthed',
        'Rotting',
        'Yellow_Spotted',
    ),
    palette=[
        (
            220,
            20,
            60,
        ),
        (
            119,
            11,
            32,
        ),
        (
            0,
            0,
            142,
        ),
        (
            0,
            0,
            230,
        ),
        (
            106,
            0,
            228,
        ),
        (
            0,
            60,
            100,
        ),
    ])
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=-1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet',
        zero_init_residual=False),
    bbox_head=dict(
        anchor_generator=dict(
            base_sizes=[
                8,
                16,
                32,
                64,
                128,
            ],
            centers=[
                (
                    3.9855072463768115,
                    3.9855072463768115,
                ),
                (
                    7.857142857142857,
                    7.857142857142857,
                ),
                (
                    15.277777777777779,
                    15.277777777777779,
                ),
                (
                    30.555555555555557,
                    30.555555555555557,
                ),
                (
                    55.0,
                    55.0,
                ),
            ],
            octave_base_scale=3,
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales_per_octave=1,
            strides=[
                7.971014492753623,
                15.714285714285714,
                30.555555555555557,
                61.111111111111114,
                110.0,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                0.1,
                0.1,
                0.2,
                0.2,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(beta=1.0, loss_weight=1.5, type='SmoothL1Loss'),
        loss_cls=dict(
            loss_weight=1.0,
            reduction='none',
            type='CrossEntropyLoss',
            use_sigmoid=False),
        num_classes=6,
        num_head_convs=1,
        num_protos=32,
        type='YOLACTHead',
        use_ohem=True),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.68,
            116.78,
            103.94,
        ],
        pad_mask=True,
        std=[
            58.4,
            57.12,
            57.38,
        ],
        type='DetDataPreprocessor'),
    mask_head=dict(
        in_channels=256,
        loss_mask_weight=6.125,
        loss_segm=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        max_masks_to_train=100,
        num_classes=6,
        num_protos=32,
        type='YOLACTProtonet',
        with_seg_branch=True),
    neck=dict(
        add_extra_convs='on_input',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        start_level=1,
        type='FPN',
        upsample_cfg=dict(mode='bilinear')),
    test_cfg=dict(
        iou_thr=0.5,
        mask_thr=0.5,
        mask_thr_binary=0.5,
        max_per_img=100,
        min_bbox_size=0,
        nms_pre=1000,
        score_thr=0.05,
        top_k=200),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            gt_max_assign_all=False,
            ignore_iof_thr=-1,
            min_pos_iou=0.0,
            neg_iou_thr=0.4,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        neg_pos_ratio=3,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler')),
    type='YOLACT')
optim_wrapper = dict(
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.1, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=100,
        gamma=0.1,
        milestones=[
            20,
            42,
            49,
            52,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                550,
                550,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/Cocoon_Morbid_coco/annotations/instances_val2017.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        550,
        550,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        backend_args=None,
        data_prefix=dict(img='images/train2017/'),
        data_root='data/Cocoon_Morbid_coco/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(
            classes=(
                'Crnshed',
                'Imprinted',
                'Mold',
                'Mouthed',
                'Rotting',
                'Yellow_Spotted',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    142,
                ),
                (
                    0,
                    0,
                    230,
                ),
                (
                    106,
                    0,
                    228,
                ),
                (
                    0,
                    60,
                    100,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(min_gt_bbox_wh=(
        4.0,
        4.0,
    ), type='FilterAnnotations'),
    dict(
        mean=[
            123.68,
            116.78,
            103.94,
        ],
        ratio_range=(
            1,
            4,
        ),
        to_rgb=True,
        type='Expand'),
    dict(
        min_crop_size=0.3,
        min_ious=(
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
        ),
        type='MinIoURandomCrop'),
    dict(keep_ratio=False, scale=(
        550,
        550,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        brightness_delta=32,
        contrast_range=(
            0.5,
            1.5,
        ),
        hue_delta=18,
        saturation_range=(
            0.5,
            1.5,
        ),
        type='PhotoMetricDistortion'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='images/val2017/'),
        data_root='data/Cocoon_Morbid_coco/',
        metainfo=dict(
            classes=(
                'Crnshed',
                'Imprinted',
                'Mold',
                'Mouthed',
                'Rotting',
                'Yellow_Spotted',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    142,
                ),
                (
                    0,
                    0,
                    230,
                ),
                (
                    106,
                    0,
                    228,
                ),
                (
                    0,
                    60,
                    100,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/Cocoon_Morbid_coco/annotations/instances_val2017.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = './work_dir_10_13/yolact'

_base_ = [
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

# 模型设置
model = dict(
    type='SOLOv2',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    
    # 使用 ResNeXt 作为骨干网络
    backbone=dict(
        type='ResNeXt',  # 更改为 ResNeXt
        depth=50,  # ResNeXt 的层数（50，101 等）
        groups=32,  # ResNeXt 的组数（即 cardinality）
        base_width=4,  # ResNeXt 的 base_width，通常为 4
        num_stages=4,
        out_indices=(0, 1, 2, 3),  # 输出不同 stage 的特征
        frozen_stages=1,  # 冻结第一阶段
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnext50_32x4d'),  # 使用 ResNeXt 的预训练权重
        style='pytorch'  # PyTorch 风格
    ),
    
    # FPN 颈部网络设置
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  # 需要确保这些与 ResNeXt 输出的通道一致
        out_channels=256,
        start_level=0,
        num_outs=5
    ),
    
    # SOLOv2 Head
    mask_head=dict(
        type='SOLOV2Head',
        num_classes=6,  # 根据自定义数据集调整类别数
        in_channels=256,
        feat_channels=512,
        stacked_convs=4,
        strides=[8, 8, 16, 32, 32],
        # 调整 scale_ranges 提高对目标的检测能力
        # scale_ranges=((1, 48), (32, 96), (64, 192), (128, 384), (256, 768)),  
        # pos_scale=0.2,
        # num_grids=[40, 36, 24, 16, 12],
        scale_ranges=((1, 48), (32, 96), (64, 192), (128, 384), (256, 768)),  
        pos_scale=0.2,
        num_grids=[64, 50, 32, 24, 16],



        cls_down_index=0,
        mask_feature_head=dict(
            feat_channels=128,
            start_level=0,
            end_level=3,
            out_channels=256,
            mask_stride=4,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
        ),
        loss_mask=dict(type='DiceLoss', use_sigmoid=True, loss_weight=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)
    ),
    
    # 测试配置
    test_cfg=dict(
        nms_pre=500,
        score_thr=0.05,
        mask_thr=0.5,
        filter_thr=0.05,
        kernel='gaussian',  # 使用高斯核
        sigma=2.0,
        max_per_img=100)
)

# 优化器配置
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=1e-3,  # 根据 batch size 进行调整
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# 数据集配置
data_root = 'data/Cocoon_Morbid_coco/'
metainfo = {
    'classes': ('Crnshed', 'Imprinted', 'Mold', 
                'Mouthed', 'Rotting', 'Yellow_Spotted'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), 
                (0, 0, 230), (106, 0, 228), (0, 60, 100)]
}

# 训练数据管道
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(4.0, 4.0)),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='PackDetInputs')
]

# 测试数据管道
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# 数据加载器
train_dataloader = dict(
    batch_size=2,  # 根据显存调整 batch size
    num_workers=4,  # 根据系统性能调整
    dataset=dict(
        pipeline=train_pipeline,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='images/train2017/')
    )
)
# training schedule for 3x
max_epochs = 100
train_cfg = dict(max_epochs=max_epochs)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

val_dataloader = dict(
    batch_size=8,  # 验证时使用较小 batch size
    num_workers=2,
    dataset=dict(
        pipeline=test_pipeline,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='images/val2017/')
    )
)

# 验证与测试评估器
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/Cocoon_Morbid_coco/annotations/instances_val2017.json',
    metric=['bbox', 'segm'],
    classwise=True
)

test_evaluator = val_evaluator

# 自定义配置和其他优化
custom_hooks = [
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]

# 环境配置
env_cfg = dict(cudnn_benchmark=True)

# 自动缩放学习率
auto_scale_lr = dict(base_batch_size=4)

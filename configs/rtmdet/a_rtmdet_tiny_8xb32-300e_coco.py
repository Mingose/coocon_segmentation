_base_ = './rtmdet_s_8xb32-300e_coco.py'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'

# 模型设置
model = dict(
    backbone=dict(
        deepen_factor=0.167,
        widen_factor=0.375,
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint=checkpoint)
    ),
    neck=dict(in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(
        in_channels=96,
        feat_channels=96,
        num_classes=6,  # 设置类别数量为6
        exp_on_reg=False
    )
)

# 训练流水线设置
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0, max_cached_images=20, random_pop=False),
    dict(type='RandomResize', scale=(1280, 1280), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='CachedMixUp', img_scale=(640, 640), ratio_range=(1.0, 1.0), max_cached_images=10, random_pop=False, pad_val=(114, 114, 114), prob=0.5),
    dict(type='PackDetInputs')
]

# 数据集设置
data_root = 'data/Morbid/'
metainfo = {
    'classes': ('Crnshed', 'Imprinted', 'Mold', 
                'Mouthed', 'Rotting', 'Yellow_Spotted'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), 
                (0, 0, 230), (106, 0, 228), (0, 60, 100)]
}

# 训练数据加载器
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        pipeline=train_pipeline,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',  # 训练集注释文件
        data_prefix=dict(img='train2017/')
    )
)

# 验证数据加载器
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val2017.json',  # 验证集注释文件
        data_prefix=dict(img='val2017/')
    )
)

# 验证评估器
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox'],  # 只评估 bbox
    classwise=True
)

# 测试评估器
test_evaluator = val_evaluator

# 训练时间设置，最大 epoch 数设为 20
# max_epochs = 20
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器和学习率调度
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,  # 与 max_epochs 保持一致
        by_epoch=True,
        milestones=[16, 19],
        gamma=0.1)
]

# 自动调整学习率的设置
auto_scale_lr = dict(base_batch_size=32)

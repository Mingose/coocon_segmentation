_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_20e.py', 
    '../_base_/default_runtime.py'
]

# 新配置继承了基本配置，并做了必要的修改
# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=7),  # 更新类别数为7
        mask_head=dict(
            num_classes=7,  # 更新类别数为7
            loss_mask=dict(
                type='DiceLoss',  # 使用DiceLoss提升小目标分割的准确性
                loss_weight=1.0  # 设置权重
            )
        )
    )
)

# 数据集相关配置
data_root = 'data/Cocoon_Morbid_coco/'
metainfo = {
    'classes': (
        'Crnshed', 'Imprinted', 'Mold', 
        'Mouthed','Rotting', 'Yellow_Spotted'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), 
                (0, 0, 230), (106, 0, 228), (0, 60, 100)]
}

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='images/train2017/')
    )
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='images/val2017/')
    )
)

test_dataloader = val_dataloader

# 评价指标相关配置
val_evaluator = dict(
    type='CocoMetric', 
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox', 'segm']  # 评估 bbox 和 segm
)
test_evaluator = val_evaluator

# 优化器设置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',  # 使用AdamW优化器
        lr=1e-4,  # 设置学习率
        weight_decay=0.01)  # 设置权重衰减
)

# 学习率调度策略
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,  # 训练的epoch数
        by_epoch=True,
        milestones=[16, 19],  # 在这些epoch调整学习率
        gamma=0.1
    )
]

# 自动缩放学习率设置
auto_scale_lr = dict(base_batch_size=32)

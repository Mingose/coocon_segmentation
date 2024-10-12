_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_20e.py',
    '../_base_/default_runtime.py'
]

# 更改模型的类别数以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=6,  # 匹配数据集的6个类别
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=6,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                ),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=6,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067],
                ),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0))
        ]
    ),
    # 定义 train_cfg 和 test_cfg，确保所有层次正确
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5),
            sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        rcnn=[
            dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5),
                 sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
                 pos_weight=-1,
                 debug=False),
            dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.6, neg_iou_thr=0.6, min_pos_iou=0.6),
                 sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
                 pos_weight=-1,
                 debug=False),
            dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.7, min_pos_iou=0.7),
                 sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
                 pos_weight=-1,
                 debug=False)
        ]
    ),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_num=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
    )
)

# 数据集相关配置
data_root = 'data/Cocoon_Morbid_coco/'
metainfo = {
    'classes': ('Crnshed', 'Imprinted', 'Mold', 
                'Mouthed', 'Rotting', 'Yellow_Spotted'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), 
                (0, 0, 230), (106, 0, 228), (0, 60, 100)]
}

train_dataloader = dict(
    batch_size=8,  # 使用的 batch size
    dataset=dict(
        type='CocoDataset',  # 指定使用 COCO 数据集
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',  # 注释文件路径
        data_prefix=dict(img='images/train2017/'),  # 图像文件路径
        metainfo=metainfo,  # 自定义类别和配色方案
        pipeline=[  # 训练数据的预处理管道
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),  # 修改此处的 img_scale 为 scale
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=1,  # 验证时可以使用较小的 batch size
    dataset=dict(
        type='CocoDataset',  # 使用 COCO 数据集
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',  # 验证集注释文件路径
        data_prefix=dict(img='images/val2017/'),  # 验证集图像文件路径
        metainfo=metainfo,  # 自定义类别和配色方案
        pipeline=[  # 验证数据的预处理管道
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),  # 修改此处的 img_scale 为 scale
            dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
        ]
    )
)

# 评价指标相关配置
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox', 'segm'],  # 评估 bbox 和 segm
    classwise=True  # 按类别输出评估结果
)

test_evaluator = val_evaluator  # 测试时使用同样的评估配置

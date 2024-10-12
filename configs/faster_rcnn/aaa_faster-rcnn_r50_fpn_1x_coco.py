_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# 配置 TensorBoard 日志钩子
# default_hooks = dict(
#     logger=dict(type='LoggerHook', interval=10),
#     tensorboard=dict(type='TensorboardLoggerHook')
# )
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        #bbox_head=dict(num_classes=12), mask_head=dict(num_classes=12)))
        bbox_head=dict(num_classes=1), ))

# 修改数据集相关配置
data_root = './data/Cocoon_coco/'
metainfo = {
    'classes':('Cocoon',),   
    'palette':[(220, 20, 60),]
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='images/train2017/')))

val_dataloader = dict(
    dataset=dict(   
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='images/val2017/')))
test_dataloader = val_dataloader

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator
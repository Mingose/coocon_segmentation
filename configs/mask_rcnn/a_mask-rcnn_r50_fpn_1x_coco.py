_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_20e.py', '../_base_/default_runtime.py'
]

# 新配置继承了基本配置，并做了必要的修改
# _base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=7), mask_head=dict(num_classes=7)))

# 修改数据集相关配置
data_root = 'data/Cocoon_Morbid_coco/'
metainfo = {
    'classes':
        ('Crnshed', 'Imprinted', 'Mold', 
         'Mouthed','Rotting', 'Yellow_Spotted'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), 
         (0, 0, 230), (106, 0, 228),(0, 60, 100)]
}
train_dataloader = dict(
    batch_size=8,
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

# training schedule for 3x
max_epochs = 100
train_cfg = dict(max_epochs=max_epochs)

test_dataloader = val_dataloader

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator
import os
import json
from PIL import Image

def validate_and_fix_coco_annotations(data_root, ann_file, img_prefix, output_ann_file=None):
    # 检查标注文件路径
    ann_path = os.path.join(data_root, ann_file)
    if not os.path.exists(ann_path):
        print(f"标注文件不存在: {ann_path}")
        return
    
    # 加载 COCO 格式的标注文件
    with open(ann_path, 'r') as f:
        coco_data = json.load(f)
    
    # 检查图像文件路径
    img_root = os.path.join(data_root, img_prefix)
    if not os.path.exists(img_root):
        print(f"图像文件夹不存在: {img_root}")
        return
    
    fixed_annotations = []
    invalid_bboxes = 0
    
    # 遍历所有注释，验证 bbox 是否有效，并过滤掉无效的 bbox
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        
        # 验证每个对象的图像是否存在
        img_info = next((img for img in coco_data['images'] if img['id'] == img_id), None)
        if img_info is None:
            print(f"图像 ID {img_id} 没有对应的图像文件")
            continue
        
        img_path = os.path.join(img_root, img_info['file_name'])
        if not os.path.exists(img_path):
            print(f"图像文件不存在: {img_path}")
            continue
        
        # 验证 bbox 是否有效
        bbox = ann.get('bbox', [])
        if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
            fixed_annotations.append(ann)  # 有效的 bbox
        else:
            print(f"无效的 bbox: {bbox} 在图像 {img_info['file_name']}")
            invalid_bboxes += 1
        
        # 检查图像是否可打开
        try:
            img = Image.open(img_path)
            img.verify()
        except (IOError, SyntaxError) as e:
            print(f"图像文件损坏或不可打开: {img_path}，错误: {e}")
    
    # 更新标注数据
    coco_data['annotations'] = fixed_annotations
    
    # 如果有无效 bbox 被修复，保存新的标注文件
    if output_ann_file:
        output_path = os.path.join(data_root, output_ann_file)
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=4)
        print(f"修正后的标注文件已保存至: {output_path}")
    
    print(f"无效 bbox 数量: {invalid_bboxes}")
    print("标注文件和图像检查完成")


# 数据集根目录
data_root = 'data/Cocoon_Morbid_coco/'
print("训练")
# 训练集验证和修复
validate_and_fix_coco_annotations(
    data_root, 
    'annotations/instances_train2017.json', 
    'images/train2017/', 
    output_ann_file='annotations/instances_train2017_fixed.json'
)
print("验证")
# 验证集验证和修复
validate_and_fix_coco_annotations(
    data_root, 
    'annotations/instances_val2017.json', 
    'images/val2017/', 
    output_ann_file='annotations/instances_val2017_fixed.json'
)


# # 数据集根目录
# data_root = 'data/Cocoon_coco/'

# # 训练集验证
# print("训练")
# validate_coco_annotations(data_root, 'annotations/instances_train2017.json', 'images/train2017/')
# print("验证")
# # 验证集验证
# validate_coco_annotations(data_root, 'annotations/instances_val2017.json', 'images/val2017/')
# 数据集根目录


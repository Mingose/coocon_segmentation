import os
import json

def check_category_ids_and_num_classes(ann_file):
    # 检查标注文件路径
    if not os.path.exists(ann_file):
        print(f"标注文件不存在: {ann_file}")
        return

    # 加载 COCO 格式的标注文件
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    categories = coco_data.get('categories', [])
    
    # 检查是否存在类别信息
    if not categories:
        print("标注文件中没有类别信息")
        return

    # 打印所有类别的 ID 和名称
    print("检测到的类别信息：")
    category_ids = []
    for category in categories:
        print(f"Category ID: {category['id']}, Name: {category['name']}")
        category_ids.append(category['id'])

    # 检查 category_id 是否从 1 开始连续
    sorted_category_ids = sorted(category_ids)
    if sorted_category_ids[0] != 1 or sorted_category_ids[-1] != len(sorted_category_ids):
        print("类别 ID 不连续或不从 1 开始，请修复类别 ID")
    else:
        print("类别 ID 从 1 开始并且连续")

    # 计算 num_classes
    num_classes = len(category_ids) + 1  # 包括背景类
    print(f"建议的 num_classes 值（包括背景类）：{num_classes}")

# 使用示例
ann_file = 'data/Cocoon_Morbid_coco/annotations/instances_train2017_fixed.json'  # 替换为你的标注文件路径
check_category_ids_and_num_classes(ann_file)

import os
import json

def replace_category_id(ann_file, output_ann_file, incorrect_id, correct_id):
    # 检查标注文件路径
    if not os.path.exists(ann_file):
        print(f"标注文件不存在: {ann_file}")
        return

    # 加载 COCO 格式的标注文件
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    # 替换 categories 中的错误 ID
    for category in coco_data['categories']:
        if category['id'] == incorrect_id:
            print(f"将 Category ID {incorrect_id} 替换为 {correct_id}")
            category['id'] = correct_id

    # 替换 annotations 中对应的错误 category_id
    for annotation in coco_data['annotations']:
        if annotation['category_id'] == incorrect_id:
            print(f"在注释中将 Category ID {incorrect_id} 替换为 {correct_id}")
            annotation['category_id'] = correct_id

    # 保存修正后的标注文件
    with open(output_ann_file, 'w') as f:
        json.dump(coco_data, f, indent=4)
    print(f"修正后的标注文件已保存至: {output_ann_file}")

ann_file = 'data/Cocoon_Morbid_coco/annotations/instances_train2017_fixed.json'  # 原始标注文件路径
output_ann_file = 'data/Cocoon_Morbid_coco/annotations/instances_train2017.json'  # 修正后的标注文件路径

ann_file2 = 'data/Cocoon_Morbid_coco/annotations/instances_val2017_fixed.json'  # 原始标注文件路径
output_ann_file2 = 'data/Cocoon_Morbid_coco/annotations/instances_val2017.json'  # 修正后的标注文件路径


incorrect_id = 6  # 错误的 Category ID
correct_id = 5  # 正确的 Category ID

print("train_fix...")
replace_category_id(ann_file, output_ann_file, incorrect_id, correct_id)
print("val_fix...")
replace_category_id(ann_file2, output_ann_file2, incorrect_id, correct_id)
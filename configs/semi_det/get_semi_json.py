import json

def remove_annotations(coco_json_file, output_json_file):
    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)
    
    # 删除annotations字段
    if 'annotations' in coco_data:
        del coco_data['annotations']
    
    # 构造无标注JSON文件
    unlabeled_data = {
        "images": coco_data['images'],
        "categories": coco_data['categories'],
        
    }

    # 写入无标注JSON文件
    with open(output_json_file, 'w') as f:
        json.dump(unlabeled_data, f, indent=4)

# 输入文件路径
coco_json_file = "/data0/xiaoyiming/FLIR_v2_M3FD/annotations/flir_v2_train_class3.json"
output_json_file = "/data0/xiaoyiming/FLIR_v2_M3FD/annotations/flir_v2_train_unlabeled.json"

# 删除标注并生成无标注JSON文件
remove_annotations(coco_json_file, output_json_file)

print("无标注JSON文件已生成:", output_json_file)

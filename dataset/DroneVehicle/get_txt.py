import os

# 定义要遍历的文件夹路径
folder_path = 'DroneVehicle/val/data_ir'

# 获取文件夹内所有.jpg文件的路径
jpg_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.jpg')]

# 将文件路径保存为VOC数据集的imagesets格式，写入train.txt文件
with open('DroneVehicle/val/val.txt', 'w') as f:
    for jpg_file in jpg_files:
        # 在VOC数据集的imagesets格式中，每行包含一个文件的路径，不包含文件扩展名
        file_name = os.path.splitext(os.path.basename(jpg_file))[0]
        f.write(file_name + '\n')

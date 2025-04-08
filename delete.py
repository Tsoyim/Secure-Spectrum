

import os

file_name_list = []

for i in [6, 10]:
    for a in range(i):
        file_name = f"./experiment_result/MADQN_no/model/ep_3000_agent_{a}_V2I_{i}_V2V_{i}_Eve_1.pt"
        file_name_list.append(file_name)


for file_name in file_name_list:
    # 检查文件是否存在且为普通文件
    if os.path.isfile(file_name):
        try:
            os.remove(file_name)  # 删除文件
            print(f"成功删除文件 {file_name}。")
        except OSError as e:
            print(f"删除文件时出错：{e}")
    else:
        print(f"文件 {file_name} 不存在或不是普通文件。")
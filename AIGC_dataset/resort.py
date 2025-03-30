def organize_scores(reference_path, target_path, output_path):
    # 读取target文件并创建字典
    target_dict = {}
    with open(target_path, 'r', encoding='utf-8') as file:
        for line in file:
            name, score = line.strip().split(',')
            target_dict[name] = score

    # 打开output文件准备写入
    with open(output_path, 'w', encoding='utf-8') as output_file:
        # 逐行读取reference文件
        with open(reference_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 获取视频名称和对应的分数
                name, _ = line.strip().split(',')
                score = target_dict.get(name, 'Score Not Found')  # 如果找不到分数，输出'Score Not Found'
                # 写入新的一行到output文件
                output_file.write(f"{name},{score}\n")

# 调用函数
organize_scores('all_data_TQ.txt', 'all_data_TW.txt', 'all_data_TW.txt')

print("The scores have been organized according to the reference file order.")
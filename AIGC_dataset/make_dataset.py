import pandas as pd

# 读取 all_data_mos.txt 文件
mos_file_path = "all_data_mos.txt"
mos_data = pd.read_csv(mos_file_path, header=None, names=["file_name", "mos_score"])

# 读取 prompt_data.xlsx 文件
prompt_file_path = "prompt_data.xlsx"
prompt_data = pd.read_excel(prompt_file_path, header=None, names=["label", "description"])

# 清理 description 中的异常回车符
prompt_data["description"] = prompt_data["description"].str.replace(r"\r|\n", " ", regex=True)

# 从 file_name 中提取标签
mos_data["label"] = mos_data["file_name"].apply(lambda x: x.split("+")[1].split(".")[0])

# 将 mos_data 和 prompt_data 按 label 合并
merged_data = pd.merge(mos_data, prompt_data, on="label", how="left")

# 生成新的文件内容，每行格式：file_name｜description｜mos_score
output_file_path = "aigc_merged_data.txt"
with open(output_file_path, "w", encoding="utf-8") as f:
    for _, row in merged_data.iterrows():
        line = f"{row['file_name']}|{row['description']}|{row['mos_score']}\n"
        f.write(line)

print(f"合并后的数据已保存至 {output_file_path}")
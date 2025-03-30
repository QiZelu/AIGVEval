import json

def extract_fields(input_path, output_path):
    with open(input_path, 'r', encoding="utf-8") as f_in:
        data = json.load(f_in)

    with open(output_path, 'w', encoding='utf-8') as f_out:  # 关键修复点
        # f_out.write("video|prompt|gt_score\n")
        for item in data:
            video = item['video']  
            prompt = item['prompt']
            line = f"{video}|{prompt}\n"
            f_out.write(line)
# 使用示例
extract_fields("test.json", "test.txt")
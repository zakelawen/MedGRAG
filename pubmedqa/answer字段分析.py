import json

def check_answer_types(input_file):
    answer_counts = {}

    with open(input_file, 'r') as infile:
        for line in infile:
            data = json.loads(line.strip())
            answer = data.get("answer", "").lower()  # 转为小写以保证统一匹配

            # 统计不同 answer 类型的数量
            if answer in answer_counts:
                answer_counts[answer] += 1
            else:
                answer_counts[answer] = 1

    # 输出各类型的数量
    for answer_type, count in answer_counts.items():
        print(f"Answer type: '{answer_type}' -> Count: {count}")

# 输入文件名
input_file = '/home/user1/zjs/Pubmedqa/pubmedqa.jsonl'  # 请修改为实际文件名

# 调用函数进行查询
check_answer_types(input_file)

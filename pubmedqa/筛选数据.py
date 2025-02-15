import json


# 定义函数来处理文件
def filter_data(input_file, output_file, start_line=301, no_limit=150, maybe_limit=150):
    no_count = 0
    maybe_count = 0
    filtered_data = []

    with open(input_file, 'r') as infile:
        # 跳过前300行
        for line_num, line in enumerate(infile, 1):
            if line_num < start_line:
                continue

            data = json.loads(line.strip())
            answer = data.get("answer", "").lower()

            if answer == "no" and no_count < no_limit:
                filtered_data.append(data)
                no_count += 1
            elif answer == "maybe" and maybe_count < maybe_limit:
                filtered_data.append(data)
                maybe_count += 1

            # 如果已经找到足够的"no"和"maybe"类型的条目，停止读取
            if no_count == no_limit and maybe_count == maybe_limit:
                break

    # 将结果保存到新文件
    with open(output_file, 'w') as outfile:
        for entry in filtered_data:
            outfile.write(json.dumps(entry) + '\n')


# 输入输出文件名
input_file = '/home/user1/zjs/Pubmedqa/pubmedqa.jsonl'  # 请修改为实际文件名
output_file = '2.jsonl'

# 执行筛选函数
filter_data(input_file, output_file)

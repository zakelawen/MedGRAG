import json
import os
import csv
import re

# 定义路径
jsonl_file = '/home/user1/zjs/实验/最终测试数据进行RAG实验/medqa.jsonl'  # 替换为您的 JSONL 文件路径
output_tsv = 'output.tsv'  # 输出 TSV 文件名

# 文件路径模板
base_dir1 = '/home/user1/zjs/实验/实验最终代码/medqa/filtered_results'
base_dir2 = '/home/user1/zjs/实验/实验最终代码/medqa/filtered_results'
base_dir3 = '/home/user1/zjs/实验/实验最终代码/medqa/filtered_results'
base_dir4 = '/home/user1/zjs/实验/实验最终代码/medqa/gpt_generated_results'

# 定义一个函数来清理文本
def clean_text(text):
    if not isinstance(text, str):
        return ''
    # 定义所有需要移除的引号字符，包括标准引号和弯引号
    quote_pattern = r'[\"\'“”‘’]'
    # 使用正则表达式移除引号
    text = re.sub(quote_pattern, '', text)
    # 替换换行符和回车符为空格
    text = text.replace('\n', ' ').replace('\r', ' ')
    # 替换制表符为空格
    text = text.replace('\t', ' ')
    # 移除多余的空格
    text = ' '.join(text.split())
    return text

# 读取 JSONL 文件
data = []
with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            print(f"警告: 第 {line_num} 行为空，跳过。")
            continue
        try:
            json_obj = json.loads(line)
            data.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"错误: 无法解析第 {line_num} 行。错误信息: {e}")
            continue

# 读取文件内容的函数
def read_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()  # 读取文件内容并去掉首尾空白
    else:
        print(f"警告: 文件 {file_path} 不存在。")
        return ''  # 文件不存在时返回空字符串

# 准备写入 TSV 文件
with open(output_tsv, 'w', encoding='utf-8', newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
    # 写入表头，不包括 'graph' 字段
    writer.writerow(['question', 'answer', 'options', 'combined_info'])

    for idx, entry in enumerate(data, start=1):
        # 构建各个文件路径
        folder1 = os.path.join(base_dir1, f"{idx}/question/summarized_definitions.txt")
        folder2 = os.path.join(base_dir2, f"{idx}/options/summarized_definitions.txt")
        folder3 = os.path.join(base_dir3, f"{idx}/question/filtered_paths.txt")
        folder4 = os.path.join(base_dir3, f"{idx}/options/filtered_paths.txt")
        folder5 = os.path.join(base_dir4, f"{idx}/{idx}_gpt_generated_knowledge.txt")

        # 读取文件内容
        summarized_definitions_question = read_file(folder1)
        summarized_definitions_options = read_file(folder2)
        filtered_paths_question = read_file(folder3)
        filtered_paths_options = read_file(folder4)
        gpt_generated_knowledge = read_file(folder5)

        # 将四个文件内容合并成一个字段
        combined_info = ' '.join([
            summarized_definitions_question,
            summarized_definitions_options,
            filtered_paths_question,
            filtered_paths_options,
            gpt_generated_knowledge
        ])

        # 构建 options 字符串，使用分号分隔选项，且使用点作为分隔符
        options = entry.get('options', {})
        sorted_options = sorted(options.items(), key=lambda x: x[0])  # 按选项编号排序
        processed_options_list = [f"{key}. {clean_text(value)}" for key, value in sorted_options]
        options_str = '; '.join(processed_options_list)

        # 构建 answer 字符串
        answer_idx = entry.get('answer_idx')
        answer = clean_text(entry.get('answer', ''))
        if answer_idx and answer_idx in options:
            answer_text = options[answer_idx]
            answer = f"{answer_idx}. {clean_text(answer_text)}"

        # 处理 question 字段中的换行符和引号
        question = clean_text(entry.get('question', ''))

        # 写入一行数据，不包括 'graph' 字段
        writer.writerow([question, answer, options_str, combined_info])

print(f"TSV 文件已生成：{output_tsv}")

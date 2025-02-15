# import json
# import os
# import csv
# import re  # 导入正则表达式模块
#
# # 定义路径
# jsonl_file = '/home/user1/zjs/实验/实验/MedQAdata/1.jsonl'  # 替换为您的 JSONL 文件路径
# graph_dir = '/home/user1/zjs/实验/final/本地embedding/gpt_generated_results/'  # 图结构文件所在目录
# output_tsv = 'output.tsv'  # 输出 TSV 文件名
#
# # 读取 JSONL 文件
# data = []
# with open(jsonl_file, 'r', encoding='utf-8') as f:
#     for line in f:
#         # 如果 JSONL 中每行就是一个完整的 JSON 对象，这样读取一般是没问题的
#         data.append(json.loads(line.strip()))
#
# # 准备写入 TSV
# with open(output_tsv, 'w', encoding='utf-8', newline='') as tsvfile:
#     writer = csv.writer(tsvfile, delimiter='\t')
#     # 写入表头，确保字段顺序为 question, answer, options, graph
#     writer.writerow(['question', 'answer', 'options', 'graph'])
#
#     for idx, entry in enumerate(data, start=1):
#         # 构建 graph 文件路径
#         graph_file = os.path.join(graph_dir, f"{idx}/{idx}_graph_triples.txt")
#         if os.path.exists(graph_file):
#             with open(graph_file, 'r', encoding='utf-8') as gf:
#                 # 先读取并去掉前后空白
#                 raw_graph_content = gf.read().strip()
#                 # 去掉换行符，避免写入TSV出现换行
#                 raw_graph_content = raw_graph_content.replace('\n', ' ').replace('\r', ' ')
#                 # 使用正则表达式去除括号之间多余空白
#                 graph_content = re.sub(r'\)\s*\(', ')(', raw_graph_content)
#         else:
#             graph_content = ''
#             print(f"警告: 图文件 {graph_file} 不存在。")
#
#         # 构建 options 字符串，使用分号分隔选项，且使用点作为分隔符
#         # 按选项编号排序以确保顺序一致
#         sorted_options = sorted(entry['options'].items(), key=lambda x: x[0])
#         # 在拼接 options 前先处理选项中的换行符，避免写入 TSC 时出现问题
#         processed_options_list = []
#         for key, value in sorted_options:
#             # 去掉选项中的换行符
#             clean_value = value.replace('\n', ' ').replace('\r', ' ')
#             processed_options_list.append(f"{key}. {clean_value}")
#         options_str = '; '.join(processed_options_list)
#
#         # 构建 answer 字符串
#         answer_idx = entry.get('answer_idx')
#         if answer_idx and answer_idx in entry['options']:
#             answer_text = entry['options'][answer_idx]
#             # 同样去掉换行符
#             answer_text = answer_text.replace('\n', ' ').replace('\r', ' ')
#             answer = f"{answer_idx}. {answer_text}"
#         else:
#             # 如果没有找到对应的 answer_idx 或选项，使用原始的 answer 字段
#             # 同样做换行符处理
#             answer = entry.get('answer', '').replace('\n', ' ').replace('\r', ' ')
#             print(f"警告: 无法找到 answer_idx '{answer_idx}' 对应的选项。使用原始答案文本。")
#
#         # 处理 question 字段中的换行符
#         question = entry.get('question', '').replace('\n', ' ').replace('\r', ' ')
#
#         # 写入一行数据，字段顺序保持一致：question, answer, options, graph
#         writer.writerow([question, answer, options_str, graph_content])
#
# print(f"TSV 文件已生成：{output_tsv}")



import json
import os
import csv
import re

# 定义路径
jsonl_file = '/home/user1/zjs/实验/实验/MedQAdata/1.jsonl'  # 替换为您的 JSONL 文件路径
graph_dir = '/home/user1/zjs/实验/final/本地embedding/gpt_generated_results/'  # 图结构文件所在目录
output_tsv = 'output.tsv'  # 输出 TSV 文件名

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

# 准备写入 TSV
with open(output_tsv, 'w', encoding='utf-8', newline='') as tsvfile:
    # 设置 csv.writer，不使用引号包裹字段，避免选项中出现引号
    writer = csv.writer(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
    # 写入表头，确保字段顺序为 question, answer, options, graph
    writer.writerow(['question', 'answer', 'options', 'graph'])

    for idx, entry in enumerate(data, start=1):
        # 构建 graph 文件路径
        graph_file = os.path.join(graph_dir, f"{idx}/{idx}_graph_triples.txt")
        if os.path.exists(graph_file):
            try:
                with open(graph_file, 'r', encoding='utf-8') as gf:
                    # 先读取并去掉前后空白
                    raw_graph_content = gf.read().strip()
                    # 去掉换行符，避免写入TSV出现换行
                    raw_graph_content = raw_graph_content.replace('\n', ' ').replace('\r', ' ')
                    # 使用正则表达式去除括号之间多余空白
                    graph_content = re.sub(r'\)\s*\(', ')(', raw_graph_content)
                    # 移除所有引号
                    graph_content = clean_text(graph_content)
            except Exception as e:
                graph_content = ''
                print(f"错误: 读取图文件 {graph_file} 时出错。错误信息: {e}")
        else:
            graph_content = ''
            print(f"警告: 图文件 {graph_file} 不存在。")

        # 构建 options 字符串，使用分号分隔选项，且使用点作为分隔符
        # 按选项编号排序以确保顺序一致
        options = entry.get('options', {})
        if not isinstance(options, dict):
            print(f"警告: 第 {idx} 条数据的 'options' 字段不是字典。使用空选项。")
            sorted_options = []
        else:
            sorted_options = sorted(options.items(), key=lambda x: x[0])

        # 在拼接 options 前先处理选项中的换行符和引号，避免写入 TSV 时出现问题
        processed_options_list = []
        for key, value in sorted_options:
            clean_value = clean_text(value)
            processed_options_list.append(f"{key}. {clean_value}")
        options_str = '; '.join(processed_options_list)

        # 构建 answer 字符串
        answer_idx = entry.get('answer_idx')
        if answer_idx and answer_idx in options:
            answer_text = options[answer_idx]
            answer_text = clean_text(answer_text)
            answer = f"{answer_idx}. {answer_text}"
        else:
            # 如果没有找到对应的 answer_idx 或选项，使用原始的 answer 字段
            # 同样做换行符和引号处理
            answer = clean_text(entry.get('answer', ''))
            if not answer:
                print(f"警告: 第 {idx} 条数据没有 'answer' 字段或为空。")
            else:
                print(f"警告: 无法找到 answer_idx '{answer_idx}' 对应的选项。使用原始答案文本。")

        # 处理 question 字段中的换行符和引号
        question = clean_text(entry.get('question', ''))
        if not question:
            print(f"警告: 第 {idx} 条数据没有 'question' 字段或为空。")

        # 写入一行数据，字段顺序保持一致：question, answer, options, graph
        try:
            writer.writerow([question, answer, options_str, graph_content])
        except csv.Error as e:
            print(f"错误: 在写入第 {idx} 行时出现问题。错误信息: {e}")
            print(f"问题数据: {question}, {answer}, {options_str}, {graph_content}")
        except Exception as e:
            print(f"错误: 在写入第 {idx} 行时遇到未知错误。错误信息: {e}")
            print(f"问题数据: {question}, {answer}, {options_str}, {graph_content}")

print(f"TSV 文件已生成：{output_tsv}")

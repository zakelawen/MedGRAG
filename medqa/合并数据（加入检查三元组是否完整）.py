# import json
# import os
# import csv
# import re  # 导入正则表达式模块
#
# # 定义路径
# jsonl_file = '/home/user2/shiyan/实验/实验/MedQAdata/1.jsonl'  # 替换为您的 JSONL 文件路径
# graph_dir = '/home/user2/shiyan/gpt_generated_results/'  # 图结构文件所在目录
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
#     # 记录有问题的行号及详情
#     problematic_rows = []
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
#         # 检查 graph_content 中的每个三元组是否完整
#         # 假设每个三元组被括号包围，例如：(s1; p1; o1)(s2; p2; o2)
#         triples = re.findall(r'\((.*?)\)', graph_content)
#         for triple in triples:
#             semicolon_count = triple.count(';')
#             if semicolon_count != 2:
#                 error_message = f"错误: 行 {idx} 中的三元组 '{triple}' 不完整，包含 {semicolon_count} 个分号。"
#                 print(error_message)
#                 problematic_rows.append((idx, triple))
#                 # 如果需要记录所有问题三元组，可以继续检查
#                 # 否则，使用 `break` 跳出当前行的检查
#                 # break  # 如果只需记录每行至少一个问题，不需要继续检查
#         # 如果希望只记录每行一次错误，可以在发现第一个错误后跳出
#         # else:
#             # pass  # 三元组完整，无需操作
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
#     # 检查是否有问题的行
#     if problematic_rows:
#         print("\n以下行包含不完整的三元组：")
#         for row_num, triple in problematic_rows:
#             print(f"行号: {row_num}, 三元组: '{triple}'")
#     else:
#         print("所有三元组均完整。")
#
# print(f"TSV 文件已生成：{output_tsv}")
import json
import os
import csv
import re  # 导入正则表达式模块

# 定义路径
jsonl_file = '/home/user1/zjs/Medqa/1.jsonl'  # 替换为您的 JSONL 文件路径
graph_dir = '/home/user1/zjs/实验/实验最终代码/medqa/gpt_generated_results_simple/'  # 图结构文件所在目录
output_tsv = 'output.tsv'  # 输出 TSV 文件名

def parse_triples(graph_str):
    """
    从给定字符串中提取三元组子串。
    当遇到最外层的一对括号 () 时，就将里面的内容提取出来加入列表。
    """
    triplets = []
    stack = []
    current = ""

    for char in graph_str:
        if char == '(':
            if not stack:
                current = ""
            stack.append(char)
            if len(stack) > 1:
                current += char
        elif char == ')':
            if stack:
                stack.pop()
                if stack:
                    current += char
                else:
                    triplets.append(current)
        else:
            if stack:
                current += char

    return triplets

# 读取 JSONL 文件
data = []
with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

# 准备写入 TSV
with open(output_tsv, 'w', encoding='utf-8', newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    # 写入表头，确保字段顺序为 question, answer, options, graph
    writer.writerow(['question', 'answer', 'options', 'graph'])

    # 记录有问题的行号及详情
    problematic_rows = []

    for idx, entry in enumerate(data, start=1):
        # 构建 graph 文件路径
        graph_file = os.path.join(graph_dir, f"{idx}/{idx}_graph_triples.txt")
        if os.path.exists(graph_file):
            with open(graph_file, 'r', encoding='utf-8') as gf:
                # 先读取并去掉前后空白
                raw_graph_content = gf.read().strip()
                # 去掉换行符，避免写入TSV出现换行
                raw_graph_content = raw_graph_content.replace('\n', ' ').replace('\r', ' ')
                # 使用正则表达式去除括号之间多余空白 (可留可删)
                graph_content = re.sub(r'\)\s*\(', ')(', raw_graph_content)
        else:
            graph_content = ''
            print(f"警告: 图文件 {graph_file} 不存在。")

        # --------------------- 只改这里 ---------------------
        # 原先是: triples = re.findall(r'\((.*?)\)', graph_content)
        triples = parse_triples(graph_content)
        # ---------------------------------------------------

        # 检查每个三元组中分号个数
        for triple in triples:
            semicolon_count = triple.count(';')
            if semicolon_count != 2:
                error_message = f"错误: 行 {idx} 中的三元组 '{triple}' 不完整，包含 {semicolon_count} 个分号。"
                print(error_message)
                problematic_rows.append((idx, triple))
                # 如需只记录一次，您可在此处 break

        # 构建 options 字符串
        sorted_options = sorted(entry['options'].items(), key=lambda x: x[0])
        processed_options_list = []
        for key, value in sorted_options:
            clean_value = value.replace('\n', ' ').replace('\r', ' ')
            processed_options_list.append(f"{key}. {clean_value}")
        options_str = '; '.join(processed_options_list)

        # 构建 answer 字符串
        answer_idx = entry.get('answer_idx')
        if answer_idx and answer_idx in entry['options']:
            answer_text = entry['options'][answer_idx]
            answer_text = answer_text.replace('\n', ' ').replace('\r', ' ')
            answer = f"{answer_idx}. {answer_text}"
        else:
            answer = entry.get('answer', '').replace('\n', ' ').replace('\r', ' ')
            print(f"警告: 无法找到 answer_idx '{answer_idx}' 对应的选项。使用原始答案文本。")

        # 处理 question 字段中的换行符
        question = entry.get('question', '').replace('\n', ' ').replace('\r', ' ')

        # 写入一行数据，字段顺序保持一致：question, answer, options, graph
        writer.writerow([question, answer, options_str, graph_content])

    # 检查是否有问题的行
    if problematic_rows:
        print("\n以下行包含不完整的三元组：")
        for row_num, triple in problematic_rows:
            print(f"行号: {row_num}, 三元组: '{triple}'")
    else:
        print("所有三元组均完整。")

print(f"TSV 文件已生成：{output_tsv}")

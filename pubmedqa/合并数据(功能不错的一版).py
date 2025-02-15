import json
import os
import csv
import re  # 导入正则表达式模块

# 定义路径
jsonl_file = '/home/user1/zjs/Pubmedqa/1.jsonl'  # 替换为您的 JSONL 文件路径
graph_dir = '/home/user1/zjs/实验/实验最终代码/pubmedqa/gpt_generated_results'  # 图结构文件所在目录
output_tsv = 'output2.tsv'  # 输出 TSV 文件名

# 读取 JSONL 文件
data = []
with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        # 如果 JSONL 中每行就是一个完整的 JSON 对象，这样读取一般是没问题的
        data.append(json.loads(line.strip()))

# 准备写入 TSV
with open(output_tsv, 'w', encoding='utf-8', newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    # 写入表头，确保字段顺序为 question, answer, graph
    writer.writerow(['question', 'answer', 'graph'])

    # 记录有问题的行号
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
                # 使用正则表达式去除括号之间多余空白
                graph_content = re.sub(r'\)\s*\(', ')(', raw_graph_content)
        else:
            graph_content = ''
            print(f"警告: 图文件 {graph_file} 不存在。")

        # 检查 graph_content 中的每个三元组是否完整
        # 假设每个三元组被括号包围，例如：(s1; p1; o1)(s2; p2; o2)
        triples = re.findall(r'\((.*?)\)', graph_content)
        for triple in triples:
            semicolon_count = triple.count(';')
            if semicolon_count != 2:
                print(f"错误: 行 {idx} 中的三元组 '{triple}' 不完整，包含 {semicolon_count} 个分号。")
                problematic_rows.append(idx)
                break  # 可以选择是否继续检查该行中的其他三元组

        # 构建 answer 字符串
        answer = entry.get('answer', '').replace('\n', ' ').replace('\r', ' ')

        # 处理 question 字段中的换行符
        question = entry.get('question', '').replace('\n', ' ').replace('\r', ' ')

        # 写入一行数据，字段顺序保持一致：question, answer, graph
        writer.writerow([question, answer, graph_content])

    if problematic_rows:
        print("\n以下行包含不完整的三元组：")
        for row in problematic_rows:
            print(f"行号: {row}")
    else:
        print("所有三元组均完整。")

print(f"TSV 文件已生成：{output_tsv}")

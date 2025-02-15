# filter_medqa.py

import os
import csv
import logging
import json
from typing import List, Optional
from tqdm import tqdm
import httpx
from openai import OpenAI
from dotenv import load_dotenv
from neo4j import GraphDatabase
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志，仅显示INFO及以上级别，并记录到文件v
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("filter_medqa.log"),
        logging.StreamHandler()
    ]
)

# 加载环境变量
load_dotenv()
OPENAI_API_KEY = "sk-kydgPim7qKMOHeX77f8fBd194c364a429106Ff96323b39E2"
UMLS_API_KEY = "0823ae3b-da1a-4dad-a683-b9d1ef353414"
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

if not OPENAI_API_KEY:
    logging.error("未找到 OPENAI_API_KEY，请在 .env 文件中设置。")
    exit(1)
if not UMLS_API_KEY:
    logging.error("未找到 UMLS_API_KEY，请在 .env 文件中设置。")
    exit(1)

# 初始化Neo4j客户端
class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def find_next_hop_paths(self, node_name: str, k: int = 1) -> List[str]:
        """
        查找指定节点的下一跳路径，最多返回k条路径。
        """
        query = """
        MATCH p = (n)-[*..1]-(m)
        WHERE n.node_name = $node_name
        RETURN nodes(p) AS Nodes, relationships(p) AS Relationships
        LIMIT $k
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters={"node_name": node_name, "k": k})
                paths = []
                for record in result:
                    nodes = record["Nodes"]
                    relationships = record["Relationships"]
                    # 构建路径字符串
                    path_str = ""
                    for i in range(len(relationships)):
                        source = nodes[i]["node_name"]
                        rel = relationships[i].type
                        target = nodes[i + 1]["node_name"]
                        path_str += f"{source} -[{rel}]-> {target} -> "
                    # 移除最后的箭头
                    path_str = path_str.rstrip(" -> ")
                    paths.append(path_str)
            return paths
        except Exception as e:
            logging.error(f"执行Cypher查询时出错: {e}")
            return []

# 初始化OpenAI客户端
class OpenAIClientWrapper:
    """
    封装与OpenAI API交互的客户端。
    """
    def __init__(self, api_key: str, base_url: str = "https://hk.xty.app/v1"):
        """
        初始化OpenAI客户端。
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(base_url=base_url, follow_redirects=True)
        )

    def filter_definitions(self, context_text: str, definitions: List[List[str]]) -> List[List[str]]:
        """
        使用LLM筛选有用的UMLS定义，去除无用和冗余的定义。
        context_text: 问题和选项的合并文本。
        definitions: List of [CUI, 名称, 来源, 定义]
        """
        if not definitions:
            return []

        definitions_text = "\n".join([f"CUI: {defi[0]}; 名称: {defi[1]}; 来源: {defi[2]}; 定义: {defi[3]}" for defi in definitions])
        prompt = f"""
Task:
Based on the following context, remove any irrelevant or redundant UMLS definitions
Context:
{context_text}

UMLS Definitions:
{definitions_text}

Filtered UMLS Definitions:
"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specializing in medical knowledge filtering."},
                    {"role": "user", "content": prompt},
                ],
                model="gpt-4-turbo-2024-04-09",
                temperature=0.0,
            )
            filtered_definitions_text = response.choices[0].message.content.strip()
            # 解析筛选后的定义
            filtered_definitions = []
            for line in filtered_definitions_text.split('\n'):
                parts = line.split('; ')
                if len(parts) >= 4:
                    cui = parts[0].replace("CUI: ", "").strip()
                    name = parts[1].replace("名称: ", "").strip()
                    source = parts[2].replace("来源: ", "").strip()
                    definition = parts[3].replace("定义: ", "").strip()
                    filtered_definitions.append([cui, name, source, definition])
            return filtered_definitions
        except Exception as e:
            logging.error(f"筛选UMLS定义时出错: {e}")
            return []

    def summarize_definitions(self, filtered_definitions: List[List[str]]) -> Optional[str]:
        """
        使用LLM总结筛选后的UMLS定义。
        """
        if not filtered_definitions:
            return None

        definitions_text = "\n".join([f"{defi[1]}: {defi[3]}" for defi in filtered_definitions])
        prompt = f"""
Task:
Please summarize the following UMLS definitions. Get a comprehensive and non-redundant definition. 
Filtered UMLS Definitions:
{definitions_text}

Summary:
"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specializing in summarizing medical definitions."},
                    {"role": "user", "content": prompt},
                ],
                model="gpt-4-turbo-2024-04-09",
                temperature=0.0,
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            logging.error(f"总结UMLS定义时出错: {e}")
            return None

    def filter_paths(self, context_text: str, paths: List[str]) -> List[str]:
        """
        使用LLM筛选有用的路径，去除无用的路径。如果没有有用路径，返回空列表。
        context_text: 问题和选项的合并文本。
        paths: List of path strings.
        """
        if not paths:
            return []

        paths_text = "\n".join(paths)
        prompt = f"""
Task:
Based on the following context, please filter out the Neo4j paths that are helpful in solving the problem. Remove any irrelevant paths. If no paths are helpful, respond with "None".
Context:
{context_text}

Neo4j Paths:
{paths_text}

Filtered Neo4j Paths:
"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specializing in medical knowledge path filtering."},
                    {"role": "user", "content": prompt},
                ],
                model="gpt-4-turbo-2024-04-09",
                temperature=0.0,
            )
            filtered_paths_text = response.choices[0].message.content.strip()
            if filtered_paths_text.lower() == "none":
                return []
            # 解析筛选后的路径
            filtered_paths = [line.strip() for line in filtered_paths_text.split('\n') if line.strip()]
            return filtered_paths
        except Exception as e:
            logging.error(f"筛选路径时出错: {e}")
            return []

    def should_extend_path(self, path: str, context_text: str) -> bool:
        """
        使用LLM判断是否需要扩展路径。
        如果需要扩展，返回True；否则，返回False。
        """
        prompt = f"""
Task:
Based on the following context and path, please determine whether this path needs to be extended to better solve the problem. If it needs to be extended, respond with "Yes". Otherwise, respond with "No".

Context:
{context_text}

Path:
{path}

Response:
"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specializing in medical path analysis."},
                    {"role": "user", "content": prompt},
                ],
                model="gpt-4-turbo-2024-04-09",
                temperature=0.0,
            )
            answer = response.choices[0].message.content.strip().lower()
            return answer == "yes"
        except Exception as e:
            logging.error(f"判断是否需要扩展路径时出错: {e}")
            return False

# 读取UMLS定义文件
def read_definitions(file_path: str) -> List[List[str]]:
    definitions = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                definitions.append([row['CUI'], row['名称'], row['来源'], row['定义']])
        logging.info(f"读取UMLS定义文件 {file_path}，共 {len(definitions)} 条。")
    except FileNotFoundError:
        logging.error(f"定义文件 {file_path} 不存在。")
    except Exception as e:
        logging.error(f"读取定义文件 {file_path} 时出错: {e}")
    return definitions

# 读取路径文件
def read_paths(file_path: str) -> List[str]:
    paths = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            paths = [line.strip() for line in file if line.strip()]
        logging.info(f"读取路径文件 {file_path}，共 {len(paths)} 条。")
    except FileNotFoundError:
        logging.error(f"路径文件 {file_path} 不存在。")
    except Exception as e:
        logging.error(f"读取路径文件 {file_path} 时出错: {e}")
    return paths

# 保存筛选后的UMLS定义
def save_filtered_definitions(definitions: List[List[str]], file_path: str):
    """
    保存筛选后的UMLS定义到CSV文件。
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["CUI", "名称", "来源", "定义"])
            for defi in definitions:
                writer.writerow(defi)
        logging.info(f"筛选后的UMLS定义已保存到 {file_path}")
    except Exception as e:
        logging.error(f"保存UMLS定义到 {file_path} 时出错: {e}")

# 保存总结后的UMLS定义
def save_summarized_definitions(summary: str, file_path: str):
    """
    保存总结后的UMLS定义到TXT文件。
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode='w', encoding='utf-8') as file:
            file.write(summary)
        logging.info(f"总结后的UMLS定义已保存到 {file_path}")
    except Exception as e:
        logging.error(f"保存总结后的UMLS定义到 {file_path} 时出错: {e}")

# 保存筛选后的路径
def save_filtered_paths(paths: List[str], file_path: str):
    """
    保存筛选后的路径到TXT文件。
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode='w', encoding='utf-8') as file:
            for path in paths:
                file.write(f"{path}\n")
        logging.info(f"筛选后的路径已保存到 {file_path}")
    except Exception as e:
        logging.error(f"保存路径到 {file_path} 时出错: {e}")

# 扩展路径（最多三跳）
def extend_paths(
    paths: List[str],
    context_text: str,
    llm_client: OpenAIClientWrapper,
    neo4j_client: Neo4jClient,
    k: int = 1,
    max_hops: int = 3,
    max_workers: int = 500  # 添加最大线程数参数
) -> List[str]:
    """
    使用LLM判断是否需要扩展路径，并进行扩展，最多扩展到max_hops跳。
    引入并行处理以提升性能。
    """
    current_hops = 1
    all_paths = paths.copy()

    while current_hops < max_hops:
        new_paths = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有路径扩展任务
            future_to_path = {executor.submit(process_single_path, path, context_text, llm_client, neo4j_client, k): path for path in all_paths}
            for future in as_completed(future_to_path):
                try:
                    path_result = future.result()
                    if path_result:
                        new_paths.extend(path_result)
                except Exception as e:
                    logging.error(f"处理路径时出错: {e}")

        # 筛选新路径
        filtered_new_paths = llm_client.filter_paths(context_text, new_paths)
        if not filtered_new_paths:
            break
        all_paths = filtered_new_paths
        current_hops += 1
    return all_paths

def process_single_path(path: str, context_text: str, llm_client: OpenAIClientWrapper, neo4j_client: Neo4jClient, k: int) -> List[str]:
    """
    处理单个路径，判断是否需要扩展并执行扩展操作。
    返回扩展后的路径列表。
    """
    extended_paths = []
    # 判断是否需要扩展
    need_extension = llm_client.should_extend_path(path, context_text)
    if need_extension:
        # 获取路径的最后一个节点
        last_node = path.split("->")[-1].strip()
        # 使用Cypher查询下一跳路径
        next_hop_paths = neo4j_client.find_next_hop_paths(last_node, k)
        if next_hop_paths:
            for next_path in next_hop_paths:
                # 拼接当前路径与下一跳路径，形成新路径
                try:
                    rel_type = next_path.split('-[')[1].split(']->')[0]
                    target_node = next_path.split(']-> ')[1]
                    extended_path = f"{path} -[{rel_type}]-> {target_node}"
                    extended_paths.append(extended_path)
                except IndexError:
                    logging.error(f"解析路径时出错: {next_path}")
    else:
        extended_paths.append(path)
    return extended_paths

# 处理单个上下文（问题或选项）
def process_context(
    context_text: str,
    definitions_file: str,
    paths_file: str,
    output_dir: str,
    llm_client: OpenAIClientWrapper,
    neo4j_client: Neo4jClient,
    k: int = 1,
    max_hops: int = 3
):
    """
    处理单个上下文（问题或选项），筛选UMLS定义和路径，并进行路径扩展及总结。
    """
    # 读取数据
    definitions = read_definitions(definitions_file)
    paths = read_paths(paths_file)

    # 筛选UMLS定义（LLM1）
    filtered_definitions = llm_client.filter_definitions(context_text, definitions)

    # 总结筛选后的UMLS定义（LLM4）
    summarized_definitions = llm_client.summarize_definitions(filtered_definitions)

    # 筛选路径（LLM2）
    filtered_paths = llm_client.filter_paths(context_text, paths)

    if filtered_paths:
        # 扩展路径（最多三跳，使用LLM3）
        extended_paths = extend_paths(filtered_paths, context_text, llm_client, neo4j_client, k=k, max_hops=max_hops)
        # 筛选扩展后的路径（LLM2）
        filtered_paths = llm_client.filter_paths(context_text, extended_paths)
    else:
        filtered_paths = []

    # 保存结果
    filtered_definitions_file = os.path.join(output_dir, "filtered_definitions.csv")
    summarized_definitions_file = os.path.join(output_dir, "summarized_definitions.txt")
    filtered_paths_file = os.path.join(output_dir, "filtered_paths.txt")

    save_filtered_definitions(filtered_definitions, filtered_definitions_file)

    if summarized_definitions:
        save_summarized_definitions(summarized_definitions, summarized_definitions_file)

    if filtered_paths:
        save_filtered_paths(filtered_paths, filtered_paths_file)
    else:
        logging.info(f"{os.path.basename(output_dir)}部分未筛选到有用的路径。")

# 处理单个MedQA数据行
def process_filtered_medqa(
    line_number: int,
    question_text: str,
    options: dict,
    input_dir: str,
    output_dir: str,
    llm_client: OpenAIClientWrapper,
    neo4j_client: Neo4jClient,
    k: int = 1,
    max_hops: int = 3
):
    """
    处理单个MedQA数据行，筛选UMLS定义和路径，并进行路径扩展及总结。
    同时处理问题和选项部分。
    """
    # 处理问题部分
    question_definitions_file = os.path.join(input_dir, "question", "definitions.csv")
    question_paths_file = os.path.join(input_dir, "question", "paths.txt")
    question_output_dir = os.path.join(output_dir, "question")

    process_context(
        context_text=question_text,
        definitions_file=question_definitions_file,
        paths_file=question_paths_file,
        output_dir=question_output_dir,
        llm_client=llm_client,
        neo4j_client=neo4j_client,
        k=k,
        max_hops=max_hops
    )

    # 处理选项部分
    if options:
        # 将options字典转换为列表，并提取所有选项的文本
        options_text = " ".join([text for key, text in options.items()])
        options_definitions_file = os.path.join(input_dir, "options", "definitions.csv")
        options_paths_file = os.path.join(input_dir, "options", "paths.txt")
        options_output_dir = os.path.join(output_dir, "options")

        process_context(
            context_text=options_text,
            definitions_file=options_definitions_file,
            paths_file=options_paths_file,
            output_dir=options_output_dir,
            llm_client=llm_client,
            neo4j_client=neo4j_client,
            k=k,
            max_hops=max_hops
        )
    else:
        logging.info(f"行 {line_number} 没有找到 'options' 字段。")

# 主函数
def main():
    """
    主处理函数，处理整个JSONL文件中的所有数据行。
    """
    # 配置路径
    input_jsonl = "/home/user1/zjs/实验/最终测试数据进行RAG实验/medmcqa.jsonl"  # 请替换为实际路径
    output_root_dir = "filtered_results"

    # 初始化客户端
    llm_client = OpenAIClientWrapper(api_key=OPENAI_API_KEY)
    neo4j_client = Neo4jClient(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)

    # 处理JSONL文件
    try:
        with open(input_jsonl, 'r', encoding='utf-8') as infile:
            for current_line_num, line in enumerate(tqdm(infile, desc="处理行"), start=1):
                try:

                    output_dir = os.path.join(output_root_dir, str(current_line_num))

                    if os.path.exists(output_dir):
                        logging.info(f"行 {current_line_num} 已处理，跳过。")
                        continue

                    data = json.loads(line)
                    question_text = data.get("question", "")
                    options = data.get("options", {})
                    if not question_text:
                        logging.warning(f"行 {current_line_num} 没有找到 'question' 字段。")
                        continue

                    # 定义输入和输出目录
                    input_dir = os.path.join("result", str(current_line_num))
                    output_dir = os.path.join(output_root_dir, str(current_line_num))

                    process_filtered_medqa(
                        line_number=current_line_num,
                        question_text=question_text,
                        options=options,
                        input_dir=input_dir,
                        output_dir=output_dir,
                        llm_client=llm_client,
                        neo4j_client=neo4j_client,
                        k=1,
                        max_hops=3
                    )

                except json.JSONDecodeError:
                    logging.error(f"行 {current_line_num} JSON 解码错误。")
                except Exception as e:
                    logging.error(f"行 {current_line_num} 发生意外错误: {e}")
    except FileNotFoundError:
        logging.error(f"输入文件 {input_jsonl} 不存在。")
    except Exception as e:
        logging.error(f"处理JSONL文件时出错: {e}")
    finally:
        # 关闭Neo4j客户端
        neo4j_client.close()
        logging.info("所有资源已关闭，处理完成。")

if __name__ == "__main__":
    main()

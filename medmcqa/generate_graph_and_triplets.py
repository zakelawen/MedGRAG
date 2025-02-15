# generate_knowledge_and_graph.py

import os
import json
import logging
from typing import List, Dict
from tqdm import tqdm
import httpx
from dotenv import load_dotenv
from openai import OpenAI

# 导入filter_medqa.py中的函数

# 配置日志，仅显示INFO及以上级别，并记录到文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("generate_knowledge_and_graph.log"),
        logging.StreamHandler()
    ]
)

# 加载环境变量
load_dotenv()
OPENAI_API_KEY = "sk-kydgPim7qKMOHeX77f8fBd194c364a429106Ff96323b39E2" # 从 .env 文件中读取 API 密钥
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://hk.xty.app/v1")  # 默认值可根据需要调整

if not OPENAI_API_KEY:
    logging.error("未找到 OPENAI_API_KEY，请在 .env 文件中设置。")
    exit(1)


# 初始化OpenAI客户端
class OpenAIClientWrapper:
    """
    封装与OpenAI API交互的客户端。
    """

    def __init__(self, api_key: str, base_url: str = "https://api.xty.app/v1"):
        """
        初始化OpenAI客户端。
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(base_url=base_url, follow_redirects=True)
        )

    def generate_medical_knowledge(self, question: str, options: Dict[str, str], answer: str) -> str:
        """
        使用LLM生成相关的医学知识。
        """
        prompt = f"""
Task:
Please provide relevant medical knowledge that helps to understand or solve the problem based on the following medical questions, options. Only output knowledge, the content should be useful, concise and correct.

Question:
{question}

Options:
"""
        for key, option in options.items():
            prompt += f"{key}: {option}\n"
        prompt += f"\nRelevant medical knowledge:"

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-2024-04-09",
                messages=[
                    {"role": "system",
                     "content": "You are an assistant with extensive knowledge in the field of medical science."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # 调整温度以获得更一致的输出
            )
            medical_knowledge = response.choices[0].message.content.strip()
            return medical_knowledge
        except Exception as e:
            logging.error(f"生成医学知识时出错: {e}")
            return ""

    def generate_graph_triples(self, question: str, options: Dict[str, str], knowledge: str) -> str:
        """
        使用LLM根据问题、选项和相关知识生成图谱（三元组）。
        返回一个字符串，格式为 (subject; predicate; object)(subject; predicate; object)...
        """
        prompt = f"""
Task:
Based on the following medical question, options and relevant medical knowledge, please create a knowledge graph to solve this problem, expressed in the form of triples (a; b; c),each triple has two semicolons,each triple has three parts.Such as:(a; b; c)(d; e; f)(h; i; j)...
Requirements:
1. The knowledge graph should contain the key information of the problem, and ensure that it does not contain common sense knowledge, such as knowledge about night and day.
2. The knowledge graph should be able to use the relevant medical knowledge provided to point out the reasons why the options are correct and wrong, but it cannot directly say that the options are right or wrong, and ensure that it can be expressed clearly without redundancy.
3. The output is in the form of triples, and there is no need to separate the triplets,for example (a; b; c)(d; e; f)
4. Knowledge that is not very helpful to the problem can be directly discarded
5. The triples must be complete! ! !
6. You must ensure that the output is a triple in the correct format!!! (a; b; c)(d; e; f)(h; i; j)
Question:
{question}

Options:
"""
        for key, option in options.items():
            prompt += f"{key}: {option}\n"
        prompt += f"\nRelevant medical knowledge:\n{knowledge}\n\nKnowledge graph triplets:"

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-2024-04-09",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who is good at creating knowledge graphs from medical information."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                temperature=0.0,  # 调整温度以获得更一致的输出
            )
            triples_text = response.choices[0].message.content.strip()
            return triples_text
        except Exception as e:
            logging.error(f"生成图谱三元组时出错: {e}")
            return ""


# 保存GPT生成的医学知识
def save_medical_knowledge(knowledge: str, file_path: str):
    """
    保存GPT生成的医学知识到TXT文件。
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode='w', encoding='utf-8') as file:
            file.write(knowledge)
        logging.info(f"GPT生成的医学知识已保存到 {file_path}")
    except Exception as e:
        logging.error(f"保存医学知识到 {file_path} 时出错: {e}")


# 保存知识图谱（三元组）文本
def save_graph_triples_text(triples_text: str, file_path: str):
    """
    保存知识图谱三元组到TXT文件。
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode='w', encoding='utf-8') as file:
            file.write(triples_text)
        logging.info(f"知识图谱三元组文本已保存到 {file_path}")
    except Exception as e:
        logging.error(f"保存知识图谱三元组到 {file_path} 时出错: {e}")


# 处理单个MedQA数据行
def process_medqa_line(
        line_number: int,
        question: str,
        options: Dict[str, str],
        answer: str,
        input_dir: str,
        output_dir: str,
        llm_client: OpenAIClientWrapper
):
    """
    处理单个MedQA数据行，生成并保存医学知识和知识图谱。
    """
    logging.info(f"开始处理行 {line_number}：问题='{question}'，选项={options}，答案='{answer}'")

    # 定义输出文件路径
    knowledge_file_path = os.path.join(output_dir, f"{line_number}_gpt_generated_knowledge.txt")
    graph_file_path_txt = os.path.join(output_dir, f"{line_number}_graph_triples.txt")

    # 定义筛选后的定义和路径文件路径
    # 从 'question' 子目录读取
    question_summarized_definitions_file = os.path.join(input_dir, "question", "summarized_definitions.txt")
    question_filtered_paths_file = os.path.join(input_dir, "question", "filtered_paths.txt")

    # 从 'options' 子目录读取
    options_summarized_definitions_file = os.path.join(input_dir, "options", "summarized_definitions.txt")
    options_filtered_paths_file = os.path.join(input_dir, "options", "filtered_paths.txt")

    # 检查文件是否存在
    for file_path in [question_summarized_definitions_file, question_filtered_paths_file,
                      options_summarized_definitions_file, options_filtered_paths_file]:
        if not os.path.exists(file_path):
            logging.warning(f"文件不存在：{file_path}")

    # 读取筛选后的定义（summarized_definitions.txt）
    question_summarized_definitions = []
    if os.path.exists(question_summarized_definitions_file):
        try:
            with open(question_summarized_definitions_file, mode='r', encoding='utf-8') as f:
                question_summarized_definitions = [line.strip() for line in f if line.strip()]
            logging.info(f"读取问题部分的总结定义，共 {len(question_summarized_definitions)} 条。")
        except Exception as e:
            logging.error(f"读取文件 {question_summarized_definitions_file} 时出错: {e}")
    else:
        logging.warning(f"文件不存在：{question_summarized_definitions_file}")

    options_summarized_definitions = []
    if os.path.exists(options_summarized_definitions_file):
        try:
            with open(options_summarized_definitions_file, mode='r', encoding='utf-8') as f:
                options_summarized_definitions = [line.strip() for line in f if line.strip()]
            logging.info(f"读取选项部分的总结定义，共 {len(options_summarized_definitions)} 条。")
        except Exception as e:
            logging.error(f"读取文件 {options_summarized_definitions_file} 时出错: {e}")
    else:
        logging.warning(f"文件不存在：{options_summarized_definitions_file}")

    combined_summarized_definitions = question_summarized_definitions + options_summarized_definitions

    # 读取筛选后的路径
    question_filtered_paths = []
    if os.path.exists(question_filtered_paths_file):
        try:
            with open(question_filtered_paths_file, mode='r', encoding='utf-8') as f:
                question_filtered_paths = [line.strip() for line in f if line.strip()]
            logging.info(f"读取问题部分的筛选路径，共 {len(question_filtered_paths)} 条。")
        except Exception as e:
            logging.error(f"读取文件 {question_filtered_paths_file} 时出错: {e}")
    else:
        logging.warning(f"文件不存在：{question_filtered_paths_file}")

    options_filtered_paths = []
    if os.path.exists(options_filtered_paths_file):
        try:
            with open(options_filtered_paths_file, mode='r', encoding='utf-8') as f:
                options_filtered_paths = [line.strip() for line in f if line.strip()]
            logging.info(f"读取选项部分的筛选路径，共 {len(options_filtered_paths)} 条。")
        except Exception as e:
            logging.error(f"读取文件 {options_filtered_paths_file} 时出错: {e}")
    else:
        logging.warning(f"文件不存在：{options_filtered_paths_file}")

    combined_filtered_paths = question_filtered_paths + options_filtered_paths

    # 生成并保存GPT提供的相关医学知识
    medical_knowledge = llm_client.generate_medical_knowledge(question, options,answer)
    save_medical_knowledge(medical_knowledge, knowledge_file_path)

    # 合并所有知识
    all_combined_knowledge = ""
    if combined_summarized_definitions:
        all_combined_knowledge += "\n".join(combined_summarized_definitions) + "\n"
    if combined_filtered_paths:
        all_combined_knowledge += "\n".join(combined_filtered_paths) + "\n"
    if medical_knowledge:
        all_combined_knowledge += medical_knowledge

    logging.info(f"生成知识图谱所用的合并知识长度：{len(all_combined_knowledge)} 字符。")

    # 生成知识图谱文本
    graph_triples_text = llm_client.generate_graph_triples(question, options, all_combined_knowledge)

    # 保存到 .txt，而不是 .json
    save_graph_triples_text(graph_triples_text, graph_file_path_txt)
    logging.info(f"完成处理行 {line_number}。")


# 主函数
def main():
    """
    主处理函数，处理整个JSONL文件中的所有数据行。
    """
    # 配置路径
    input_jsonl = "/home/user1/zjs/实验/最终测试数据进行RAG实验/medmcqa.jsonl"   # 请替换为实际路径
    filtered_definitions_root = "filtered_results"  # 假设之前的筛选结果保存在此目录
    output_root_dir = "gpt_generated_results"

    # 初始化OpenAI客户端
    llm_client = OpenAIClientWrapper(api_key=OPENAI_API_KEY, base_url=BASE_URL)

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
                    question = data.get("question", "")
                    options = data.get("options", {})
                    answer = data.get("answer", "")
                    if not question:
                        logging.warning(f"行 {current_line_num} 没有找到 'question' 字段。")
                        continue

                    # 定义输入和输出目录
                    input_dir = os.path.join(filtered_definitions_root, str(current_line_num))
                    output_dir = os.path.join(output_root_dir, str(current_line_num))

                    # 确保输出目录存在
                    os.makedirs(output_dir, exist_ok=True)

                    # 处理当前行
                    process_medqa_line(
                        line_number=current_line_num,
                        question=question,
                        options=options,
                        answer=answer,
                        input_dir=input_dir,
                        output_dir=output_dir,
                        llm_client=llm_client
                    )

                except json.JSONDecodeError:
                    logging.error(f"行 {current_line_num} JSON 解码错误。")
                except Exception as e:
                    logging.error(f"行 {current_line_num} 发生意外错误: {e}")
    except FileNotFoundError:
        logging.error(f"输入文件 {input_jsonl} 不存在。")
    except Exception as e:
        logging.error(f"处理JSONL文件时出错: {e}")



if __name__ == "__main__":
    main()

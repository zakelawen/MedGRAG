# # openai_client.py

# import numpy as np
# import httpx
# from openai import OpenAI
# import logging
# from typing import Optional, List
# ##############################
# # 并发所需的库
# ##############################
# import concurrent.futures

# class OpenAIClient:
#     """
#     封装与OpenAI API交互的客户端。
#     """

#     def __init__(self, api_key: str, base_url: str = "https://api.xty.app/v1"):
#         """
#         初始化OpenAI客户端。

#         :param api_key: OpenAI API密钥。
#         :param base_url: OpenAI API的基础URL。
#         """
#         self.client = OpenAI(
#             base_url=base_url,
#             api_key=api_key,
#             http_client=httpx.Client(base_url=base_url, follow_redirects=True)
#         )

#     def get_embedding(self, text: str) -> Optional[np.ndarray]:
#         """
#         获取文本的嵌入向量。

#         :param text: 输入文本。
#         :return: 嵌入向量或None（如果出错）。
#         """
#         try:
#             response = self.client.embeddings.create(
#                 model="text-embedding-3-large",
#                 input=text,
#                 encoding_format="float",
#             )
#             embedding = response.data[0].embedding
#             return np.array(embedding, dtype=np.float32)
#         except Exception as e:
#             logging.error(f"获取嵌入向量时出错: {e}")
#             return None

#     ##############################
#     # 新增并发 Embedding 方法
#     ##############################
#     def get_embeddings_concurrent(self, texts: List[str]) -> List[Optional[np.ndarray]]:
#         """
#         并发获取多个文本的 Embedding，减少顺序调用造成的时间损耗。
#         :param texts: 文本列表
#         :return: 与 texts 对应的 embedding 向量列表（可能有 None）
#         """
#         results = [None] * len(texts)

#         # 线程池大小可自行调整，例如 5、10 等
#         with concurrent.futures.ThreadPoolExecutor(max_workers=500) as executor:
#             # 提交每个文本的 get_embedding 任务
#             future_to_index = {}
#             for i, txt in enumerate(texts):
#                 future = executor.submit(self.get_embedding, txt)
#                 future_to_index[future] = i

#             # 收集完成的任务
#             for future in concurrent.futures.as_completed(future_to_index):
#                 idx = future_to_index[future]
#                 try:
#                     emb = future.result()
#                     results[idx] = emb
#                 except Exception as exc:
#                     logging.error(f"并发获取文本索引 {idx} 的 Embedding 时出错: {exc}")
#                     results[idx] = None

#         return results

#     def extract_key_entities(self, text: str, prompt_type: str = "question",
#                                  model: str = "gpt-3.5-turbo-1106") -> Optional[str]:
#             """
#             提取文本中的关键医疗实体，支持不同的提示词类型。

#             :param text: 输入文本。
#             :param prompt_type: 提示词类型，区分'question'或'options'。
#             :param model: 使用的LLM模型。
#             :return: 提取的实体或None（如果出错）。
#             """
#             prompts = {
#                 "question": f"""
# There are some samples:
# ### Instruction:\n'Learn to extract entities from the following medical questions.'\n\n### Input:\n
# Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma\n\n ### Output:
# chronic urethral obstruction 
# benign prismatic hyperplasia 
# kidney prismatic hyperplasia 
# \n\n
# ### Instruction:\n'Learn to extract entities from the following medical questions.'\n\n### Input:\n
# Which vitamin is supplied from only animal source?\n\n ### Output:
# vitamin
# animal source
# \n\n
#  Try to output:
#  ### Instruction:\n'Learn to extract entities from the following medical questions.'\n\n### Input:\n
# {text}\n\n ### Output:
#     """,
#                 "options": f"""
# There are some samples:
# \n\n
# ### Instruction:\n'Learn to extract entities from the following medical question options.'\n\n### Input:\n
# "A": "Hyperplasia", "B": "Hyperophy", "C": "Atrophy", "D": "Dyplasia"\n\n ### Output:
# Hyperplasia
# Hyperophy
# Atrophy
# Dyplasia
# \n\n
# Instruction:\n'Learn to extract entities from the following medical question optinos.'\n\n### Input:\n
# "A": "Vitamin C", "B": "Vitamin B7", "C": "Vitamin B12", "D": "Vitamin D"\n\n ### Output:
# Vitamin C
# Vitamin B7
# Vitamin B12
# Vitamin D
# \n\n
# Instruction:\n'Learn to extract entities from the following medical question options.'\n\n### Input:\n
# "A": "0.7", "B": "0.8", "C": "0.9", "D": "1"\n\n ### Output:
# no medical entities
# \n\n
# Try to output:
# ### Instruction:\n'Learn to extract entities from the following medical question options.'\n\n### Input:\n
# {text}\n\n ### Output:
# """
#             }

#             prompt = prompts.get(prompt_type, prompts["question"])

#             try:
#                 response = self.client.chat.completions.create(
#                     messages=[
#                         {"role": "system", "content": "You are a helpful medical assistant."},
#                         {"role": "user", "content": prompt},
#                     ],
#                     model=model,
#                     temperature=0.0,
#                 )
#                 entities = response.choices[0].message.content.strip()
#                 return entities
#             except Exception as e:
#                 logging.error(f"处理文本时出错: {text}\n错误信息: {e}")
#                 return None

#     def select_relevant_relationships(
#                 self, node_name: str, all_relationships: List[str], context: str, model: str = "gpt-4-turbo-2024-04-09"
#         ) -> List[str]:
#             """
#             根据上下文和所有关系类型，使用LLM筛选出最相关的关系类型。

#             :param node_name: 当前节点的名称。
#             :param all_relationships: 所有可用的关系类型列表。
#             :param context: 当前处理的上下文信息。
#             :param model: 使用的LLM模型。
#             :return: 筛选后的最多三条关系类型列表。
#             """
#             prompt = f"""
#     You are an expert in medical knowledge graphs. Given the current node "{node_name}" and its context:

#     {context}

#     Here is a list of all possible relationship types in the graph:

#     {', '.join(f'"{rel}"' for rel in all_relationships)}

#     Please select the top three most relevant relationship types that should be used to explore connections from this node. Provide only the relationship types, separated by commas.

#     Example Output:
#     "protein_protein", "drug_effect", "pathway_protein"

#     Your Selection:
#     """

#             try:
#                 response = self.client.chat.completions.create(
#                     messages=[
#                         {"role": "system", "content": "You are a helpful assistant."},
#                         {"role": "user", "content": prompt},
#                     ],
#                     model=model,
#                     temperature=0.0,
#                 )
#                 selection = response.choices[0].message.content.strip()
#                 # 解析输出，提取关系类型
#                 selected_relationships = [rel.strip().strip('"') for rel in selection.split(",")[:3]]
#                 return selected_relationships
#             except Exception as e:
#                 logging.error(f"筛选关系类型时出错: {e}")
#                 return []
# openai_client.py

import numpy as np
import httpx
from openai import OpenAI
import logging
from typing import Optional, List
##############################
# 并发所需的库
##############################
import concurrent.futures

class OpenAIClient:
    """
    封装与OpenAI API交互的客户端。
    """

    def __init__(self, api_key: str, base_url: str = "https://hk.xty.app/v1"):
        """
        初始化OpenAI客户端。

        :param api_key: OpenAI API密钥。
        :param base_url: OpenAI API的基础URL。
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(base_url=base_url, follow_redirects=True)
        )

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        获取文本的嵌入向量。

        :param text: 输入文本。
        :return: 嵌入向量或None（如果出错）。
        """
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=text,
                encoding_format="float",
            )
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logging.error(f"获取嵌入向量时出错: {e}")
            return None

    ##############################
    # 新增并发 Embedding 方法
    ##############################
    def get_embeddings_concurrent(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        并发获取多个文本的 Embedding，减少顺序调用造成的时间损耗。
        :param texts: 文本列表
        :return: 与 texts 对应的 embedding 向量列表（可能有 None）
        """
        results = [None] * len(texts)

        # 线程池大小可自行调整，例如 5、10 等
        with concurrent.futures.ThreadPoolExecutor(max_workers=500) as executor:
            # 提交每个文本的 get_embedding 任务
            future_to_index = {}
            for i, txt in enumerate(texts):
                future = executor.submit(self.get_embedding, txt)
                future_to_index[future] = i

            # 收集完成的任务
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    emb = future.result()
                    results[idx] = emb
                except Exception as exc:
                    logging.error(f"并发获取文本索引 {idx} 的 Embedding 时出错: {exc}")
                    results[idx] = None

        return results

    def extract_key_entities(self, text: str, prompt_type: str = "question",
                                 model: str = "gpt-4-turbo-2024-04-09") -> Optional[str]:
            """
            提取文本中的关键医疗实体，支持不同的提示词类型。

            :param text: 输入文本。
            :param prompt_type: 提示词类型，区分'question'或'options'。
            :param model: 使用的LLM模型。
            :return: 提取的实体或None（如果出错）。
            """
            prompts = {
                "question": f"""
Task:
Please extract all medical entities from the following medical question. 

Example Input:
"Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma"

Expected Output:
chronic urethral obstruction 
benign prismatic hyperplasia 
kidney prismatic 

    The text is: 
    "{text}"
    """,
                "options": f"""
Task:
Please extract all clinically significant medical entities from the following medical options.

Example Input1:
"A": "Hyperplasia", "B": "Hyperophy", "C": "Atrophy", "D": "Dyplasia"
Expected Output1:
Hyperplasia
Hyperophy
Atrophy
Dyplasia

Example Input2:
"A": "Vitamin C", "B": "Vitamin B7", "C": "Vitamin B12", "D": "Vitamin D"
Expected Output2:
Vitamin C
Vitamin B7
Vitamin B12
Vitamin D

Example Input3:
"A": "0.7", "B": "0.8", "C": "0.9", "D": "1"
Expected Output3:
no medical entities
    The text is: 
    "{text}"
    """
            }

            prompt = prompts.get(prompt_type, prompts["question"])

            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful medical assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    model=model,
                    temperature=0.0,
                )
                entities = response.choices[0].message.content.strip()
                return entities
            except Exception as e:
                logging.error(f"处理文本时出错: {text}\n错误信息: {e}")
                return None

    def select_relevant_relationships(
                self, node_name: str, all_relationships: List[str], context: str, model: str = "gpt-4-turbo-2024-04-09"
        ) -> List[str]:
            """
            根据上下文和所有关系类型，使用LLM筛选出最相关的关系类型。

            :param node_name: 当前节点的名称。
            :param all_relationships: 所有可用的关系类型列表。
            :param context: 当前处理的上下文信息。
            :param model: 使用的LLM模型。
            :return: 筛选后的最多三条关系类型列表。
            """
            prompt = f"""
    You are an expert in medical knowledge graphs. Given the current node "{node_name}" and its context:

    {context}

    Here is a list of all possible relationship types in the graph:

    {', '.join(f'"{rel}"' for rel in all_relationships)}

    Please select the top three most relevant relationship types that should be used to explore connections from this node. Provide only the relationship types, separated by commas.

    Example Output:
    "protein_protein", "drug_effect", "pathway_protein"

    Your Selection:
    """

            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    model=model,
                    temperature=0.0,
                )
                selection = response.choices[0].message.content.strip()
                # 解析输出，提取关系类型
                selected_relationships = [rel.strip().strip('"') for rel in selection.split(",")[:3]]
                return selected_relationships
            except Exception as e:
                logging.error(f"筛选关系类型时出错: {e}")
                return []

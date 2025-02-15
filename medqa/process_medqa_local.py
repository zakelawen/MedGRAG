# process_medqa.py

import os
import json
import logging
from tqdm import tqdm
import numpy as np
import requests
from dotenv import load_dotenv
import faiss
import csv
from typing import List, Tuple, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import torch
import gc

# 本地客户端 (替换OpenAI)
from local_embedding_client import LocalEmbeddingClient
from openai_client import OpenAIClient
# Neo4j部分
from neo4j_client import Neo4jClient

#########################################
#  FAISS 相关函数
#########################################

def load_faiss_index(index_path: str) -> faiss.Index:
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"索引文件 {index_path} 不存在。请确保路径正确并且索引已构建。")

    logging.info(f"从 {index_path} 加载 FAISS 内积索引...")
    index = faiss.read_index(index_path)
    logging.info("FAISS 内积索引加载完成。")
    return index

def search_similar_vectors_faiss_ip(index: faiss.Index, query_vector: np.ndarray, top_k: int = 5) -> Tuple[List[float], List[int]]:
    query_vector = query_vector.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query_vector)
    similarities, indices = index.search(query_vector, top_k)
    return similarities[0].tolist(), indices[0].tolist()

def get_or_create_faiss_index(embeddings: np.ndarray, index_path: str) -> faiss.Index:
    if os.path.exists(index_path):
        try:
            index = load_faiss_index(index_path)
            return index
        except Exception as e:
            logging.error(f"加载现有索引时出错: {e}")
            logging.info("尝试重新构建索引...")

    dimension = embeddings.shape[1]
    logging.info(f"构建内积搜索索引，向量维度: {dimension}")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    logging.info(f"内积索引已保存到 {index_path}")

    return index

#########################################
#  UMLS 相关函数
#########################################

def search_umls_with_api_key(api_key: str, term: str, max_pages: int = 10) -> Tuple[List[dict], int]:
    url = "https://uts-ws.nlm.nih.gov/rest/search/current"
    all_results = []
    page_number = 1

    params = {
        "string": term,
        "apiKey": api_key,
        "inputType": "atom",
        "language": "ENG",
        "pageSize": 50,
        "pageNumber": page_number,
        "searchType": "words",
        "includeObsolete": "false",
        "includeSuppressible": "false",
        "returnIdType": "concept",
        "partialSearch": "false"
    }

    session = requests.Session()
    retries = Retry(total=2, backoff_factor=1, status_forcelist=[500, 502, 503, 504], allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    try:
        with tqdm(total=max_pages, desc=f"搜索UMLS: {term}", unit="页") as pbar:
            while page_number <= max_pages:
                params["pageNumber"] = page_number

                try:
                    response = session.get(url, params=params, timeout=3)
                    response.raise_for_status()
                    data = response.json()
                    results = data.get("result", {}).get("results", [])

                    if not results or results[0].get("ui") == "NONE":
                        break

                    all_results.extend(results)

                    if page_number >= max_pages:
                        break

                    page_number += 1
                    pbar.update(1)

                except requests.exceptions.Timeout:
                    logging.error(f"请求第 {page_number} 页时超时。")
                    break
                except requests.exceptions.RequestException as e:
                    logging.error(f"请求第 {page_number} 页时出错: {e}")
                    break
                except ValueError as e:
                    logging.error(f"解析第 {page_number} 页响应时出错: {e}")
                    break

    except Exception as e:
        logging.error(f"UMLS搜索过程出错: {e}")

    return all_results, page_number

def get_cui_definition(api_key: str, cui: str) -> Optional[dict]:
    url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/definitions"
    params = {"apiKey": api_key}

    try:
        response = requests.get(url, params=params, timeout=3)
        response.raise_for_status()
        results = response.json()
        return results
    except requests.exceptions.Timeout:
        logging.error(f"请求 CUI {cui} 的定义时超时。")
        return None
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred while fetching definitions for CUI {cui}: {http_err}")
        return None
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request exception while fetching definitions for CUI {cui}: {req_err}")
        return None
    except ValueError as json_err:
        logging.error(f"JSON decoding failed while fetching definitions for CUI {cui}: {json_err}")
        return None

def umls_search_and_definitions_optimized(api_key: str, term: str, max_cuis: int = 1) -> List[dict]:
    search_results, _ = search_umls_with_api_key(api_key, term, max_pages=1)
    detailed_results = []

    if search_results:
        for result in search_results[:max_cuis]:
            cui = result.get("ui")
            name = result.get("name")
            root_source = result.get("rootSource")
            uri = result.get("uri")

            definitions = []
            if cui:
                definition_results = get_cui_definition(api_key, cui)
                if definition_results and "result" in definition_results:
                    for definition in definition_results["result"]:
                        source = definition.get("rootSource", "未知来源")
                        value = definition.get("value", "无定义内容")
                        definitions.append({"source": source, "definition": value})

            if definitions:
                detailed_results.append({
                    "cui": cui,
                    "name": name,
                    "root_source": root_source,
                    "uri": uri,
                    "definitions": definitions
                })

    return detailed_results

def is_empty_result(umls_results: List[dict], cypher_paths: List[str]) -> bool:
    return (not umls_results) or (not cypher_paths)

def save_umls_results_to_file(umls_table: List[List[str]], filename: str):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["CUI", "名称", "来源", "定义"])

            merged_results = {}
            for def_item in umls_table:
                cui, name, source, value = def_item
                key = (cui, name)
                if key in merged_results:
                    merged_results[key][3] += " ; " + value
                else:
                    merged_results[key] = [cui, name, source, value]

            for key, value in merged_results.items():
                writer.writerow(value)
        logging.info(f"UMLS 查询结果已保存到 {filename}")
    except Exception as e:
        logging.error(f"保存 UMLS 查询结果时出错: {e}")

def save_cypher_paths_to_file(cypher_table: List[List[str]], filename: str):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode='w', encoding='utf-8') as file:
            for row in cypher_table:
                file.write(f"{row[0]}\n")
        logging.info(f"Neo4j 查询路径结果已保存到 {filename}")
    except Exception as e:
        logging.error(f"保存 Neo4j 查询路径结果时出错: {e}")

#########################################
#  计算余弦相似度
#########################################

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1 is None or vec2 is None:
        return 0.0
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))



def process_entities(
    entity: str,
    current_line_num: int,
    label: str,
    openai_client: OpenAIClient,
    embedding_client: LocalEmbeddingClient,
    umls_api_key: str,
    neo4j_client: Neo4jClient,
    faiss_index: faiss.Index,
    node_indices: np.ndarray,
    k: int,
    relevant_relationships: Optional[List[str]] = None,
    context_text: str = ""
) -> Tuple[List[List[str]], List[List[str]]]:
    umls_results = []
    cypher_results = []

    logging.info(f"行 {current_line_num} {label} 处理实体: {entity}")

    # 提取实体（使用OpenAIClient）
    extracted_entities = openai_client.extract_key_entities(entity, prompt_type='question')
    if extracted_entities:
        entities = [e.strip() for e in extracted_entities.split('\n') if e.strip()]
    else:
        entities = [entity]  # 如果提取失败，使用原始实体

    # UMLS 查询
    entity_definitions = umls_search_and_definitions_optimized(umls_api_key, entity, max_cuis=1)
    if entity_definitions:
        for def_item in entity_definitions:
            cui = def_item['cui']
            name = def_item['name']
            root_source = def_item['root_source']
            uri = def_item['uri']

            definitions = []
            for def_detail in def_item['definitions']:
                source = def_detail['source']
                value = def_detail['definition']
                definitions.append([cui, name, source, value])

            umls_results.extend(definitions)
    else:
        logging.info(f"行 {current_line_num} {label} 实体 '{entity}' 无UMLS定义结果。")

    # Cypher 查询
    try:
        paths, cypher_exec_time = neo4j_client.find_k_hop_paths_by_name(entity, k, relationship_types=relevant_relationships)
        if len(paths) > 10:
            logging.info(f"实体 '{entity}' 查询到路径数量为 {len(paths)} 条，开始进行相似度筛选，仅保留最相似的 10 条。")

            # 获取上下文embedding（使用本地模型）
            context_embedding = embedding_client.get_local_embedding(context_text) if context_text else None

            if context_embedding is None:
                logging.error("上下文嵌入为空，无法进行相似度筛选。")
            else:
                # 分批处理路径，每批200条
                batch_size = 50
                top_k_per_batch = 10
                top_paths_candidates = []

                for i in range(0, len(paths), batch_size):
                    batch_paths = paths[i:i + batch_size]
                    logging.info(f"处理路径批次 {i // batch_size + 1}，包含 {len(batch_paths)} 条路径。")

                    # 获取批次路径的嵌入
                    path_embeddings = embedding_client.get_local_embeddings_concurrent(batch_paths)
                    # 过滤掉获取失败的嵌入
                    valid_indices = [idx for idx, emb in enumerate(path_embeddings) if emb is not None]
                    valid_paths = [batch_paths[idx] for idx in valid_indices]
                    valid_embeddings = [path_embeddings[idx] for idx in valid_indices]

                    # 计算余弦相似度
                    similarities = [cosine_similarity(context_embedding, emb) for emb in valid_embeddings]

                    # 获取当前批次中相似度最高的前10条路径
                    top_indices = np.argsort(similarities)[-top_k_per_batch:][::-1]  # 降序排序
                    for idx in top_indices:
                        top_paths_candidates.append((valid_paths[idx], similarities[idx]))

                # 从所有批次候选中选择相似度最高的前10条路径
                if top_paths_candidates:
                    # 按相似度降序排序
                    top_paths_candidates.sort(key=lambda x: x[1], reverse=True)
                    best_paths = top_paths_candidates[:10]
                    paths = [bp[0] for bp in best_paths]
                    logging.info(f"最终筛选出 {len(paths)} 条最相似的路径。")
                else:
                    paths = []
                    logging.info("未筛选出任何相似路径。")

        if paths:
            cypher_results.append([f"{entity} - 路径 ({cypher_exec_time:.4f} 秒)"])
            for path in paths:
                cypher_results.append([path])
        else:
            logging.info(f"行 {current_line_num} {label} 实体 '{entity}' 无Cypher路径结果。")
    except Exception as e:
        logging.error(f"行 {current_line_num} {label} 实体 '{entity}' Cypher 查询时出错: {e}")

    # 检查UMLS/Cypher
    if is_empty_result(entity_definitions, paths):
        logging.info(f"行 {current_line_num} {label} 实体 '{entity}' 在 UMLS 或 Cypher 查询中未找到信息。进行相似度搜索...")

        # FAISS 回溯
        try:
            entity_embedding = openai_client.get_embedding(entity)  # 使用API获取实体嵌入
        except Exception as e:
            logging.error(f"行 {current_line_num} {label} 实体 '{entity}' 获取嵌入向量时出错: {e}")
            return umls_results, cypher_results

        if entity_embedding is None:
            logging.error(f"行 {current_line_num} {label} 实体 '{entity}' 无法获取嵌入向量。")
            return umls_results, cypher_results

        try:
            similarities, indices = search_similar_vectors_faiss_ip(
                faiss_index,
                entity_embedding,
                top_k=5
            )
        except Exception as e:
            logging.error(f"行 {current_line_num} {label} 实体 '{entity}' FAISS 搜索时出错: {e}")
            return umls_results, cypher_results

        for i in range(len(indices)):
            node_id = node_indices[indices[i]]
            similarity = similarities[i]
            logging.info(f"节点 ID = {node_id}, 相似度 = {similarity:.4f}")

            if similarity < 0.9 and i != 0:
                continue

            node_name = neo4j_client.get_node_names([node_id]).get(node_id, "未找到对应的node_name")

            # UMLS
            try:
                node_definitions = umls_search_and_definitions_optimized(umls_api_key, node_name, max_cuis=1)
                if node_definitions:
                    for def_item in node_definitions:
                        cui = def_item['cui']
                        name = def_item['name']
                        for def_detail in def_item['definitions']:
                            source = def_detail['source']
                            value = def_detail['definition']
                            umls_results.append([cui, name, source, value])
            except Exception as e:
                logging.error(f"节点 '{node_name}' 的 UMLS 查询时出错: {e}")

            # K-hop路径
            try:
                node_paths, node_cypher_time = neo4j_client.find_k_hop_paths_by_index(node_id, k, relationship_types=relevant_relationships)
                if node_paths:
                    if len(node_paths) > 10:
                        logging.info(f"FAISS 回溯查询: 节点 '{node_name}' 路径共 {len(node_paths)} 条，执行相似度筛选仅保留 10 条。")

                        context_embedding = embedding_client.get_local_embedding(context_text) if context_text else None
                        if context_embedding is None:
                            logging.error("上下文嵌入为空，无法进行相似度筛选。")
                            node_paths = []
                        else:
                            # 分批处理路径，每批200条
                            batch_size = 50
                            top_k_per_batch = 10
                            top_paths_candidates_node = []

                            for i in range(0, len(node_paths), batch_size):
                                batch_node_paths = node_paths[i:i + batch_size]
                                logging.info(f"处理节点路径批次 {i // batch_size + 1}，包含 {len(batch_node_paths)} 条路径。")

                                # 获取批次路径的嵌入
                                node_paths_embeddings = embedding_client.get_local_embeddings_concurrent(batch_node_paths)
                                # 过滤掉获取失败的嵌入
                                valid_indices = [idx for idx, emb in enumerate(node_paths_embeddings) if emb is not None]
                                valid_node_paths = [batch_node_paths[idx] for idx in valid_indices]
                                valid_node_embeddings = [node_paths_embeddings[idx] for idx in valid_indices]

                                # 计算余弦相似度
                                similarities_node = [cosine_similarity(context_embedding, emb) for emb in valid_node_embeddings]

                                # 获取当前批次中相似度最高的前10条路径
                                top_indices_node = np.argsort(similarities_node)[-top_k_per_batch:][::-1]  # 降序排序
                                for idx in top_indices_node:
                                    top_paths_candidates_node.append((valid_node_paths[idx], similarities_node[idx]))

                            # 从所有批次候选中选择相似度最高的前10条路径
                            if top_paths_candidates_node:
                                # 按相似度降序排序
                                top_paths_candidates_node.sort(key=lambda x: x[1], reverse=True)
                                best_node_paths = top_paths_candidates_node[:10]
                                node_paths = [bp[0] for bp in best_node_paths]
                                logging.info(f"最终筛选出 {len(node_paths)} 条最相似的节点路径。")
                            else:
                                node_paths = []
                                logging.info("未筛选出任何相似的节点路径。")

                    cypher_results.append([f"节点 '{node_name}' - 路径 ({node_cypher_time:.4f} 秒)"])
                    for path in node_paths:
                        cypher_results.append([path])
                else:
                    logging.info(f"节点 '{node_name}' 未找到路径。")
            except Exception as e:
                logging.error(f"节点 '{node_name}' 的 Cypher 查询时出错: {e}")

            if i == 0:
                break

    return umls_results, cypher_results






#########################################
# 处理主 JSONL
#########################################

def process_medqa_jsonl(
    input_file_path: str,
    openai_client: OpenAIClient,
    embedding_client: LocalEmbeddingClient,
    neo4j_client: Neo4jClient,
    faiss_index: faiss.Index,
    node_indices: np.ndarray,
    embeddings: np.ndarray,
    api_key: str,
    k: int = 1,
    target_lines: Optional[List[int]] = None
):
    target_set = set(target_lines) if target_lines else None
    result_dir = "result"

    # 获取所有关系类型
    all_relationships = neo4j_client.get_all_relationship_types()
    logging.info(f"获取到所有关系类型，共 {len(all_relationships)} 种。")

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for current_line_num, line in enumerate(tqdm(infile, desc="处理行"), start=1):

            line_folder = os.path.join(result_dir, str(current_line_num))
            if os.path.exists(line_folder):
                logging.info(f"行 {current_line_num} 已处理，跳过。")
                continue

            if target_set and current_line_num not in target_set:
                continue
            try:
                data = json.loads(line)

                question_text = data.get("question", "")
                options = data.get("options", {})

                if not question_text and not options:
                    continue

                line_folder = os.path.join(result_dir, str(current_line_num))
                question_folder = os.path.join(line_folder, "question")
                options_folder = os.path.join(line_folder, "options")
                os.makedirs(question_folder, exist_ok=True)
                os.makedirs(options_folder, exist_ok=True)

                # ----- 处理 question -----
                if question_text:
                    # 使用OpenAIClient提取实体
                    question_entities_str = openai_client.extract_key_entities(question_text, prompt_type='question')
                    if question_entities_str:
                        question_entities = [e.strip() for e in question_entities_str.split('\n') if e.strip()]
                    else:
                        question_entities = question_text.split()  # 如果提取失败，使用简单分割

                    umls_table_question = []
                    cypher_table_question = []

                    for entity in question_entities:
                        context = f"问题描述: {question_text}\n选项: {json.dumps(options, ensure_ascii=False)}"
                        umls_res, cypher_res = process_entities(
                            entity=entity,
                            current_line_num=current_line_num,
                            label='question',
                            openai_client=openai_client,
                            embedding_client=embedding_client,
                            umls_api_key=api_key,
                            neo4j_client=neo4j_client,
                            faiss_index=faiss_index,
                            node_indices=node_indices,
                            k=k,
                            relevant_relationships=None,
                            context_text=context
                        )
                        umls_table_question.extend(umls_res)
                        cypher_table_question.extend(cypher_res)

                    # 保存UMLS查询结果
                    if umls_table_question:
                        umls_filename = os.path.join(question_folder, "definitions.csv")
                        save_umls_results_to_file(umls_table_question, umls_filename)

                    # 保存Cypher查询结果
                    if cypher_table_question:
                        cypher_filename = os.path.join(question_folder, "paths.txt")
                        save_cypher_paths_to_file(cypher_table_question, cypher_filename)

                # ----- 处理 options -----
                if options:
                    umls_table_options = []
                    cypher_table_options = []
                    for option_label, option_text in options.items():
                        if not option_text:
                            continue

                        # 使用OpenAIClient提取实体
                        option_entities_str = openai_client.extract_key_entities(option_text, prompt_type='options')
                        if option_entities_str:
                            option_entities = [e.strip() for e in option_entities_str.split('\n') if e.strip()]
                        else:
                            option_entities = option_text.split()  # 如果提取失败，使用简单分割

                        for entity in option_entities:
                            context = f"问题描述: {question_text}\n选项: {json.dumps(options, ensure_ascii=False)}"
                            umls_res, cypher_res = process_entities(
                                entity=entity,
                                current_line_num=current_line_num,
                                label=f"options - {option_label}",
                                openai_client=openai_client,
                                embedding_client=embedding_client,
                                umls_api_key=api_key,
                                neo4j_client=neo4j_client,
                                faiss_index=faiss_index,
                                node_indices=node_indices,
                                k=k,
                                relevant_relationships=None,
                                context_text=context
                            )
                            umls_table_options.extend(umls_res)
                            cypher_table_options.extend(cypher_res)

                    if umls_table_options:
                        umls_filename_options = os.path.join(options_folder, "definitions.csv")
                        save_umls_results_to_file(umls_table_options, umls_filename_options)

                    if cypher_table_options:
                        cypher_filename_options = os.path.join(options_folder, "paths.txt")
                        save_cypher_paths_to_file(cypher_table_options, cypher_filename_options)

            except json.JSONDecodeError:
                logging.error(f"行 {current_line_num} JSON 解码错误。")
            except Exception as e:
                logging.error(f"行 {current_line_num} 发生意外错误: {e}")

            if target_set and current_line_num >= max(target_set):
                break

def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 你的 API_KEY / 路径
    openai_api_key = "sk-sd9PHQbWAIwyVXA00b16CcF2B49a4f80B7A042F14fB5B9B5"  # 请替换为实际的OpenAI API密钥
    umls_api_key = "0823ae3b-da1a-4dad-a683-b9d1ef353414"       # 请替换为实际的UMLS API密钥

    # 路径
    faiss_index_path = "/home/user1/zjs/实验/实验/import/faiss_index_flat_ip.index"
    node_indices_npy = "/home/user1/zjs/实验/实验/import/node_indices.npy"
    embeddings_npy = "/home/user1/zjs/实验/实验/import/embeddings.npy"
    input_jsonl = "/home/user1/zjs/实验/最终测试数据进行RAG实验/medqa.jsonl"

    # 加载 node_indices
    try:
        node_indices = np.load(node_indices_npy).astype(str)
        logging.info(f"加载 node_indices 成功，共 {len(node_indices)} 个节点。")
    except Exception as e:
        logging.error(f"加载 node_indices 时出错: {e}")
        return

    # 加载 embeddings
    try:
        embeddings = np.load(embeddings_npy).astype(np.float32)
        logging.info(f"加载 embeddings 成功，形状: {embeddings.shape}。")
    except Exception as e:
        logging.error(f"加载 embeddings 时出错: {e}")
        return

    # 初始化 FAISS
    try:
        faiss_index = get_or_create_faiss_index(embeddings, faiss_index_path)
    except Exception as e:
        logging.error(f"加载或创建 FAISS 索引时出错: {e}")
        return

    # ---------------------------
    #  关键改动：使用本地客户端和OpenAI客户端
    # ---------------------------
    try:
        # 初始化 OpenAIClient
        openai_client = OpenAIClient(
            api_key=openai_api_key,
            base_url="https://api.xty.app/v1"  # 请根据实际情况调整
        )
        logging.info("初始化 OpenAI 客户端成功。")

        # 简单测试
        test_embedding = openai_client.get_embedding("测试测试")
        if test_embedding is not None:
            logging.info(f"OpenAI embedding 长度: {len(test_embedding)}")
        else:
            logging.error("无法获取 OpenAI embedding。")
            return
    except Exception as e:
        logging.error(f"初始化 OpenAI 客户端时出错: {e}")
        return

    try:
        # 使用上下文管理器初始化 LocalEmbeddingClient
        with LocalEmbeddingClient(
            model_name_or_path="/mnt/zjs/model/",
            device="cuda",
            max_length=32768
        ) as embedding_client:
            logging.info("初始化 NV-Embed-v2 本地客户端并加载模型成功。")

            # 简单测试
            test_local_embedding = embedding_client.get_local_embedding("测试测试")
            if test_local_embedding is not None:
                logging.info(f"本地 embedding 长度: {len(test_local_embedding)}")
            else:
                logging.error("无法获取本地 embedding。")
                # 即使测试失败，也继续执行以便后续处理

            # 初始化 Neo4j
            try:
                neo4j_client = Neo4jClient(uri="neo4j://localhost:7687", user="neo4j", password="12345678")
                logging.info("初始化 Neo4j 客户端成功。")
                with neo4j_client.driver.session() as session:
                    test_result = session.run("RETURN 1 AS number")
                    record = test_result.single()
                    if record and record["number"] == 1:
                        logging.info("成功连接到 Neo4j 数据库。")
                    else:
                        logging.error("无法验证 Neo4j 数据库连接。")
            except Exception as e:
                logging.error(f"初始化 Neo4j 客户端时出错: {e}")
                return

            # 准备执行
            k = 1
            target_line_numbers = None  # 如果需要处理特定行，可以设置为列表，例如 [1, 2, 3]

            try:
                process_medqa_jsonl(
                    input_file_path=input_jsonl,
                    openai_client=openai_client,
                    embedding_client=embedding_client,
                    neo4j_client=neo4j_client,
                    faiss_index=faiss_index,
                    node_indices=node_indices,
                    embeddings=embeddings,
                    api_key=umls_api_key,
                    k=k,
                    target_lines=target_line_numbers
                )
            except Exception as e:
                logging.error(f"处理 JSONL 文件时出错: {e}")
            finally:
                # Neo4j 客户端关闭
                try:
                    neo4j_client.close()
                    logging.info("Neo4j 客户端已关闭。")
                except Exception as e:
                    logging.error(f"关闭 Neo4j 客户端时出错: {e}")

            # 模型将在上下文管理器退出时自动卸载

    except Exception as e:
        logging.error(f"初始化本地Embedding客户端时出错: {e}")
        return

    logging.info("所有资源已关闭，处理完成。")

if __name__ == "__main__":
    main()

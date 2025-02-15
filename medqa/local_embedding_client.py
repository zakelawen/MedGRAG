# # local_embedding_client.py

# import torch
# import torch.nn.functional as F
# import numpy as np
# import logging
# from typing import List, Optional
# from transformers import AutoModel

# class LocalEmbeddingClient:
#     """
#     使用 NV-Embed-v2 模型生成文本embedding的客户端示例，支持上下文管理器。
#     """

#     def __init__(
#         self,
#         model_name_or_path: str = "/mnt/zjs/model/",
#         device: str = "cuda",
#         max_length: int = 32768
#     ):
#         """
#         初始化本地模型
#         :param model_name_or_path: 模型名称或路径
#         :param device: "cuda" 或 "cpu"
#         :param max_length: 最大序列长度, NV-Embed-v2可支持到 32768
#         """
#         self.model_name_or_path = model_name_or_path
#         self.device = device
#         self.max_length = max_length
#         self.model = None  # 初始化时不加载模型

#     def load_model(self):
#         """
#         加载模型到指定设备。
#         """
#         if self.model is None:
#             logging.info(f"加载模型 '{self.model_name_or_path}' 到设备 '{self.device}'...")
#             try:
#                 self.model = AutoModel.from_pretrained(
#                     self.model_name_or_path,
#                     trust_remote_code=True,
#                     low_cpu_mem_usage=True,   # 减少CPU内存使用
#                     device_map="auto",
#                           max_memory={
#                            0: "20.0GB",
#                            1: "30.0GB"
#                           },   # 自动设备映射
#                     torch_dtype=torch.float16
#                 )
#                 logging.info("模型加载完成。")
#             except Exception as e:
#                 logging.error(f"加载模型时出错: {e}")
#                 raise

#     def unload_model(self):
#         """
#         卸载模型并释放显卡资源。
#         """
#         if self.model is not None:
#             logging.info("卸载本地模型并释放显卡资源...")
#             del self.model
#             self.model = None
#             torch.cuda.empty_cache()
#             logging.info("模型已卸载。")

#     def __enter__(self):
#         """
#         上下文管理器入口，加载模型。
#         """
#         self.load_model()
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         """
#         上下文管理器出口，卸载模型。
#         """
#         self.unload_model()

#     def get_local_embedding(self, text: str) -> Optional[np.ndarray]:
#         """
#         获取单条文本的Embedding, 返回 np.ndarray (shape: [dim])
#         """
#         if not text.strip():
#             return None

#         if self.model is None:
#             logging.error("模型未加载，请先调用 load_model 方法或使用上下文管理器。")
#             return None

#         # 假设 `encode` 方法存在，并返回 [batch_size, embedding_dim] 的tensor
#         try:
#             with torch.no_grad():
#                 embeddings = self.model.encode(
#                     [text],
#                     instruction="",      # 可自行指定, 比如 "Instruct: ...\nQuery: "
#                     max_length=self.max_length
#                 )
#                 embeddings = embeddings.to(self.device)
#                 embeddings = F.normalize(embeddings, p=2, dim=1)  # L2归一化

#                 # 取第0个, 转到CPU并转numpy
#                 return embeddings[0].detach().cpu().numpy()
#         except Exception as e:
#             logging.error(f"获取本地嵌入时出错: {e}")
#             return None

#     def get_local_embeddings_concurrent(self, texts: List[str]) -> List[Optional[np.ndarray]]:
#         """
#         批量获取多个文本的Embedding, 返回 List[np.ndarray]
#         """
#         if not texts:
#             return []

#         if self.model is None:
#             logging.error("模型未加载，请先调用 load_model 方法或使用上下文管理器。")
#             return [None] * len(texts)

#         try:
#             with torch.no_grad():
#                 embeddings = self.model.encode(
#                     texts,
#                     instruction="",     # 对批量文本, 如果有需要可以设置同一个instruction
#                     max_length=self.max_length
#                 )
#                 embeddings = embeddings.to(self.device)
#                 embeddings = F.normalize(embeddings, p=2, dim=1)  # 归一化

#                 embeddings_np = embeddings.detach().cpu().numpy()
#                 # 按行拆分成 [向量, 向量, ...]
#                 results = []
#                 for i in range(embeddings_np.shape[0]):
#                     results.append(embeddings_np[i])
#                 return results
#         except Exception as e:
#             logging.error(f"批量获取本地嵌入时出错: {e}")
#             return [None] * len(texts)




# local_embedding_client.py



import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Optional
from transformers import AutoModel
import gc




class LocalEmbeddingClient:
    """
    使用 NV-Embed-v2 模型生成文本embedding的客户端示例，支持上下文管理器。
    """

    def __init__(
        self,
        model_name_or_path: str = "/mnt/zjs/model/",
        device: str = "cuda",  # 保持为 'cuda' 以支持多GPU
        max_length: int = 32768
    ):
        """
        初始化本地模型
        :param model_name_or_path: 模型名称或路径
        :param device: "cuda" 或 "cpu"
        :param max_length: 最大序列长度, NV-Embed-v2可支持到 32768
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_length = max_length
        self.model = None  # 初始化时不加载模型

    def load_model(self):
        """
        加载模型到指定设备。
        """
        if self.model is None:
            logging.info(f"加载模型 '{self.model_name_or_path}' 到设备 '{self.device}'...")
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,   # 减少CPU内存使用
                    device_map="auto", 
    #                        max_memory={
    #                                 0: "24GB",
    #                                 2: "24GB",
    # },       # 自动设备映射，支持多GPU       # 自动设备映射，支持多GPU
                    torch_dtype=torch.float16
                )
                logging.info("模型加载完成。")
            except Exception as e:
                logging.error(f"加载模型时出错: {e}")
                raise

    def unload_model(self):
        """
        卸载模型并释放显卡资源。
        """
        if self.model is not None:
            logging.info("卸载本地模型并释放显卡资源...")
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()
            logging.info("模型已卸载。")

    def __enter__(self):
        """
        上下文管理器入口，加载模型。
        """
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器出口，卸载模型。
        """
        self.unload_model()

    def get_local_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        获取单条文本的Embedding, 返回 np.ndarray (shape: [dim])
        """
        if not text.strip():
            return None

        if self.model is None:
            logging.error("模型未加载，请先调用 load_model 方法或使用上下文管理器。")
            return None

        embeddings = None  # 初始化 embeddings
        try:
            with torch.no_grad():
                # 编码文本
                embeddings = self.model.encode(
                    [text],
                    instruction="",      # 可自行指定, 比如 "Instruct: ...\nQuery: "
                    max_length=self.max_length
                )
                # 归一化
                embeddings = F.normalize(embeddings, p=2, dim=1)
                # 移动到CPU并转为numpy
                embedding_np = embeddings[0].detach().cpu().numpy()
        except Exception as e:
            logging.error(f"获取本地嵌入时出错: {e}")
            return None
        finally:
            # 删除临时变量并清理缓存
            if embeddings is not None:
                del embeddings
            torch.cuda.empty_cache()
            gc.collect()

        return embedding_np

    def get_local_embeddings_concurrent(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        批量获取多个文本的Embedding, 返回 List[np.ndarray]
        """
        if not texts:
            return []

        if self.model is None:
            logging.error("模型未加载，请先调用 load_model 方法或使用上下文管理器。")
            return [None] * len(texts)

        embeddings = None  # 初始化 embeddings
        try:
            with torch.no_grad():
                # 分批处理以避免显存不足
                batch_size = 50  # 根据实际情况调整
                embeddings_list = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        instruction="",     # 对批量文本, 如果有需要可以设置同一个instruction
                        max_length=self.max_length
                    )
                    # 归一化
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                    # 移动到CPU并转为numpy
                    batch_embeddings_np = batch_embeddings.detach().cpu().numpy()
                    embeddings_list.extend(batch_embeddings_np)

            return embeddings_list
        except Exception as e:
            logging.error(f"批量获取本地嵌入时出错: {e}")
            return [None] * len(texts)
        finally:
            # 删除临时变量并清理缓存
            if embeddings is not None:
                del embeddings
            torch.cuda.empty_cache()
            gc.collect()

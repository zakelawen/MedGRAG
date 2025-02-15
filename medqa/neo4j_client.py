# neo4j_client.py

from neo4j import GraphDatabase
import time
import logging
from typing import List, Tuple, Optional

class Neo4jClient:
    """
    封装与Neo4j数据库交互的客户端。
    """

    def __init__(self, uri: str, user: str, password: str):
        """
        初始化Neo4j客户端。

        :param uri: Neo4j数据库URI。
        :param user: 用户名。
        :param password: 密码。
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """
        关闭Neo4j驱动。
        """
        self.driver.close()

    def get_node_names(self, node_indices: List[str]) -> dict:
        """
        根据 node_index 列表获取对应的 node_name。

        :param node_indices: 节点索引列表。
        :return: 字典映射 node_index 到 node_name。
        """
        node_indices_str = [str(index) for index in node_indices]
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n) WHERE n.node_index IN $indices RETURN n.node_index AS node_index, n.node_name AS node_name",
                indices=node_indices_str
            )
            return {record["node_index"]: record["node_name"] for record in result}

    def get_all_relationship_types(self) -> List[str]:
        """
        获取Neo4j数据库中所有的关系类型。

        :return: 关系类型列表。
        """
        with self.driver.session() as session:
            result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
            return [record["relationshipType"] for record in result]

    def find_k_hop_paths_by_name(
        self, node_name: str, k: int, relationship_types: Optional[List[str]] = None
    ) -> Tuple[List[str], float]:
        """
        基于 node_name 查找指定节点的 k 跳路径，支持关系类型过滤。

        :param node_name: 节点名称。
        :param k: 跳数。
        :param relationship_types: 需要限制的关系类型列表。
        :return: 路径列表和执行时间。
        """
        total_start_time = time.time()

        if relationship_types:
            # 构建关系类型字符串，例如: REL1|REL2|REL3
            rel_types_str = "|".join(rel for rel in relationship_types)
            relationship_filter = f"[:{rel_types_str}*{k}..{k}]"
        else:
            relationship_filter = f"[*{k}..{k}]"

        query = f"""
        MATCH p = (n)-{relationship_filter}-(m)
        WHERE n.node_name = $node_name
        RETURN nodes(p) AS Nodes, relationships(p) AS Relationships
        """

        with self.driver.session() as session:
            result = session.run(query, parameters={"node_name": node_name})

            paths = []
            for record in result:
                nodes = [node["node_name"] if "node_name" in node else "Unknown" for node in record["Nodes"]]
                relationships = [rel.type if rel else "Unknown" for rel in record["Relationships"]]

                path = []
                for i in range(len(relationships)):
                    path.append(nodes[i])
                    path.append(f"-[{relationships[i]}]->")
                path.append(nodes[-1])

                paths.append(" ".join(path))

        total_end_time = time.time()
        total_time = total_end_time - total_start_time

        return paths, total_time

    def find_k_hop_paths_by_index(
        self, node_index: str, k: int, relationship_types: Optional[List[str]] = None
    ) -> Tuple[List[str], float]:
        """
        基于 node_index 查找指定节点的 k 跳路径，支持关系类型过滤。

        :param node_index: 节点索引。
        :param k: 跳数。
        :param relationship_types: 需要限制的关系类型列表。
        :return: 路径列表和执行时间。
        """
        total_start_time = time.time()

        if relationship_types:
            rel_types_str = "|".join(rel for rel in relationship_types)
            relationship_filter = f"[:{rel_types_str}*{k}..{k}]"
        else:
            relationship_filter = f"[*{k}..{k}]"

        query = f"""
        MATCH p = (n)-{relationship_filter}-(m)
        WHERE n.node_index = $node_index
        RETURN nodes(p) AS Nodes, relationships(p) AS Relationships
        """

        with self.driver.session() as session:
            result = session.run(query, parameters={"node_index": node_index})

            paths = []
            for record in result:
                nodes = [node["node_name"] if "node_name" in node else "Unknown" for node in record["Nodes"]]
                relationships = [rel.type if rel else "Unknown" for rel in record["Relationships"]]

                path = []
                for i in range(len(relationships)):
                    path.append(nodes[i])
                    path.append(f"-[{relationships[i]}]->")
                path.append(nodes[-1])

                paths.append(" ".join(path))

        total_end_time = time.time()
        total_time = total_end_time - total_start_time

        return paths, total_time

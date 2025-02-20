import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os

from openai import OpenAI
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_community.graphs import Neo4jGraph
from load_config import CHAT_MODEL, API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


class Entity(BaseModel):
    id: str
    type: str
    name: str

class Relation(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    type: str

class GraphData(BaseModel):
    entities: List[Entity]
    relations: List[Relation]

# 三元组：头部实体 ➡️ 关系 ➡️ 尾部实体
HERD_ENTITIES = ["症状", "疾病", "药物", "治疗方法", "患者特征"]
RELATION_TYPES = ["导致", "缓解", "治疗", "伴随", "属于", "包含"]

class LLMGraphBuilder:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY)
        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )

    def extract_entities_and_relations(self, text: str) -> GraphData:
        completion = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at structured data extraction. "
                        "You will be given unstructured text and should extract entities and relations "
                        "to form knowledge graph triples."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
                    请从以下文本中提取实体和关系，形成知识图谱的三元组。
                    实体类型是开放的，可以根据文本内容自行判断，常见的包括但不限于：{', '.join(HERD_ENTITIES)} 等。
                    关系类型是固定的，只能从以下类型中选择：{', '.join(RELATION_TYPES)}
                    请以JSON格式输出，格式如下：
                    {{
                        "entities": [
                            {{"id": "E1", "type": "症状", "name": "头痛"}},
                            {{"id": "E2", "type": "疾病", "name": "偏头痛"}}
                        ],
                        "relations": [
                            {{"from": "E1", "to": "E2", "type": "属于"}}
                        ]
                    }}
                    注意：关系的type必须是给定的几种之一。
                    文本内容：
                    {text}
                    """
                }
            ],
            # 这里用 "response_format" 只是示例，实际你可以视自己的需求来决定
            response_format={"type": "json_object"}
        )
        return GraphData.parse_raw(completion.choices[0].message.content)

    def validate_relations(self, graph_data: GraphData) -> GraphData:
        # 只保留允许的关系类型
        valid_relations = [
            rel for rel in graph_data.relations
            if rel.type in RELATION_TYPES
        ]
        return GraphData(entities=graph_data.entities, relations=valid_relations)

    def generate_cypher_queries(self, graph_data: GraphData, graph_label: str) -> List[str]:
        """
        graph_label 用于区分不同子文件夹对应的知识图谱。
        可以将 graph_label 视为一个额外的标签（Label），比如 "FolderA"、"FolderB" 等。
        """
        queries = []

        # 生成创建/合并实体的 Cypher 语句：这里在实体类型的 Label 前增加一个 "graph_label"
        # 例如 MERGE (e:FolderA:症状 {id: "E1", name: "头痛"})
        # 这样就能在同一个 Neo4j 数据库里，用不同的 Label 区分子文件夹
        for entity in graph_data.entities:
            query = f"""
            MERGE (e:{graph_label}:{entity.type} {{id: '{entity.id}'}})
            ON CREATE SET e.name = '{entity.name}'
            ON MATCH SET e.name = '{entity.name}'
            """
            queries.append(query)

        # 生成创建/合并关系的 Cypher 语句：关系两端也加上 graph_label
        # 使得只有同一子文件夹 (graph_label) 下的实体才能互相关联
        for relation in graph_data.relations:
            query = f"""
            MATCH (e1:{graph_label} {{id: '{relation.from_}'}})
            MATCH (e2:{graph_label} {{id: '{relation.to}'}})
            MERGE (e1)-[r:{relation.type}]->(e2)
            """
            queries.append(query)
        return queries

    def build_graph(self, text: str, graph_label: str):
        # 提取实体和关系
        graph_data = self.extract_entities_and_relations(text)
        # 校验关系合法性
        validated_graph_data = self.validate_relations(graph_data)
        # 生成 Cypher 查询
        queries = self.generate_cypher_queries(validated_graph_data, graph_label)
        # 执行 Cypher 查询，更新 Neo4j 图数据库
        for query in queries:
            self.graph.query(query)
        print(f"知识图谱 {graph_label} 更新完成。")

    def query_graph(self, cypher_query: str) -> List[Dict[str, Any]]:
        return self.graph.query(cypher_query)


if __name__ == "__main__":
    # 假设根目录为 "database"
    root_folder_path = "./data/raw_knowledge"
    
    builder = LLMGraphBuilder()
    
    # 获取 root_folder_path 下所有的子文件夹
    subfolders = [
        d for d in os.listdir(root_folder_path)
        if os.path.isdir(os.path.join(root_folder_path, d))
    ]
    
    # 遍历每个子文件夹，并将其视作一个独立的知识图谱
    for subfolder_name in subfolders:
        subfolder_path = os.path.join(root_folder_path, subfolder_name)
        
        # 获取子文件夹下所有的 .txt 和 .md 文件
        all_files = os.listdir(subfolder_path)
        text_files = [
            f for f in all_files
            if f.endswith(".txt") or f.endswith(".md")
        ]
        
        # 依次读取每个文本文件的内容并写入当前子文件夹的知识图谱
        for filename in text_files:
            file_path = os.path.join(subfolder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
            
            # 这里的 graph_label 用子文件夹名 subfolder_name 来区分
            builder.build_graph(text, graph_label=subfolder_name)

    # 测试查询：比如查询所有 Label 中名为 '头痛' 的节点
    # 这里可以写类似 "MATCH (n) WHERE n.name='头痛' RETURN n"，
    # 或者给定具体 Label "MATCH (n:某个文件夹名) RETURN n" 之类的灵活查询。
    query_result = builder.query_graph("MATCH (n) RETURN n LIMIT 50")
    print("查询结果:", query_result)
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
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
                {"role": "system", "content": "You are an expert at structured data extraction. You will be given unstructured text and should extract entities and relations to form knowledge graph triples."},
                {"role": "user", "content": f"""
                请从以下文本中提取实体和关系，形成知识图谱的三元组。
                实体类型是开放的，可以根据文本内容自行判断，常见的包括但不限于：{', '.join(HERD_ENTITIES)}等。
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
                """}
            ],
            response_format={"type": "json_object"}
        )
        return GraphData.parse_raw(completion.choices[0].message.content)

    def validate_relations(self, graph_data: GraphData) -> GraphData:
        valid_relations = [rel for rel in graph_data.relations if rel.type in RELATION_TYPES]
        return GraphData(entities=graph_data.entities, relations=valid_relations)

    def generate_cypher_queries(self, graph_data: GraphData) -> List[str]:
        queries = []
        for entity in graph_data.entities:
            query = f"""
            MERGE (e:{entity.type} {{id: '{entity.id}'}})
            ON CREATE SET e.name = '{entity.name}'
            ON MATCH SET e.name = '{entity.name}'
            """
            queries.append(query)
        for relation in graph_data.relations:
            query = f"""
            MATCH (e1 {{id: '{relation.from_}'}})
            MATCH (e2 {{id: '{relation.to}'}})
            MERGE (e1)-[r:{relation.type}]->(e2)
            """
            queries.append(query)
        return queries

    def build_graph(self, text: str):
        graph_data = self.extract_entities_and_relations(text)
        validated_graph_data = self.validate_relations(graph_data)
        queries = self.generate_cypher_queries(validated_graph_data)
        for query in queries:
            self.graph.query(query)
        print("知识图谱更新完成。")

    def query_graph(self, cypher_query: str) -> List[Dict[str, Any]]:
        return self.graph.query(cypher_query)


if __name__ == "__main__":
    builder = LLMGraphBuilder()
    with open("database/file/test1.md", "r", encoding="utf-8") as file:
        text1 = file.read()
    builder.build_graph(text1)
    # with open("database/file/test2.txt", "r", encoding="utf-8") as file:
    #     text2 = file.read()
    # builder.build_graph(text2)
    query_result = builder.query_graph("MATCH (n) RETURN n LIMIT 10")
    print("查询结果:", query_result)
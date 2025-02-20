import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import instructor
from openai import OpenAI
from typing import List, Optional
from pydantic import BaseModel, Field
from neo4j import GraphDatabase

from load_config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, API_KEY, CHAT_MODEL

import logging
from logging_config import setup_logging

logger = logging.getLogger(__name__)
es = setup_logging()

NODE_TYPES = ["症状", "疾病", "药物", "治疗方法", "患者特征"]
VALID_RELATIONSHIPS = ["导致", "缓解", "治疗", "伴随", "属于", "包含"]

class QueryElement(BaseModel):
    head_entity: str = Field(..., description="查询的头部实体")
    relationship: str = Field(..., description=f"关系，必须是以下之一：{VALID_RELATIONSHIPS}")
    tail_type: Optional[str] = Field(
        None,
        description="尾部实体的类型，99%的情况下应该保持为None，除非明确要求特定节点类型"
    )

class QueryElements(BaseModel):
    queries: List[QueryElement] = Field(..., description="查询元素列表")

class GraphQA:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.client = instructor.from_openai(OpenAI(api_key=API_KEY), mode=instructor.Mode.JSON)

    def generate_cypher_query(self, query_element: QueryElement, graph_label: Optional[str] = None) -> str:
        """
        生成 Cypher 查询语句；若传入了 graph_label，则只匹配该子知识图谱。
        """
        tail_type_clause = f":{query_element.tail_type}" if query_element.tail_type else ""
        
        # 如果有子知识图谱 label，则构造一个 ":FolderA" 这样的字符串，否则为空
        subfolder_head = f":{graph_label}" if graph_label else ""
        subfolder_tail = f":{graph_label}" if graph_label else ""
        
        # 把上面构造的 label 放到 MATCH 里
        cypher = f"""
        MATCH (h{subfolder_head} {{name: '{query_element.head_entity}'}})-[r:{query_element.relationship}]->(t{subfolder_tail}{tail_type_clause})
        RETURN h, r, t
        LIMIT 10
        """
        logger.debug(f"Generated Cypher query: {cypher}")
        return cypher

    def execute_cypher_query(self, cypher_query: str) -> List[dict]:
        with self.driver.session() as session:
            result = session.run(cypher_query)
            records = result.fetch(10)
            logger.info(f"Executed Cypher query, returned {len(records)} records")
            simplified_results = []
            for record in records:
                simplified_results.append({
                    'h': record['h']['name'],
                    'r': record['r'].type,
                    't': record['t']['name']
                })
            return simplified_results

    def query(self, question: str, graph_label: Optional[str] = None) -> List[dict]:
        """
        增加 graph_label，用于指定要检索哪个子知识图谱；
        如果不传，则默认检索全局。
        """
        logger.info(f"Received question: {question}")
        try:
            query_elements: QueryElements = self.client.chat.completions.create(
                model=CHAT_MODEL,
                response_model=QueryElements,
                messages=[
                    {
                        "role": "system",
                        "content": f"""
                        你是一个AI助手，专门用于理解问题并提取查询Neo4j图数据库所需的关键元素。
                        有效的关系类型包括：{', '.join(VALID_RELATIONSHIPS)}
                        节点类型包括：{', '.join(NODE_TYPES)}
                        如果问题涉及多个实体或关系，请生成多个查询元素。每个查询元素应包含头部实体和关系。
                        在99%的情况下，不需要指定尾部实体的类型。只有在问题中明确要求特定类型的尾部实体时才应指定尾部类型。
                        注意：头部实体通常与病症或疾病术语有关。
                        """
                    },
                    {
                        "role": "user",
                        "content": f"请从以下问题中提取所需的查询元素：{question}"
                    }
                ]
            )
            
            logger.info(f"Generated {len(query_elements.queries)} query elements")
            
            all_results = []
            for query_element in query_elements.queries:
                # 调用 generate_cypher_query 时，额外传入 graph_label
                cypher_query = self.generate_cypher_query(query_element, graph_label=graph_label)
                logger.debug(f"Generated Cypher query: {cypher_query}")
                
                result = self.execute_cypher_query(cypher_query)
                all_results.extend(result)
                
                if not result:
                    logger.warning(f"Empty result for query: {cypher_query}")
            
            if not all_results:
                logger.warning("All queries returned empty results")
            
            return all_results
        except Exception as e:
            logger.error(f"Error occurred during query processing: {str(e)}", exc_info=True)
            return []

    def close(self):
        self.driver.close()
        logger.info("Closed Neo4j driver connection")

def run(query, graph_label=None):
    """
    run 函数同样增加一个可选的 graph_label 参数。
    """
    qa = GraphQA()
    try:
        result = qa.query(query, graph_label=graph_label)
        return result
    finally:
        qa.close()


if __name__ == "__main__":
    # 示例：查询子知识图谱 "FolderA"
    print(run("焦虑障碍会导致什么？", graph_label="FolderA"))
    # 如果想查询全局（包括所有子知识图谱），则可以不传或传 None
    # print(run("焦虑障碍会导致什么？"))
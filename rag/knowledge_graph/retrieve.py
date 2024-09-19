import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import asyncio
import instructor
from openai import AsyncOpenAI
from typing import List, Optional
from pydantic import BaseModel, Field
from neo4j import AsyncGraphDatabase

from load_config import GPT4O, OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

import logging
from logging_config import setup_logging

logger = logging.getLogger(__name__)
es = setup_logging()

NODE_TYPES = ["症状", "疾病", "药物", "治疗方法", "患者特征"]
VALID_RELATIONSHIPS = ["包含", "分类", "伴随", "导致", "缓解", "治疗", "属于"]

class QueryElement(BaseModel):
    head_entity: str = Field(..., description="查询的头部实体")
    relationship: str = Field(..., description=f"关系，必须是以下之一：{VALID_RELATIONSHIPS}")
    tail_type: Optional[str] = Field(None, description=f"尾部实体的类型，如果已知的话，必须是以下之一：{NODE_TYPES}")

class QueryElements(BaseModel):
    queries: List[QueryElement] = Field(..., description="查询元素列表")

class AsyncGraphQA:
    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.client = instructor.apatch(AsyncOpenAI(api_key=OPENAI_API_KEY))

    async def generate_cypher_query(self, query_element: QueryElement) -> str:
        cypher = f"""
        MATCH (h {{name: '{query_element.head_entity}'}})-[r:{query_element.relationship}]->(t{f':{query_element.tail_type}' if query_element.tail_type else ''})
        RETURN h, r, t
        LIMIT 10
        """
        logger.debug(f"Generated Cypher query: {cypher}")
        return cypher

    async def execute_cypher_query(self, cypher_query: str) -> List[dict]:
        async with self.driver.session() as session:
            result = await session.run(cypher_query)
            records = await result.fetch(10)
            logger.info(f"Executed Cypher query, returned {len(records)} records")
            simplified_results = []
            for record in records:
                simplified_results.append({
                    'h': record['h']['name'],
                    'r': record['r'].type,
                    't': record['t']['name']
                })
            return simplified_results

    async def query(self, question: str) -> List[dict]:
        logger.info(f"Received question: {question}")
        try:
            query_elements: QueryElements = await self.client.chat.completions.create(
                model=GPT4O,
                response_model=QueryElements,
                messages=[
                    {"role": "system", "content": f"""
                    你是一个AI助手，专门用于理解问题并提取查询Neo4j图数据库所需的关键元素。
                    节点类型包括：{', '.join(NODE_TYPES)}
                    有效的关系类型包括：{', '.join(VALID_RELATIONSHIPS)}
                    如果问题涉及多个实体或关系，请生成多个查询元素。每个查询元素应包含头部实体、关系，以及可能的尾部实体类型（如果能从问题中推断出）。
                    不需要指定头部实体的类型，因为查询会基于实体名称进行。
                    注意：头部实体通常与精神疾病的术语或病症有关。
                    """},
                    {"role": "user", "content": f"请从以下问题中提取所需的查询元素：{question}"}
                ]
            )
            
            logger.info(f"Generated {len(query_elements.queries)} query elements")
            
            all_results = []
            for query_element in query_elements.queries:
                cypher_query = await self.generate_cypher_query(query_element)
                logger.debug(f"Generated Cypher query: {cypher_query}")
                
                result = await self.execute_cypher_query(cypher_query)
                all_results.extend(result)
                
                if not result:
                    logger.warning(f"Empty result for query: {cypher_query}")
                
            if not all_results:
                logger.warning("All queries returned empty results")
            
            return all_results
        except Exception as e:
            logger.error(f"Error occurred during query processing: {str(e)}", exc_info=True)
            return []

    async def close(self):
        await self.driver.close()
        logger.info("Closed Neo4j driver connection")

async def run(query):
    qa = AsyncGraphQA()
    try:
        result = await qa.query(query)
        return result
    finally:
        await qa.close()

if __name__ == "__main__":
    print(asyncio.run(run("焦虑障碍会导致什么？")))
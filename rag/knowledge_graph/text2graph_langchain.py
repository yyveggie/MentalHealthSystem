import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph

from load_config import GPT4O, OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

llm = ChatOpenAI(temperature=0, model_name=GPT4O, api_key=OPENAI_API_KEY)
llm_transformer = LLMGraphTransformer(llm=llm)

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

with open("data/test1.txt", "r", encoding="utf-8") as file: text = file.read()
documents = [Document(page_content=text)]

graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(f"Nodes: {graph_documents[0].nodes}")
print(f"Relationships: {graph_documents[0].relationships}")

graph.add_graph_documents(graph_documents)

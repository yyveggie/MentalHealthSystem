import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import asyncio
import json
from textwrap import dedent
from langchain_community.utilities import SearxSearchWrapper

from utils.match_words import filter_json_by_keyword

class SearxSearch:
    def __init__(self, searx_host="http://127.0.0.1:8080"):
        self.search = SearxSearchWrapper(searx_host=searx_host)

    async def get_results(self, query):
        results = self.search.results(
            query,
            num_results=20,
            engines=["wikipedia", "encyclopedia", "google", "bing", "wikidata", "duckduckgo", "brave", "exa"],
        )
        return [
            {
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "link": result.get("link", "")
            }
            for result in results
        ]

    async def start_search(self, query: str):
        results = await self.get_results(query)
        return json.dumps(results, ensure_ascii=False, indent=2)

class QueryProcessor:
    def __init__(self):
        self.searx = SearxSearch()
    
    async def __call__(self, search_keywords: str):
        contexts_and_urls = await self.searx.start_search(search_keywords)
        filtered_results = filter_json_by_keyword(contexts_and_urls, search_keywords, "title")
        
        # return dedent(
        #     f"""
        #     我是一个像你一样的 AI 助手，帮助你完成用户的请求。这是我获得的网络数据，请你选择性的摘取重要信息之后，将其重新组织后并返回给用户。
        #     </START>{filtered_results}</END>
        #     在你的回应之后，你必须按以下格式提供你使用的来源：
        #     [你的回应]
        #     来源：
        #     [1] 来源 1 URL
        #     [2] 来源 2 URL
        #     """
        # ).strip()
        return filtered_results

async def run(search_keywords: str):
    processor = QueryProcessor()
    return await processor(search_keywords=search_keywords)

async def main():
    search_keywords = "抑郁症患者该如何生活？"
    result = await run(search_keywords)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
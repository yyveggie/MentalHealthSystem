import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import json
from typing import Dict
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
import os
import PyPDF2
import docx2txt

from load_config import GROQ_API_KEY, GROQ_LLAMA3_70B

ROLE = "专业医疗病历摘要助手"
CONTEXT = """
作为一名专业的医疗病历摘要助手，你的任务是根据患者和医生的对话记录，生成一份专业、简洁而全面的现病史摘要。请遵循以下指导原则：

1. 时间顺序：按照症状出现和发展的时间顺序组织信息。
2. 主要症状：突出描述主要症状，包括其性质、持续时间、频率和严重程度。
3. 相关症状：列出与主要症状相关的次要症状。
4. 诱因和加重/缓解因素：描述可能引发或影响症状的因素。
5. 既往治疗：简要说明患者已尝试的治疗方法及其效果。
6. 影响日常生活的程度：描述症状如何影响患者的日常活动和生活质量。
7. 相关检查：列出患者已经完成的相关检查及结果（如有）。
8. 家族史和个人史：如果对当前病情有影响，简要提及相关的家族病史或个人病史。
9. 用语专业化：使用医学术语，但确保内容仍然清晰易懂。
10. 客观性：保持描述的客观性，不加入个人判断或诊断。
11. 简洁性：保持摘要简洁，通常控制在300-500字之内。

请注意：不要在摘要中包含治疗建议或诊断结论。专注于呈现患者的症状和病情发展过程。
"""

class FileProcessor:
    @staticmethod
    def read_file(file_path: str) -> str:
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        if file_extension == '.pdf':
            return FileProcessor._read_pdf(file_path)
        elif file_extension == '.txt':
            return FileProcessor._read_txt(file_path)
        elif file_extension == '.docx':
            return FileProcessor._read_docx(file_path)
        elif file_extension == '.json':
            return FileProcessor._read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    @staticmethod
    def _read_pdf(file_path: str) -> str:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return ' '.join(page.extract_text() for page in reader.pages)

    @staticmethod
    def _read_txt(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    @staticmethod
    def _read_docx(file_path: str) -> str:
        # 使用 docx2txt 来读取 .docx 文件
        return docx2txt.process(file_path)

    @staticmethod
    def _read_json(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return json.dumps(data, ensure_ascii=False)


class MedicalFileSummarizer:
    def __init__(self):
        self.model = ChatGroq(temperature=0.3, model=GROQ_LLAMA3_70B, api_key=GROQ_API_KEY)
        self.file_processor = FileProcessor()

    def summarize(self, file_content: str) -> str:
        messages = [
            SystemMessage(content=f"你是{ROLE}。{CONTEXT}"),
            HumanMessage(content=f"""
请基于以下医生和患者的对话记录，生成一份专业的现病史摘要：

{file_content}

请确保你的摘要符合之前提供的指导原则，特别注意时间顺序、症状描述的详细程度、客观性和专业性。
            """)
        ]
        
        response = self.model(messages)
        return response.content

    def process_file(self, file_path: str) -> str:
        file_content = self.file_processor.read_file(file_path)
        return self.summarize(file_content)

def run(file_path):
    summarizer = MedicalFileSummarizer()
    
    try:
        summary = summarizer.process_file(file_path)
        return f"现病史摘要:\n{summary}"
    except Exception as e:
        return f"处理文件时出错: {str(e)}"


if __name__ == "__main__":
    file_path = "medical_dialogue.pdf"  # 可以是 .pdf, .txt, .docx, 或 .json
    print(run(file_path=file_path))
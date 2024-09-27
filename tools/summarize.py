import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import json
from prompts import main_system

import os
import PyPDF2
import docx2txt

import logging
from logging_config import setup_logging

logger = logging.getLogger(__name__)
es = setup_logging()

class FileProcessor:
    @staticmethod
    def read_file(file_path: str) -> str:
        logger.info(f"Reading file: {file_path}")
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        try:
            if file_extension == '.pdf':
                content = FileProcessor._read_pdf(file_path)
            elif file_extension == '.txt':
                content = FileProcessor._read_txt(file_path)
            elif file_extension == '.docx':
                content = FileProcessor._read_docx(file_path)
            elif file_extension == '.json':
                content = FileProcessor._read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            logger.info(f"Successfully read file: {file_path}")
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}", exc_info=True)
            raise

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
        return docx2txt.process(file_path)

    @staticmethod
    def _read_json(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return json.dumps(data, ensure_ascii=False)


class MedicalFileSummarizer:
    def __init__(self):
        logger.info("Initializing MedicalFileSummarizer")
        self.file_processor = FileProcessor()

    def summarize_prompt(self, file_content: str) -> str:
        logger.info("Generating summary prompt")
        return main_system.summary_prompt(file_content)

    def process_file(self, file_path: str) -> str:
        logger.info(f"Processing file: {file_path}")
        try:
            file_content = self.file_processor.read_file(file_path)
            summary_prompt = self.summarize_prompt(file_content)
            logger.info(f"Successfully processed file: {file_path}")
            return summary_prompt
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            raise

def run(file_path):
    logger.info(f"Starting file processing for: {file_path}")
    summarizer = MedicalFileSummarizer()
    
    try:
        summary_prompt = summarizer.process_file(file_path)
        logger.info("File processing completed successfully")
        return summary_prompt
    except Exception as e:
        error_message = f"处理文件时出错: {str(e)}"
        logger.error(error_message, exc_info=True)
        return error_message

if __name__ == "__main__":
    file_path = "./database/file/hpi.txt"  # 可以是 .pdf, .txt, .docx, 或 .json
    logger.info("Starting main script execution")
    result = run(file_path=file_path)
    print(result)
    logger.info("Main script execution completed")
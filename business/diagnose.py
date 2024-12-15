import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import json
import instructor
from openai import OpenAI
from typing import Optional, List
from pydantic import BaseModel, Field
import logging

from prompts import main_system
from rag.historical_exp.calculate_similarity import PatientDiagnosisAPI
from load_config import CHAT_MODEL, API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiagnosisPossibility(BaseModel):
    病症: str = Field(..., description="诊断的病症名称")
    置信度: float = Field(..., description="诊断的置信度（0-1之间的浮点数）", ge=0, le=1)
    理由: str = Field(..., description="该诊断的支持理由")

class DiagnosisResult(BaseModel):
    诊断结果: List[DiagnosisPossibility] = Field(
        ..., 
        description="可能的诊断列表，按置信度降序排序",
        min_items=1,
        max_items=5
    )

class MedicalDiagnosisProcessor:
    def __init__(self):
        try:
            self.client = instructor.patch(OpenAI(api_key=API_KEY))
            self.model_name = CHAT_MODEL
            self.historical_exp_api = PatientDiagnosisAPI()
            logger.info("Successfully initialized MedicalDiagnosisProcessor")
        except Exception as e:
            logger.error(f"Error initializing MedicalDiagnosisProcessor: {str(e)}")
            raise

    def process_diagnosis(self, json_input: str) -> Optional[DiagnosisResult]:
        """
        处理诊断请求并返回结构化的诊断结果
        
        Args:
            json_input: 输入的JSON字符串
            
        Returns:
            DiagnosisResult: 包含多个可能诊断的结构化输出
        """
        try:
            input_data = json.loads(json_input)
            logger.info("Successfully parsed JSON input")
            
            vector_results = self.get_vector_results(json_input)
            if not vector_results:
                logger.warning("No vector results found")
                vector_results = []
            
            system_prompt = main_system.diagnosis_system_prompt()
            user_prompt = main_system.diagnosis_user_prompt(json_input=input_data, vector_results=vector_results)
            logger.info("Generated diagnosis prompt")
            
            diagnosis_result = self.client.chat.completions.create(
                model=self.model_name,
                response_model=DiagnosisResult,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            logger.info("Successfully generated diagnosis results")
            return diagnosis_result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error in process_diagnosis: {str(e)}", exc_info=True)
            return None

    def get_vector_results(self, input_data: str) -> list:
        """
        获取向量检索结果
        
        Args:
            input_data: JSON字符串输入
            
        Returns:
            list: 检索结果列表
        """
        try:
            vector_results = self.historical_exp_api.process_query(input_data)
            logger.info(f"Retrieved vector results: {len(vector_results) if vector_results else 0} items")
            return vector_results
        except Exception as e:
            logger.error(f"Error in get_vector_results: {str(e)}", exc_info=True)
            return []

def main():
    try:
        processor = MedicalDiagnosisProcessor()
        test_input = {
            "过敏史": "药物过敏史：未发现；食物过敏史：否认",
            "个人史": "否认长期接触有毒有害物质史，否认严重创伤史，否认长期卧床史，否认手术史。",
            "婚育史": "已婚，已育一子",
            "家族史": "父母健在，否认家族遗传病史",
            "诊疗经过": "患者2023年09月21日08时56分入院，入院后完善相关检查，予以文拉法辛缓释胶囊75mg qd、氯硝西泮0.5mg bid、阿立哌唑5mg qn、右佐匹克隆3mg qm改善情绪及睡眠等治疗，患者病情稳定，过程顺利，于2023年09月30日10时18分出院。"
        }
        
        result = processor.process_diagnosis(json.dumps(test_input))
        if result:
            print("\n诊断结果：")
            for i, diagnosis in enumerate(result.诊断结果, 1):
                print(f"\n可能性 {i}:")
                print(f"病症: {diagnosis.病症}")
                print(f"置信度: {diagnosis.置信度}")
                print(f"理由: {diagnosis.理由}")
        else:
            print("\n未能生成诊断结果")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        print(f"\n程序执行出错: {str(e)}")

if __name__ == "__main__":
    main()
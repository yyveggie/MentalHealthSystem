import json
import instructor
from openai import OpenAI
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
import logging
import uuid

from prompts import main_system
from business.feedback_collector import FeedbackCollector
from rag.historical_exp.calculate_similarity import TwoStageRetrieval

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
            self.historical_exp_api = TwoStageRetrieval()
            self.feedback_collector = FeedbackCollector()
            logger.info("Successfully initialized MedicalDiagnosisProcessor")
        except Exception as e:
            logger.error(f"Error initializing MedicalDiagnosisProcessor: {str(e)}")
            raise

    def process_diagnosis(self, query: Dict[str, str], retrieval_strategy: str = 'two_stage') -> Optional[Dict]:
        """
        处理诊断任务，支持选择检索方案
        :param query: 查询字典（包含不同的病历特征）
        :param retrieval_strategy: 检索方案，'two_stage' 或 'vector_only'
        :return: 诊断结果字典
        """
        try:
            # 根据选择的检索方案进行处理
            if retrieval_strategy == 'two_stage':
                logger.info("使用两阶段检索方案")
                retrieved_list = self.historical_exp_api.retrieve_similar_cases(
                    query_texts=query,
                    n=10,
                    k=3,
                )
            elif retrieval_strategy == 'vector_only':
                logger.info("使用纯向量检索方案")
                retrieved_list = self.historical_exp_api._vector_only_retrieval(query, k=3)
            else:
                logger.error(f"无效的检索方案: {retrieval_strategy}")
                return {"error": f"无效的检索方案: {retrieval_strategy}"}

            # 如果没有检索结果，则设置为空列表
            if not retrieved_list:
                logger.warning("未找到有效的检索结果")
                retrieved_list = []

            # 将检索结果转为字符串形式
            retrieved_results_str = json.dumps(retrieved_list, ensure_ascii=False, indent=2)
            
            # 构建系统和用户提示信息
            system_prompt = main_system.diagnosis_system_prompt()
            user_prompt = main_system.diagnosis_user_prompt(
                query=query,
                retrieved_results=retrieved_results_str
            )
            logger.info("生成诊断提示信息")
            
            # 调用OpenAI接口生成诊断结果
            diagnosis_result = self.client.chat.completions.create(
                model=self.model_name,
                response_model=DiagnosisResult,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            logger.info("成功生成诊断结果")
            diagnosis_session_id = str(uuid.uuid4())
            return {
                "session_id": diagnosis_session_id,
                "diagnosis": diagnosis_result,
                "retrieved_results": retrieved_list,
                "retrieved_results_str": retrieved_results_str
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"诊断处理出错: {str(e)}", exc_info=True)
            return None

    def collect_doctor_feedback(
        self,
        doctor_id: str,
        case_quality_rating: int,
        diagnosis_accuracy: int,
        is_helpful: bool,
        original_query: Dict[str, str],
        diagnosis_results: List[object],
        retrieved_cases: List[object],
        diagnosis_session_id: str,
        comments: Optional[str] = None
    ) -> Optional[str]:
        """
        收集医生的反馈
        """
        try:
            return self.feedback_collector.collect_feedback(
                doctor_id=doctor_id,
                case_quality_rating=case_quality_rating,
                diagnosis_accuracy=diagnosis_accuracy,
                is_helpful=is_helpful,
                original_query=original_query,
                diagnosis_results=diagnosis_results,
                retrieved_cases=retrieved_cases,
                diagnosis_session_id=diagnosis_session_id,
                comments=comments
            )
        except Exception as e:
            logger.error(f"收集反馈出错: {str(e)}")
            return None

# 示例：外部调用收集医生反馈
if __name__ == "__main__":
    # 诊断处理器
    diagnosis_processor = MedicalDiagnosisProcessor()

    # 示例查询
    test_query = {
        "现病史": "现病史：患者于2020.02起出现胸闷不适，渐出现双手及后背发冷，伴有双手发麻，双下肢无力，反复担心自己得了不治之症，由此出现坐立不安、心神不宁，听到大的声响后即出现心慌不适，伴有入睡困难，多次至外院就诊，完善检查未见明显异常，患者对此半信半疑，后胸闷、发冷等症状持续不能缓解，因此感到心情烦躁，疲乏无力，伴有纳差明显，1月内体重下降4kg，更加担心自己的健康情况，于半月前至我科门诊就诊，考虑'焦虑状态'，予以舍曲林、阿普唑仑等药物治疗，患者规律服药，诉心情烦躁及胸闷、发冷等症状较前有好转。患者于2天前因症状好转自行停用阿普唑仑，再次出现上述症状加重。现为进一步诊治，门诊以'焦虑状态'收入我科。    病程中，胃纳差，夜眠差，二便基本正常，体重1月内减轻4kg。否认消极，无冲动，伤人，毁物，外跑行为。",
        "既往史": "高血压病史10年，规律服用降压药物",
        "过敏史": "对青霉素过敏",
        "诊疗经过": "给予利尿、强心等治疗",
        "体格检查": "双肺呼吸音粗，可闻及湿性啰音"
    }

    # 获取诊断结果
    diagnosis_result = diagnosis_processor.process_diagnosis(test_query, retrieval_strategy="two_stage")

    if diagnosis_result:
        # 获取诊断结果
        print("\n诊断结果：")
        for i, diagnosis in enumerate(diagnosis_result["diagnosis"].诊断结果, 1):
            print(f"\n可能性 {i}:")
            print(f"病症: {diagnosis.病症}")
            print(f"置信度: {diagnosis.置信度}")
            print(f"理由: {diagnosis.理由}")

    # 收集医生反馈
    feedback_id = diagnosis_processor.collect_doctor_feedback(
        doctor_id="DOC001",
        case_quality_rating=4,  # 评价案件质量（1-5分）
        diagnosis_accuracy=5,  # 诊断准确性（1-5分）
        is_helpful=True,  # 是否有帮助
        original_query=test_query,  # 原始查询
        diagnosis_results=diagnosis_result["diagnosis"].诊断结果,  # 诊断结果
        retrieved_cases=diagnosis_result["retrieved_results"],  # 检索到的病例
        diagnosis_session_id=diagnosis_result["session_id"],  # 诊断会话ID
        comments="诊断结果准确，帮助了患者的决策。"  # 可选的评论
    )

    if feedback_id:
        print(f"\n反馈已保存，ID: {feedback_id}")

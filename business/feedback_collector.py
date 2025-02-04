import json
from datetime import datetime
from typing import Optional, List, Dict, Union
from pydantic import BaseModel, Field
import logging
import os
import statistics

# 设置日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiagnosisFeedback(BaseModel):
    """医生反馈数据模型"""
    feedback_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    doctor_id: str = Field(..., description="医生的唯一标识符")
    case_quality_rating: int = Field(..., description="诊断质量评分(1-5)", ge=1, le=5)
    diagnosis_accuracy: int = Field(..., description="诊断准确度评分(1-5)", ge=1, le=5)
    is_helpful: bool = Field(..., description="诊断建议是否有帮助")
    comments: Optional[str] = Field(None, description="额外评论")
    
    # 原始查询和结果
    original_query: Dict[str, str]
    diagnosis_results: List[Dict[str, Union[str, float]]]
    retrieved_cases: List[Dict[str, Union[str, float, int]]]

    # 添加诊断会话ID
    diagnosis_session_id: str = Field(..., description="诊断会话的唯一标识符")


class FeedbackCollector:
    """医生反馈收集系统"""
    
    def __init__(self, feedback_dir: str = "./database/feedback_data"):
        """初始化反馈收集器"""
        self.feedback_dir = feedback_dir
        self._ensure_feedback_directory()
    
    def _ensure_feedback_directory(self):
        """确保反馈数据目录存在"""
        try:
            if not os.path.exists(self.feedback_dir):
                os.makedirs(self.feedback_dir)
                logger.info(f"Created feedback directory: {self.feedback_dir}")
        except Exception as e:
            logger.error(f"Error creating feedback directory: {str(e)}")
            raise

    def _convert_diagnosis_results(self, diagnosis_results: List[object]) -> List[Dict[str, Union[str, float]]]:
        """转换诊断结果为字典格式"""
        converted_results = []
        for result in diagnosis_results:
            if hasattr(result, 'model_dump'):
                # 如果是 Pydantic 模型，转换为字典
                converted_results.append(result.model_dump())
            elif isinstance(result, dict):
                # 如果已经是字典，直接使用
                converted_results.append(result)
            else:
                # 添加更多特征
                converted_results.append({
                    "病症": getattr(result, "病症", "Unknown"),
                    "病症ID": getattr(result, "病症ID", "Unknown"),
                    "置信度": float(getattr(result, "置信度", 0.0)),
                    "理由": getattr(result, "理由", ""),
                    "case_id": getattr(result, "case_id", ""),
                    "source_database": getattr(result, "source_database", ""),
                    "retrieval_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        return converted_results

    def _convert_retrieved_cases(self, retrieved_cases: List[object]) -> List[Dict[str, Union[str, float, int]]]:
        """转换检索结果为字典格式"""
        converted_cases = []
        for case in retrieved_cases:
            if isinstance(case, dict):
                # 删除原先的 "case_id" 和 "diagnosis_id"，新增 "inpat_id"
                converted_case = {
                    "patient_id": case.get("patient_id", "Unknown"),
                    "inpat_id": case.get("inpat_id", "Unknown"),
                    "diagnosis": case.get("diagnosis", "Unknown"),
                    "similarity": float(case.get("similarity", 0.0)),
                    "rank": int(case.get("rank", 0)),
                    "source_database": "medical_records",
                    "retrieval_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                converted_cases.append(converted_case)
            else:
                logger.warning(f"Unexpected case format: {type(case)}")
                continue
        return converted_cases

    def collect_feedback(
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
        """收集医生的反馈"""
        try:
            # 转换诊断结果和检索结果为适当的格式
            converted_diagnosis_results = self._convert_diagnosis_results(diagnosis_results)
            converted_retrieved_cases = self._convert_retrieved_cases(retrieved_cases)
            
            feedback = DiagnosisFeedback(
                doctor_id=doctor_id,
                case_quality_rating=case_quality_rating,
                diagnosis_accuracy=diagnosis_accuracy,
                is_helpful=is_helpful,
                comments=comments,
                original_query=original_query,
                diagnosis_results=converted_diagnosis_results,
                retrieved_cases=converted_retrieved_cases,
                diagnosis_session_id=diagnosis_session_id
            )
            
            # 生成文件名
            filename = f"feedback_{feedback.feedback_id}.json"
            filepath = os.path.join(self.feedback_dir, filename)
            
            # 保存反馈
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(feedback.model_dump(), f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully saved feedback: {filepath}")
            return feedback.feedback_id
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {str(e)}")
            return None

    def get_feedback(self, feedback_id: str) -> Optional[DiagnosisFeedback]:
        """获取特定反馈"""
        try:
            filepath = os.path.join(self.feedback_dir, f"feedback_{feedback_id}.json")
            if not os.path.exists(filepath):
                logger.warning(f"Feedback not found: {feedback_id}")
                return None
                
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return DiagnosisFeedback(**data)
                
        except Exception as e:
            logger.error(f"Error retrieving feedback: {str(e)}")
            return None
    
    def get_all_feedback(self) -> List[DiagnosisFeedback]:
        """获取所有反馈"""
        feedback_list = []
        try:
            for filename in os.listdir(self.feedback_dir):
                if filename.startswith("feedback_") and filename.endswith(".json"):
                    filepath = os.path.join(self.feedback_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        feedback_list.append(DiagnosisFeedback(**data))
            return feedback_list
        except Exception as e:
            logger.error(f"Error retrieving all feedback: {str(e)}")
            return feedback_list
        
    def get_feedback_statistics(self) -> Dict:
        """获取反馈统计信息"""
        all_feedback = self.get_all_feedback()
        return {
            "average_quality_rating": statistics.mean([f.case_quality_rating for f in all_feedback]),
            "average_accuracy": statistics.mean([f.diagnosis_accuracy for f in all_feedback]),
            "helpful_percentage": len([f for f in all_feedback if f.is_helpful]) / len(all_feedback)
        }

def main():
    """示例用法"""
    try:
        # 初始化反馈收集器
        feedback_collector = FeedbackCollector()
        
        # 模拟诊断结果
        test_query = {
            "现病史": "患者出现胸闷不适...",
            "既往史": "高血压病史10年...",
            "过敏史": "对青霉素过敏",
        }
        
        diagnosis_results = [
            {
                "病症": "焦虑症",
                "置信度": 0.85,
                "理由": "患者表现出典型的焦虑症状..."
            }
        ]
        
        retrieved_cases = [
            {
                "patient_id": "12345",
                "diagnosis": "焦虑障碍",
                "similarity": 0.75
            }
        ]
        
        # 收集反馈
        feedback_id = feedback_collector.collect_feedback(
            doctor_id="DOC001",
            case_quality_rating=4,
            diagnosis_accuracy=5,
            is_helpful=True,
            original_query=test_query,
            diagnosis_results=diagnosis_results,
            retrieved_cases=retrieved_cases,
            diagnosis_session_id="session123",
            comments="诊断建议对临床决策有帮助"
        )
        
        if feedback_id:
            print(f"反馈已保存，ID: {feedback_id}")
            
            # 获取保存的反馈
            feedback = feedback_collector.get_feedback(feedback_id)
            if feedback:
                print("\n保存的反馈内容：")
                print(json.dumps(feedback.model_dump(), ensure_ascii=False, indent=2))
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        print(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    main()
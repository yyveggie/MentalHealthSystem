# -*- coding: utf-8 -*-

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import Dict, Any, Type, Union
from openai import OpenAI
from abc import ABC

from preprocess.structural_standard.general_information import GeneralInformation, rewrite_general_information
from preprocess.structural_standard.present_illness_history import PresentIllnessHistory, rewrite_present_illness_history
from preprocess.structural_standard.medical_history import MedicalHistory, rewrite_medical_history
from preprocess.structural_standard.family_history import FamilyHistory, rewrite_family_history
from preprocess.structural_standard.main_complaint import ChiefComplaint, rewrite_chief_complaint
from preprocess.structural_standard.personal_history import PersonalHistory, rewrite_personal_history
from preprocess.structural_standard.all_features_at_once import AllFeatures

from config_loader import load_specific_config

config = load_specific_config(['CHAT_MODEL', 'BASE_URL', 'API_KEY'])
CHAT_MODEL = config['CHAT_MODEL']
BASE_URL = config["BASE_URL"]
API_KEY = config['API_KEY']


FEATURE_CLASS_MAP = {
    "一般资料": GeneralInformation,
    "现病史": PresentIllnessHistory,
    "既往史": MedicalHistory,
    "家族史": FamilyHistory,
    "主诉": ChiefComplaint,
    "个人史": PersonalHistory,
    "所有特征": AllFeatures,
}

FEATURE_CLASS_MAP_TEXT = {
    "一般资料": rewrite_general_information,
    "现病史": rewrite_present_illness_history,
    "主诉": rewrite_chief_complaint,
    "既往史": rewrite_medical_history,
    "家族史": rewrite_family_history,
    "个人史": rewrite_personal_history,
}


class BaseProcessor(ABC):
    """
    处理器基类，包含基础的处理逻辑：调用 OpenAI 接口将文本解析为结构化数据
    """
    def __init__(self):
        self.openai_client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
            )

    def process_text(self, text: str, structure_class: Type) -> Any:
        """
        使用 OpenAI 的 parse 功能处理文本，并将文本解析为指定的结构化数据结构。
        """
        if not text or str(text).lower() == 'nan':
            raise ValueError("Empty or invalid text input")
            
        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model=CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一名精神疾病诊断专家，负责病史规范化采集。请将以下文本解析为结构化数据，确保所有必需字段都有合理的值。"
                                   "对于未提及的可选字段可以使用 null。"
                    },
                    {
                        "role": "user",
                        "content": text
                    },
                ],
                response_format=structure_class
            )
            return completion.choices[0].message.parsed
            
        except Exception as e:
            print(f"处理文本时发生错误: {str(e)}")
            raise


class ExternalInputProcessor(BaseProcessor):
    """
    外部数据处理器：直接接收文本和类型（不依赖任何数据库）
    """
    def __init__(self, output_mode: str = "all_features"):
        """
        增加了 output_mode 超参数，用于选择输出格式：
        - "dict": 输出字典化结构
        - "text": 文本重写
        - "all_features": 一次性提取所有类型信息（默认）
        """
        super().__init__()
        self.output_mode = output_mode

    def process_single_text(
        self,
        text: str,
        feature_type: str,
        output_mode: str = None
    ) -> Union[Dict, str]:
        """
        处理单条文本数据（需要提前指定类型）
        
        参数:
            text (str): 要处理的文本
            feature_type (str): 特征类型，如 "现病史"、"既往史"等
            output_mode (str): 输出模式，可覆盖初始化时设置的默认值。
                               - "dict": 字典化结构
                               - "text": 文本重写
        返回:
            Dict 或 str: 根据 output_mode 返回字典或重写文本
        """
        # 如果没有指定 output_mode，则使用初始化设定的默认值
        if not output_mode:
            output_mode = self.output_mode

        if feature_type not in FEATURE_CLASS_MAP:
            raise ValueError(f"不支持的特征类型: {feature_type}")
            
        structure_class = FEATURE_CLASS_MAP[feature_type]

        # 根据 output_mode 分别处理
        if output_mode == "dict":
            try:
                structured_data = self.process_text(text, structure_class)
            except Exception as e:
                print(f"处理文本时发生错误: {type(e).__name__} - {str(e)}")
                raise
            return structured_data.model_dump(by_alias=True)
        elif output_mode == "text":
            rewritten_text = FEATURE_CLASS_MAP_TEXT[feature_type](text, feature_type)
            return rewritten_text
        elif output_mode == "all_features":
            try:
                all_features = self.process_text(text, structure_class)
            except Exception as e:
                print(f"处理文本时发生错误: {type(e).__name__} - {str(e)}")
                raise
            return all_features.model_dump(by_alias=True)
        else:
            raise ValueError(f"不支持的输出模式: {output_mode}")


if __name__ == "__main__":
    # 示例：初始化处理器
    processor = ExternalInputProcessor()

    # ========== 1) 已有的字典化或文本重写方式（示例） ==========
    # text_result = processor.process_single_text(
    #     # "患者有高血压病史10年，长期服用降压药物", 
    #     # "既往史",
    #     "患者于2020.02起出现胸闷不适，渐出现双手及后背发冷，伴有双手发麻，双下肢无力，反复担心自己得了不治之症，由此出现坐立不安、心神不宁，听到大的声响后即出现心慌不适，伴有入睡困难，多次至外院就诊，完善检查未见明显异常，患者对此半信半疑，后胸闷、发冷等症状持续不能缓解，因此感到心情烦躁，疲乏无力，伴有纳差明显，1月内体重下降4kg，更加担心自己的健康情况，于半月前至我科门诊就诊，考虑'焦虑状态'，予以舍曲林、阿普唑仑等药物治疗，患者规律服药，诉心情烦躁及胸闷、发冷等症状较前有好转。患者于2天前因症状好转自行停用阿普唑仑，再次出现上述症状加重。现为进一步诊治，门诊以'焦虑状态'收入我科。    病程中，胃纳差，夜眠差，二便基本正常，体重1月内减轻4kg。否认消极，无冲动，伤人，毁物，外跑行为。",
    #     "现病史",
    #     output_mode="dict"
    # )

    # text_result = processor.process_single_text(
    #     "患者于2020.02起出现胸闷不适，渐出现双手及后背发冷，伴有双手发麻，双下肢无力，反复担心自己得了不治之症，由此出现坐立不安、心神不宁，听到大的声响后即出现心慌不适，伴有入睡困难，多次至外院就诊，完善检查未见明显异常，患者对此半信半疑，后胸闷、发冷等症状持续不能缓解，因此感到心情烦躁，疲乏无力，伴有纳差明显，1月内体重下降4kg，更加担心自己的健康情况，于半月前至我科门诊就诊，考虑'焦虑状态'，予以舍曲林、阿普唑仑等药物治疗，患者规律服药，诉心情烦躁及胸闷、发冷等症状较前有好转。患者于2天前因症状好转自行停用阿普唑仑，再次出现上述症状加重。现为进一步诊治，门诊以'焦虑状态'收入我科。    病程中，胃纳差，夜眠差，二便基本正常，体重1月内减轻4kg。否认消极，无冲动，伤人，毁物，外跑行为。",
    #     "现病史",
    #     output_mode="text"
    # )

    # ========== 2) 一次性提取所有类型信息（示例） ==========
    sample_text = (
        "患者于2020.02起出现胸闷不适，渐出现双手及后背发冷，伴有双手发麻，双下肢无力，反复担心自己得了不治之症，由此出现坐立不安、心神不宁，听到大的声响后即出现心慌不适，伴有入睡困难，多次至外院就诊，完善检查未见明显异常，患者对此半信半疑，后胸闷、发冷等症状持续不能缓解，因此感到心情烦躁，疲乏无力，伴有纳差明显，1月内体重下降4kg，更加担心自己的健康情况，于半月前至我科门诊就诊，考虑'焦虑状态'，予以舍曲林、阿普唑仑等药物治疗，患者规律服药，诉心情烦躁及胸闷、发冷等症状较前有好转。患者于2天前因症状好转自行停用阿普唑仑，再次出现上述症状加重。现为进一步诊治，门诊以'焦虑状态'收入我科。病程中，胃纳差，夜眠差，二便基本正常，体重1月内减轻4kg。否认消极，无冲动，伤人，毁物，外跑行为。"
    )
    text_result = processor.process_single_text(
        sample_text,
        "所有特征",
        output_mode="all_features"
    )
    
    print(text_result)
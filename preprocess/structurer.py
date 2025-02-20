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

from load_config import CHAT_MODEL, API_KEY


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
    def __init__(self, api_key: str):
        self.openai_client = OpenAI(api_key=api_key)

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
    def __init__(self, api_key: str, output_mode: str = "dict"):
        """
        增加了 output_mode 超参数，用于选择输出格式：
        - "dict": 输出字典化结构（默认）
        - "text": 文本重写
        """
        super().__init__(api_key)
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
            structured_data = self.process_text(text, structure_class)
            return structured_data.model_dump(by_alias=True)
        elif output_mode == "text":
            rewritten_text = FEATURE_CLASS_MAP_TEXT[feature_type](text, feature_type)
            return rewritten_text
        elif output_mode == "all_features":
            all_features = self.process_text(text, structure_class)
            return all_features.model_dump(by_alias=True)
        else:
            raise ValueError(f"不支持的输出模式: {output_mode}")


if __name__ == "__main__":
    # 示例：初始化处理器
    processor = ExternalInputProcessor(API_KEY, output_mode="dict")

    # ========== 1) 已有的字典化或文本重写方式（示例） ==========
    # dict_result = processor.process_single_text(
    #     "患者有高血压病史10年，长期服用降压药物", 
    #     "既往史",
    #     output_mode="dict"
    # )
    # print(dict_result)

    # text_result = processor.process_single_text(
    #     "患者，男性，45岁，因“反复胸闷、气短3个月，加重1周”前来就诊。",
    #     "现病史",
    #     output_mode="text"
    # )
    # print(text_result)

    # ========== 2) 一次性提取所有类型信息（示例） ==========
    sample_text = (
        "患者，男性，45岁，因“反复胸闷、气短3个月，加重1周”前来就诊。"
        "患者自述3个月前开始无明显诱因下出现胸闷、气短症状，活动后加重，休息后可稍缓解。"
        "既往有高血压病史5年；吸烟20年，家族中父亲有冠心病史。"
    )
    all_features_result = processor.process_single_text(
        sample_text,
        "所有特征",
        output_mode="all_features"
    )
    print(all_features_result)
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pydantic import BaseModel, Field
from openai import OpenAI
from textwrap import dedent
from load_config import CHAT_MODEL, API_KEY


class ChiefComplaint(BaseModel):
    """
    主诉数据模型
    用简明扼要的词句描述患者的主要症状和病期，避免使用精神科专业术语，字数尽量不超过25个字。
    """
    main_complaintd: str = Field(
        ...,
        alias="主诉",
        max_length=25,
        description="患者来诊或转诊的最主要原因和症状概括(限25字内)"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True

def rewrite_chief_complaint(text: str, feature_type: str) -> str:
    try:
        openai_client = OpenAI(api_key=API_KEY)
        completion = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": dedent(
                        "你是一名精神疾病诊断专家，负责病史规范化采集。"
                        f"下面是针对{feature_type}的具体规范要求：\n"
                        "——————————————\n"
                        "主诉即来诊或转诊的原因，用简明扼要的词句描述患者的主要症状和病期，"
                        "也是医生对现病史所作的最简明的概括，字数一般不超过25个字。"
                        "尽量使用供史者的语言，或在不改变原意的前提下稍做文字加工，"
                        "避免使用精神科专业术语。\n"
                        "——————————————\n"
                        f"请根据以上规范，对用户提供的文本进行改写，使之符合{feature_type}所需的内容结构，"
                        "并用简洁易懂的语言呈现。如原文信息不足，可适度补全或合理留空。"
                    )
                },
                {
                    "role": "user",
                    "content": f"请将以下文本以『{feature_type}』的角度进行改写：\n{text}"
                }
            ]
        )
        
        # 获取模型返回的文本
        rewritten_text = completion.choices[0].message.content
        return rewritten_text

    except Exception as e:
        print(f"处理文本时发生错误: {str(e)}")
        raise
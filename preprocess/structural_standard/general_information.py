import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pydantic import BaseModel, Field
from textwrap import dedent
from typing import Optional
from openai import OpenAI

from load_config import CHAT_MODEL, API_KEY


class GeneralInformation(BaseModel):
    """
    一般资料：根据规范仅保留必需的主要字段
    """
    name: str = Field(..., alias="姓名", description="患者姓名")
    gender: Optional[str] = Field(None, alias="性别", description="患者性别")
    age_or_birthdate: Optional[str] = Field(
        None, alias="年龄或出生年月", description="患者年龄或出生年月(如为儿童，最好记录出生年月)"
    )
    place_of_origin: Optional[str] = Field(None, alias="籍贯", description="患者籍贯，用于了解方言、生活习惯等")
    occupation: Optional[str] = Field(None, alias="职业", description="患者职业，如“中学语文老师”")
    marital_status: Optional[str] = Field(None, alias="婚姻", description="患者婚姻状况")
    family_address: Optional[str] = Field(None, alias="家庭地址", description="患者家庭住址")
    workplace_address: Optional[str] = Field(None, alias="工作单位及地址", description="工作单位及地址")
    phone_number: Optional[str] = Field(None, alias="电话", description="联系电话")
    minority_ethnicity: Optional[str] = Field(None, alias="少数民族", description="若为少数民族，需注明")

    # 供史者信息（若病史由他人提供）
    info_provider_name: Optional[str] = Field(None, alias="供史者姓名", description="提供病史者的姓名")
    provider_relationship: Optional[str] = Field(None, alias="与患者关系", description="供史者与患者的关系")
    knowledge_level: Optional[str] = Field(None, alias="对患者了解程度", description="供史者对患者情况的了解程度")
    history_date: Optional[str] = Field(None, alias="供史日期", description="提供病史的日期")

    class Config:
        # 允许通过字段的别名进行传值
        populate_by_name = True
        allow_population_by_field_name = True
        
def rewrite_general_information(text: str, feature_type: str) -> str:
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
                        "一般资料主要包括姓名、性别、年龄或出生年月(儿童最好是记录出生年月)、"
                        "籍贯(有助于了解患者使用的方言及其生活习惯)、职业(最好能有具体工种,如“老师”"
                        "明确为“中学语文老师”)、婚姻、家庭地址、工作单位及地址、电话。"
                        "如果是少数民族,须注明。如果病史是由他人提供,还应包括供史者姓名、与患者的关系、"
                        "对患者的了解程度,最后写明供史日期。\n"
                        "——————————————\n"
                        f"请根据以上规范，对用户提供的文本进行改写，使之符合{feature_type}所需的内容结构，"
                        "并尽可能提取或补全关键信息。若原文信息不足，可使用合理推测或留空处理。"
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
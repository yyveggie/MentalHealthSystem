import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pydantic import BaseModel, Field
from textwrap import dedent
from typing import Optional, List
from openai import OpenAI

from load_config import CHAT_MODEL, API_KEY


class FamilyMember(BaseModel):
    """
    家庭成员信息
    可用于描述父母、祖父母、兄弟姐妹、子女等的健康和生存状况，
    以及可能的特殊性格或不良嗜好(如酗酒)等。
    """
    living_status: str = Field(
        ...,
        alias="生存状态",
        description="生存状态：健在/已故"
    )
    health_condition: Optional[str] = Field(
        None,
        alias="健康状况",
        description="健康状况描述，如有无严重疾病、精神问题等"
    )
    age: Optional[str] = Field(
        None,
        alias="年龄",
        description="年龄或大致年龄段(若已故可说明生前年龄)"
    )
    occupation: Optional[str] = Field(
        None,
        alias="职业",
        description="当前或生前职业"
    )
    personality_or_habits: Optional[str] = Field(
        None,
        alias="性格与习惯",
        description="如性格怪僻、酗酒、孤僻、不婚等特殊情况"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class Disease(BaseModel):
    """
    疾病信息
    用于记录家族中具体疾病的名称、患病成员及治疗情况等。
    """
    name: str = Field(
        ...,
        alias="疾病名称",
        description="疾病名称，如精神分裂症、双相情感障碍等"
    )
    affected_relatives: List[str] = Field(
        ...,
        alias="患病亲属",
        description="具体哪些家庭成员患有该疾病，可填写如父亲、姑姑等"
    )
    treatment: Optional[str] = Field(
        None,
        alias="治疗情况",
        description="该疾病的治疗方式或效果，如服药、住院、康复情况"
    )
    current_status: Optional[str] = Field(
        None,
        alias="当前状态",
        description="病情目前的状态：缓解中、持续、复发等"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class FamilyHistory(BaseModel):
    """
    家族史数据模型
    包含家族遗传史(三代内精神疾病或特殊性格、酗酒、近亲结婚等)
    以及家庭整体情况(和谐度、经济状况、居住环境、邻里关系、特殊传统等)。
    """
    # 父母及祖父母信息
    father: Optional[FamilyMember] = Field(
        None,
        alias="父亲状况",
        description="父亲的健康和生存状况、年龄、性格等"
    )
    mother: Optional[FamilyMember] = Field(
        None,
        alias="母亲状况",
        description="母亲的健康和生存状况、年龄、性格等"
    )
    paternal_grandparents: Optional[List[FamilyMember]] = Field(
        None,
        alias="祖父母(父系)",
        description="父系祖父母的生存与健康状况，如需详细记录可用列表"
    )
    maternal_grandparents: Optional[List[FamilyMember]] = Field(
        None,
        alias="祖父母(母系)",
        description="母系祖父母的生存与健康状况，如需详细记录可用列表"
    )

    # 兄弟姐妹信息
    siblings_health: Optional[str] = Field(
        None,
        alias="兄弟姐妹状况",
        description="简要说明兄弟姐妹数目、健康状况、性格特点等"
    )
    # 若需要更结构化记录，可改为:
    # siblings: Optional[List[FamilyMember]] = Field(None, ...)

    # 如果本人已中老年，子女信息也具备参考意义
    children: Optional[List[FamilyMember]] = Field(
        None,
        alias="子女状况",
        description="如有子女，则记录其健康、生存状态等"
    )

    # 遗传及疾病史
    is_consanguineous_marriage: bool = Field(
        ...,
        alias="近亲结婚史",
        description="是否存在近亲结婚(如父母为堂/表亲等)"
    )
    has_hereditary_disease: bool = Field(
        ...,
        alias="有遗传病史",
        description="家族中是否有明确的遗传性疾病史(如亨廷顿舞蹈病等)"
    )
    has_mental_illness: bool = Field(
        ...,
        alias="有精神疾病史",
        description="家族中是否存在精神疾病史"
    )
    diseases: Optional[List[Disease]] = Field(
        None,
        alias="家族疾病记录",
        description="对已知家族疾病(精神或其他重大疾病)的详细记录"
    )

    # 家庭整体情况
    family_harmony: Optional[str] = Field(
        None,
        alias="家庭和谐度",
        description="家庭成员相处是否融洽，冲突原因等"
    )
    economic_situation: Optional[str] = Field(
        None,
        alias="经济情况",
        description="家庭经济状况(富裕/普通/贫困等)"
    )
    living_conditions: Optional[str] = Field(
        None,
        alias="居住条件",
        description="居住环境、住房条件等"
    )
    neighbor_relationship: Optional[str] = Field(
        None,
        alias="邻里关系",
        description="与周围邻居相处状况"
    )
    special_family_traditions: Optional[str] = Field(
        None,
        alias="家庭特殊习惯或传统",
        description="家族或家庭中是否有特殊习惯、宗教信仰或传统"
    )
    notes: Optional[str] = Field(
        None,
        alias="备注",
        description="其他补充说明"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


def rewrite_family_history(text: str, feature_type: str) -> str:
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
                        "家族史是指家族情况与病情的关系,包括以下两个方面的情况:\n"
                        "(一) 家族遗传史\n"
                        "家族遗传史是指父母两系三代(祖代、父代、本人及兄弟姊妹,如果本人已中老年,则"
                        "子代也有参考意义)的精神疾病史,还包括特殊性格、酗酒、生活方式等情况。要说明是"
                        "否近亲结婚。如有阳性病例,还要说明病情、诊断、治疗情况、目前情况等。家族成员中"
                        "如有性格怪僻、长期不结婚、与他人关系不良等情况,也应记录。有时供史者会有隐瞒家"
                        "族遗传史的倾向,因此要仔细询问。家系图有助于形象体现上述内容。\n"
                        "( 二) 家庭情况\n"
                        "家庭情况包括家庭和谐情况、经济情况、居住条件、邻里关系、家庭特殊习惯或传统"
                        "等。还有各个家庭成员(包括不住在一起但往来密切,对患者家庭或本人影响较大者)的"
                        "年龄、职业、性格爱好、与患者的关系好坏等。家庭情况对儿童患者的病情常有很大影"
                        "响,对成人患者也有间接影响。了解这一点,有利于分析病因及症状,有利于安排出院后"
                        "的照顾,预防复发。"
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
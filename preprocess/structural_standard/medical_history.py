import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pydantic import BaseModel, Field
from textwrap import dedent
from typing import Optional, List
from openai import OpenAI

from load_config import CHAT_MODEL, API_KEY

class GeneralHealth(BaseModel):
    """
    一般健康状况
    与既往史中“总体健康状况”相关，可视项目需要进行简要记录
    """
    health_status: str = Field(
        ...,
        alias="健康状况",
        description="总体健康评估，如良好、一般、较差等"
    )
    last_checkup_date: Optional[str] = Field(
        None,
        alias="体检日期",
        description="最近一次体检或健康检查的日期"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class Disease(BaseModel):
    """
    疾病记录基本模型
    结合“既往史”要求，可记录有无精神疾病史、重大躯体疾病史等，
    若为精神疾病，可通过 episodes 字段详细记录历次发作情况。
    """
    disease_name: str = Field(
        ...,
        alias="疾病名称",
        description="疾病名称，如高血压、糖尿病、抑郁症、双相障碍等"
    )
    onset_date: Optional[str] = Field(
        None,
        alias="发病时间",
        description="疾病首次发病或确诊的大致时间"
    )
    current_status: str = Field(
        ...,
        alias="当前状态",
        description="当前疾病状态，如已缓解、治疗中、持续复发等"
    )
    treatment: Optional[str] = Field(
        None,
        alias="治疗情况",
        description="已采取的治疗方式，如药物治疗、手术、心理治疗等"
    )
    medication: Optional[List[str]] = Field(
        None,
        alias="用药情况",
        description="当前或曾用药物列表"
    )
    episodes: Optional[List[str]] = Field(
        None,
        alias="历次发作记录",
        description=(
            "若为精神疾病，可在此记录各次发作的时间、病期、主要症状、治疗经过及疗效等；"
            "若为躯体疾病也可选填重要复发信息。"
        )
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class ChronicDiseases(BaseModel):
    """
    慢性病史
    用于记录主要慢性躯体疾病，如高血压、糖尿病、心脏病等。
    其他重大躯体疾病也可放在 other_diseases 里。
    """
    hypertension: Optional[Disease] = Field(
        None,
        alias="高血压",
        description="高血压病史，如有则详细记录"
    )
    diabetes: Optional[Disease] = Field(
        None,
        alias="糖尿病",
        description="糖尿病病史，如有则详细记录"
    )
    heart_disease: Optional[Disease] = Field(
        None,
        alias="心脏病",
        description="心脏病病史，如有则详细记录"
    )
    other_diseases: Optional[List[Disease]] = Field(
        None,
        alias="其他疾病",
        description="其他已确诊的慢性或重大躯体疾病"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class InfectiousDiseases(BaseModel):
    """
    传染病史
    """
    hepatitis: Optional[Disease] = Field(
        None,
        alias="肝炎",
        description="是否患过或现患肝炎"
    )
    tuberculosis: Optional[Disease] = Field(
        None,
        alias="结核",
        description="是否患过或现患结核"
    )
    other_infectious: Optional[List[Disease]] = Field(
        None,
        alias="其他传染病",
        description="其他传染病史"
    )
    contact_history: Optional[List[str]] = Field(
        None,
        alias="接触史",
        description="与传染病患者或传染源的接触史"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class Surgery(BaseModel):
    """
    手术记录
    """
    surgery_name: str = Field(
        ...,
        alias="手术名称",
        description="手术名称"
    )
    surgery_date: str = Field(
        ...,
        alias="手术日期",
        description="手术实施日期"
    )
    recovery_status: str = Field(
        ...,
        alias="恢复情况",
        description="手术后康复或恢复情况"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class Trauma(BaseModel):
    """
    外伤记录
    """
    trauma_type: str = Field(
        ...,
        alias="外伤类型",
        description="外伤类型，如骨折、颅脑外伤等"
    )
    trauma_date: str = Field(
        ...,
        alias="受伤日期",
        description="受伤具体日期或时间范围"
    )
    treatment_status: str = Field(
        ...,
        alias="治疗状态",
        description="外伤后的治疗及康复情况"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class Allergy(BaseModel):
    """
    过敏史
    既往史中需特别关注“药物过敏”，以便安全用药。
    """
    drug_allergies: Optional[List[str]] = Field(
        None,
        alias="药物过敏",
        description="药物过敏列表，若无则留空"
    )
    food_allergies: Optional[List[str]] = Field(
        None,
        alias="食物过敏",
        description="食物过敏列表，若无则留空"
    )
    other_allergies: Optional[List[str]] = Field(
        None,
        alias="其他过敏",
        description="其他过敏情况，如花粉、金属等"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class MedicalHistory(BaseModel):
    """
    既往史完整数据模型
    根据第三部分要求：
    - 有无神经精神疾病史、有无重大躯体疾病史等；
    - 对完全缓解且缓解期较长的精神疾病发作应归入既往史；
    - 建议详细记录历次精神疾病的发作时间、症状、治疗及疗效；
    - 有无药物过敏史；
    """
    general_health: GeneralHealth = Field(
        ...,
        alias="一般健康状况",
        description="整体健康状态及最近体检情况"
    )
    chronic_diseases: ChronicDiseases = Field(
        ...,
        alias="慢性病史",
        description="主要慢性躯体疾病情况，如高血压、糖尿病、心脏病等"
    )
    infectious_diseases: InfectiousDiseases = Field(
        ...,
        alias="传染病史",
        description="既往或现有传染病史"
    )
    surgeries: Optional[List[Surgery]] = Field(
        None,
        alias="手术史",
        description="手术记录，如有多次手术则列出多个"
    )
    traumas: Optional[List[Trauma]] = Field(
        None,
        alias="外伤史",
        description="外伤记录，如车祸、摔伤、颅脑外伤等"
    )
    allergies: Optional[Allergy] = Field(
        None,
        alias="过敏史",
        description="药物及其他过敏信息"
    )
    blood_transfusion: Optional[bool] = Field(
        None,
        alias="输血史",
        description="是否曾经有过输血史"
    )
    vaccination_status: Optional[str] = Field(
        None,
        alias="疫苗接种史",
        description="疫苗接种情况"
    )

    # 新增：神经精神疾病史
    neuro_psychiatric_diseases: Optional[List[Disease]] = Field(
        None,
        alias="神经精神疾病史",
        description=(
            "用于记录既往神经或精神疾病，如抑郁症、双相情感障碍等。"
            "若有多次发作且完全缓解的，应在此记录历次发作的时间、病期、治疗及疗效；"
            "若尚未完全缓解或症状近期恶化，请在现病史中记录。"
        )
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        
        
def rewrite_medical_history(text: str, feature_type: str) -> str:
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
                        "既往史是指既往的健康状况,重点包括有无神经精神疾病史、有无重大躯体疾病史、"
                        "有无药物过敏史,必要时应对既往患病情况进行系统回顾。应注意这些疾病与精神障碍"
                        "之间在时间上有无相关性,是否存在因果关系。对于已经完全缓解而且缓解期较长的既"
                        "往精神疾病发作,应列入既往史,记录历次发作的时间、病期、主要症状、治疗经过及疗效"
                        "等;如果既往发作还没有完全缓解或近期症状恶化,则不算2次发作或复发,全部病情应"
                        "列入现病史之中。既往史资料对于治疗药物种类、剂量的选择和一些特殊治疗的应用具"
                        "有重要意义,应全面记录。\n"
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
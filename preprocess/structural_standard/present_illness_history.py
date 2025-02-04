import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pydantic import BaseModel, Field
from typing import Optional, List
from textwrap import dedent
from openai import OpenAI

from load_config import CHAT_MODEL, API_KEY


class Symptom(BaseModel):
    """
    症状描述
    主要对应要求说明中的“(一) 诱因或发病因素”部分，
    也可记录症状本身的名称、严重程度、持续时间等信息。
    """
    name: str = Field(..., alias="症状名称", description="具体症状的名称，如幻觉、妄想等")
    severity: Optional[str] = Field(None, alias="严重程度", description="症状的严重程度")
    frequency: Optional[str] = Field(None, alias="发生频率", description="症状出现的频率")
    duration: Optional[str] = Field(None, alias="持续时间", description="症状持续的时长")
    # 诱因或发病因素：根据要求说明 (一) 诱因或发病因素
    triggers: Optional[List[str]] = Field(
        None,
        alias="诱发因素",
        description="导致症状出现或加重的可能因素，如生活事件、生物因素、心理因素等"
    )
    # 保留“伴随症状”以便进一步描述病情
    associated_symptoms: Optional[List[str]] = Field(
        None,
        alias="伴随症状",
        description="同时出现的其他相关症状，如躯体不适、情绪低落等"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class Medication(BaseModel):
    """
    药物信息
    可作为既往或当前用药的详细说明。
    """
    药物名称: Optional[str] = Field(None, description="药物名称，如氯丙嗪、丙咪嗪等")
    剂量: Optional[str] = Field(None, description="药物使用剂量，如每日总剂量或每次剂量")
    用法: Optional[str] = Field(None, description="药物使用方法，如口服、肌注、静滴等")
    # 根据要求说明 (四) 既往诊疗情况提及“不良反应”，可适当添加
    不良反应: Optional[str] = Field(None, description="使用该药物时是否出现显著不良反应")

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class TreatmentHistory(BaseModel):
    """
    治疗史
    对应“(四) 既往诊疗情况”，重点在于：曾在哪就诊、诊断、用药、疗效、不良反应等
    """
    hospital: Optional[str] = Field(None, alias="就诊医院", description="曾就诊的医院名称")
    department: Optional[str] = Field(None, alias="就诊科室", description="曾就诊的科室")
    diagnosis: Optional[str] = Field(None, alias="诊断结果", description="既往诊断，如抑郁症、精神分裂症等")
    medications: List[Medication] = Field(
        default_factory=list,
        alias="用药情况",
        description="既往使用的药物及其详情，包括不良反应"
    )
    treatment_outcome: Optional[str] = Field(None, alias="治疗效果", description="治疗后的效果或预后评估")
    compliance: Optional[str] = Field(None, alias="治疗依从性", description="患者是否按医嘱规律用药、复诊等")

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class PhysiologicalStatus(BaseModel):
    """
    生理状态
    对应“(五) 一般情况”中的饮食、睡眠、大小便及躯体症状等部分。
    """
    appetite: Optional[str] = Field(None, alias="饮食情况", description="患者的饮食状况，如进食量、食欲变化等")
    sleep: Optional[str] = Field(None, alias="睡眠情况", description="患者的睡眠模式及质量")
    weight_change: Optional[str] = Field(None, alias="体重变化", description="近段时间有无明显体重变化")
    bowel_bladder: Optional[str] = Field(None, alias="大小便情况", description="大小便是否正常，有无失禁等")
    physical_symptoms: Optional[List[str]] = Field(
        None,
        alias="躯体症状",
        description="患者出现的其他躯体不适症状，如头痛、乏力、心慌等"
    )
    # 一般情况里也常关注自理能力，可酌情添加
    self_care_ability: Optional[str] = Field(None, alias="生活自理能力", description="日常起居、个人卫生等方面的独立程度")

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class PsychologicalStatus(BaseModel):
    """
    心理状态
    对应“(五) 一般情况”中对患者主观情绪、思维、行为、认知情况的描述，
    也可结合(三) 病情演变，描述当前状态。
    """
    mood: Optional[str] = Field(None, alias="情绪状态", description="患者目前的情绪表现，如抑郁、焦虑、欣快等")
    anxiety_level: Optional[str] = Field(None, alias="焦虑程度", description="焦虑的主观或客观评估")
    thought_content: Optional[List[str]] = Field(
        None,
        alias="思维内容",
        description="主要思维主题，如妄想、罪恶感、夸大等"
    )
    behavior_changes: Optional[List[str]] = Field(
        None,
        alias="行为改变",
        description="近期在社交、活动等方面的明显改变"
    )
    cognitive_symptoms: Optional[List[str]] = Field(
        None,
        alias="认知症状",
        description="注意力、记忆力、理解判断等方面有无异常"
    )
    # 从要求说明 (五) 强调患者对疾病的认识及社会功能
    insight: Optional[str] = Field(None, alias="自知力", description="患者对自身疾病或症状的认识程度")
    social_adaptation: Optional[str] = Field(None, alias="社会适应情况", description="工作、学习、家庭角色、人际交往等社会功能表现")

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class RiskAssessment(BaseModel):
    """
    风险评估
    对应“(五) 一般情况”中对威胁自身或他人安全的危险行为的重点询问
    """
    suicidal_ideation: bool = Field(..., alias="自伤自杀意念", description="是否存在自杀或自伤想法")
    aggressive_behavior: bool = Field(..., alias="伤人行为", description="是否存在攻击或伤害他人的行为")
    impulsivity: bool = Field(..., alias="冲动行为", description="是否存在明显的冲动倾向，如冲动打人等")
    wandering: bool = Field(..., alias="外走行为", description="是否有走失或离家不归等情况")
    risk_factors: Optional[List[str]] = Field(None, alias="危险因素", description="其他可能存在的危险因素")
    # 根据要求说明中提到“毁物”等
    property_damage: bool = Field(False, alias="毁物行为", description="是否存在毁坏物品等破坏性行为")

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class TimelineEvent(BaseModel):
    """
    时间线事件
    对应“(三) 病情演变”，用时间顺序记录病情或事件演变。
    """
    date: Optional[str] = Field(None, alias="发生时间", description="事件发生的具体时间或阶段")
    event_type: str = Field(..., alias="事件类型", description="可用简要分类，如“症状加重”“冲动伤人”“社会功能下降”等")
    description: str = Field(..., alias="事件描述", description="具体发生了什么，以及对患者病情的影响")
    impact: Optional[str] = Field(None, alias="影响程度", description="对病情或患者生活造成的影响大小")

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class PresentIllnessHistory(BaseModel):
    """
    现病史
    结合要求说明的五个方面 (1) 诱因或发病因素 (2) 发病形式 (3) 病情演变 (4) 既往诊疗情况 (5) 一般情况
    """
    # (二) 发病形式
    onset_time: Optional[str] = Field(None, alias="发病时间", description="疾病开始的时间，或患者/家属首次注意到异常的时间")
    onset_type: Optional[str] = Field(
        None,
        alias="发病方式",
        description="急性、亚急性或缓慢起病，可填写如“1个月内出现大部分症状”等"
    )
    # (一) 主要症状（内含诱因字段），结合 Symptom 类
    presenting_symptoms: List[Symptom] = Field(
        default_factory=list,
        alias="主要症状",
        description="当前最突出的症状列表，包含可能的诱发因素"
    )
    # (三) 病情演变 - 使用时间线记录
    timeline: List[TimelineEvent] = Field(
        default_factory=list,
        alias="病程发展",
        description="按时间顺序记录病情变化或重大事件"
    )
    # (四) 既往诊疗情况
    treatment_history: List[TreatmentHistory] = Field(
        default_factory=list,
        alias="治疗经过",
        description="既往曾接受的诊疗史，如住院、门诊、服药情况等"
    )
    # (五) 一般情况
    physiological_status: PhysiologicalStatus = Field(
        ...,
        alias="生理状态",
        description="包含饮食、睡眠、大小便、躯体症状以及自理能力等"
    )
    psychological_status: PsychologicalStatus = Field(
        ...,
        alias="心理状态",
        description="包含情绪、思维、认知、社会功能及对疾病的认识等"
    )
    risk_assessment: RiskAssessment = Field(
        ...,
        alias="风险评估",
        description="是否存在自杀、自伤、伤人、毁物等威胁安全的行为"
    )

    # 可根据需要保留或删除
    admission_diagnosis: Optional[str] = Field(None, alias="入院诊断", description="本次入院时的初步诊断(若有)")
    current_medications: Optional[List[Medication]] = Field(
        None,
        alias="当前用药及剂量",
        description="目前正在使用的药物、剂量及可能的不良反应"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


def rewrite_present_illness_history(text: str, feature_type: str) -> str:
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
                        "现病史是病史的主要部分,主要包括以下几个方面:\n"
                        "(一) 诱因或发病因素\n"
                        "询问患者发病的环境背景及与患者有关的生物、心理、社会因素,以了解患者在什么"
                        "情况下发病。如有明确的原因和(或)诱因,则应详细、客观地进行描述。需强调的是,疾"
                        "病初发前发生的事件,不一定构成疾病的病因,一方面是因为部分精神障碍起病缓慢,疾"
                        "病的首发时间并不一定清晰,很多时候并不能确定生活事件与疾病发生之间的因果关"
                        "联; 另一方面,许多精神疾病的发生、发展来自多因素。有些生活事件是致病因素,如"
                        "脑外伤导致患者出现明显的认知功能损害甚至精神病性症状,中毒以后出现明显幻觉妄"
                        "想。而有些生活事件可能仅仅是诱因,如高考失利、失恋、领导批评等。当供史者提到这"
                        "些原因或诱因时,必须问明细节,认真分析,客观地理解生活事件与精神疾病之间的内在"
                        "联系,切忌轻易下结论。"
                        "( 二) 发病形式\n"
                        "精神疾病可以有不同起病形式,如急性、亚急性或缓慢起病。精神疾病中的时间标"
                        "准与内、外科疾病相比要宽松一些。"
                        "一般而言,急性起病是指有明显的起病界限, 1个月内起病;"
                        "亚急性是指3个月内起病;缓慢发病通常无明显的起病界限,多为半年以上。如"
                        "精神分裂症是一种严重的精神疾病,既可以急性起病, 1个月内出现大量的精神病性症"
                        "状,也可以缓慢起病,数年之内逐渐出现症状,家属甚至无法说清患者的起病时间。"
                        "( 三) 病情演变\n"
                        "病情演变主要包括发病症状的变化和轻重程度的变化。例如前述的性格改变,就要"
                        "说明过去怎样,现在怎样,改变程度如何。精神症状的特征之一就是精神活动在原有的"
                        "基础上发生重大的改变,如果原来的基础是“常态”,那么发生重大改变就是“异常”或"
                        "“失常”。除了说明症状的具体表现外,还要说明产生症状的背景及与症状变化有关的因素。"
                        "例如,精神疾病患者有时可有冲动打人行为,也要搞清在什么情况下打人,是针对具体"
                        "对象还是不分青红皂白? 再如,抑郁症的抑郁情绪有晨重晚轻的倾向,青年妇女的精"
                        "神状态有时与月经周期有关。这些例子都说明症状与其他因素(这里是特指时间因"
                        "素)的关系。搞清这些关系,对明确诊断、预测病情、制订治疗方案都有很大的作用。"
                        "对病情的描述自始至终要有时间顺序,先后逐年、逐月甚或逐日地分段做纵向描述。"
                        "病程长者,可重点对其近1年的情况进行详细了解,每一阶段都要有该阶段社会功能"
                        "情况(如职业角色、人际交往等方面)的描述,有助于判断是否处于疾病发病期或疾病"
                        "严重程度变化。"
                        "( 四) 既往诊疗情况\n"
                        "了解患者的既往诊疗情况对制订治疗方案有十分重要的参考价值。因此,对每位就"
                        "诊的患者均应详细询问过去的诊疗情况,包括曾在何处就诊、诊断结果、用药情况(药物"
                        "名称、剂型和剂量、用药时间)、疗效及不良反应等。患者既往治疗的病历、检查报告单、"
                        "药品等也具有一定价值,应注意全面收集资料。"
                        "( 五) 一般情况\n"
                        "主要包括患者患病期间的工作、学习和社会适应情况,与周围环境的接触情况,对疾"
                        "病的认识程度,饮食、睡眠、大小便及生活自理能力等。还应重点询问有无威胁自身或他"
                        "人安全的危险行为,如自杀、自伤、冲动、伤人、毁物、外走等,做到心中有数,重点防范。"
                        "这些资料不仅能反映疾病的严重程度,还可为疾病的诊断、治疗和护理计划的制订提供"
                        "参考。\n"
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
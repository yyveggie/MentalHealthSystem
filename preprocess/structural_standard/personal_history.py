import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pydantic import BaseModel, Field
from openai import OpenAI
from textwrap import dedent
from typing import Optional, List

from load_config import CHAT_MODEL, API_KEY


class MotherPregnancyBirth(BaseModel):
    """
    母亲怀孕及本人出生情况
    包括是否意外怀孕、孕期健康状况、分娩顺利与否等。
    """
    unexpected_pregnancy: bool = Field(
        ...,
        alias="是否意外怀孕",
        description="母亲怀孕是否在计划之外，如是则记录为 True"
    )
    severe_illness_during_pregnancy: Optional[str] = Field(
        None,
        alias="孕期严重疾病",
        description="孕期是否出现严重疾病，如有则填写疾病名称及简要情况"
    )
    radiation_exposure: bool = Field(
        ...,
        alias="放射线照射史",
        description="孕期是否受到放射线照射"
    )
    threatened_abortion: bool = Field(
        ...,
        alias="保胎情况",
        description="是否有先兆流产而需保胎"
    )
    morning_sickness: Optional[str] = Field(
        None,
        alias="妊娠反应",
        description="母亲的妊娠反应程度，如轻微、中度、严重等"
    )
    delivery_process: Optional[str] = Field(
        None,
        alias="分娩过程",
        description="分娩是否顺利，如难产、剖宫产等"
    )
    premature_or_low_weight: bool = Field(
        ...,
        alias="早产或低体重",
        description="是否早产或出生时体重过低"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class EarlyDevelopment(BaseModel):
    """
    早期发育及健康状况
    从出生到入小学前的发育情况，如喂养方式、言语运动发育、重大疾病等。
    """
    feeding_method: Optional[str] = Field(
        None,
        alias="喂养方式",
        description="母乳喂养、人工喂养或混合喂养"
    )
    primary_caregiver: Optional[str] = Field(
        None,
        alias="主要照料者",
        description="幼年时主要由谁照顾，母亲/祖母/保姆等"
    )
    speech_motor_development: Optional[str] = Field(
        None,
        alias="言语运动发育",
        description="言语、走路等发育里程碑是否按时或延迟"
    )
    bowel_bladder_control_age: Optional[str] = Field(
        None,
        alias="大小便控制年龄",
        description="何时学会自主控制大小便"
    )
    major_childhood_illnesses: Optional[List[str]] = Field(
        None,
        alias="幼年重大疾病",
        description="幼儿期患过的重大疾病(尤其是中枢神经系统疾病)"
    )
    preschool_experience: Optional[str] = Field(
        None,
        alias="幼儿园经历",
        description="是否上过幼儿园，适应情况如何"
    )
    sleep_habits: Optional[str] = Field(
        None,
        alias="睡眠习惯",
        description="婴幼儿时期的睡眠规律、夜醒、是否挑灯夜哭等"
    )
    childhood_temper: Optional[str] = Field(
        None,
        alias="幼年脾气特点",
        description="幼年时是否有特殊脾气，如易怒、胆怯、依赖性强等"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class EducationHistory(BaseModel):
    """
    教育情况
    对成年人重点了解学历、学习成绩；对儿童青少年则还需了解师生关系、爱好等。
    """
    highest_education: Optional[str] = Field(
        None,
        alias="最高学历",
        description="如小学/初中/高中/大学/研究生等"
    )
    academic_performance: Optional[str] = Field(
        None,
        alias="学习成绩",
        description="整体学习成绩状况，如优良中差"
    )
    teacher_student_relationship: Optional[str] = Field(
        None,
        alias="师生关系",
        description="与老师相处情况，是否存在明显冲突或特别亲近"
    )
    peer_relationship: Optional[str] = Field(
        None,
        alias="同学关系",
        description="与同学相处状况，有无要好的同学或社交困难"
    )
    favorite_subjects: Optional[List[str]] = Field(
        None,
        alias="偏爱科目",
        description="喜欢的学科或擅长领域"
    )
    extra_curricular_activities: Optional[List[str]] = Field(
        None,
        alias="课外活动",
        description="参与的课外或校外兴趣活动情况"
    )
    disciplinary_issues: Optional[str] = Field(
        None,
        alias="违纪情况",
        description="如逃学、打架斗殴等违纪行为"
    )
    reason_for_no_schooling: Optional[str] = Field(
        None,
        alias="未上学原因",
        description="若在学龄期没有上学，原因为何(经济、疾病、家庭因素等)"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class WorkHistory(BaseModel):
    """
    工作情况
    包括工作经历、同事关系、是否频繁调换工作等。
    """
    current_position: Optional[str] = Field(
        None,
        alias="当前工种",
        description="目前从事的工作或职务"
    )
    work_performance: Optional[str] = Field(
        None,
        alias="工作表现",
        description="对工作的适应度、完成质量等"
    )
    relationship_with_colleagues: Optional[str] = Field(
        None,
        alias="同事关系",
        description="与同事相处情况、是否有冲突"
    )
    promotions_or_changes: Optional[str] = Field(
        None,
        alias="升迁或调动",
        description="工作岗位是否经常变动，变动原因"
    )
    job_satisfaction: Optional[str] = Field(
        None,
        alias="工作满意度",
        description="对目前工作的态度，满意/不满/经常抱怨等"
    )
    labor_discipline_violations: Optional[str] = Field(
        None,
        alias="违纪或违法情况",
        description="是否经常迟到早退、违纪或有违法行为"
    )
    military_service_experience: Optional[str] = Field(
        None,
        alias="参军经历",
        description="是否曾参军，如果中途退役需说明原因"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class MarriageFamilyStatus(BaseModel):
    """
    婚恋经历和家庭状况
    包括恋爱史、婚姻状态、孩子、家庭关系等。
    """
    marital_status: Optional[str] = Field(
        None,
        alias="婚姻状况",
        description="未婚、已婚、离异、丧偶等"
    )
    love_experience: Optional[str] = Field(
        None,
        alias="恋爱经历",
        description="有无恋爱史，态度如何，若有挫折如何处理"
    )
    marriage_age: Optional[str] = Field(
        None,
        alias="结婚年龄",
        description="首次结婚的年龄"
    )
    children_info: Optional[str] = Field(
        None,
        alias="子女情况",
        description="子女数量及出生年月，子女关系"
    )
    spouse_relationship: Optional[str] = Field(
        None,
        alias="夫妻关系",
        description="夫妻感情和谐度，有无重大冲突"
    )
    family_economics: Optional[str] = Field(
        None,
        alias="家庭经济情况",
        description="家庭经济支配、家务分工、主要经济来源等"
    )
    sexual_life: Optional[str] = Field(
        None,
        alias="性生活情况",
        description="如无特殊，可简要记录；若有问题需说明"
    )
    divorce_history: Optional[str] = Field(
        None,
        alias="离婚或再婚情况",
        description="若有离异史，说明原因；再婚情况；前婚子女等"
    )
    menstrual_history: Optional[str] = Field(
        None,
        alias="月经史",
        description="女性患者的月经初潮年龄、周期规律、绝经期等"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class PremorbidPersonality(BaseModel):
    """
    病前性格
    了解患者在患病前的性格特征，对比当前症状是否异常。
    """
    baseline_mood: Optional[str] = Field(
        None,
        alias="基本心境",
        description="患者平时给人最明显的情绪基调，如乐观、悲观、暴躁等"
    )
    interpersonal_style: Optional[str] = Field(
        None,
        alias="人际交往方式",
        description="社交态度，如是否善于结交朋友、回避社交、对人冷漠等"
    )
    lifestyle_habits: Optional[str] = Field(
        None,
        alias="生活习惯",
        description="饮食、作息、爱整洁/邋遢等方面的习惯"
    )
    values_beliefs: Optional[str] = Field(
        None,
        alias="价值标准",
        description="个人道德、信念等"
    )
    self_assessment: Optional[str] = Field(
        None,
        alias="自我评价",
        description="患者对自身性格或行为方式的认识(如过度谦卑、过度自负等)"
    )
    others_opinions: Optional[str] = Field(
        None,
        alias="他人评价",
        description="周围亲友对患者性格的主要看法"
    )
    major_strengths_weaknesses: Optional[str] = Field(
        None,
        alias="主要优缺点",
        description="性格中的积极面和消极面"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class PersonalHistory(BaseModel):
    """
    个人史：综合上述各部分信息，描绘出患者从小到现今的生活经历、性格特点等
    """
    mother_pregnancy_birth: Optional[MotherPregnancyBirth] = Field(
        None,
        alias="母亲怀孕及出生情况",
        description="(一) 母亲怀孕及本人出生情况，若与现病无关可简略"
    )
    early_development: Optional[EarlyDevelopment] = Field(
        None,
        alias="早期发育及健康状况",
        description="(二) 从出生到入学前的发育、健康及养育情况"
    )
    education_history: Optional[EducationHistory] = Field(
        None,
        alias="教育情况",
        description="(三) 学历、学习表现、同学关系等"
    )
    work_history: Optional[WorkHistory] = Field(
        None,
        alias="工作情况",
        description="(四) 工作经历、同事关系、工种满意度等"
    )
    marriage_family_status: Optional[MarriageFamilyStatus] = Field(
        None,
        alias="婚恋和家庭状况",
        description="(五) 恋爱史、婚姻状况、子女及家庭关系等"
    )
    premorbid_personality: Optional[PremorbidPersonality] = Field(
        None,
        alias="病前性格",
        description="(六) 患病前的基本性格特征"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


def rewrite_personal_history(text: str, feature_type: str) -> str:
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
                        "个人史是指患者从小到现在的生活经历、性格特点。理想的精神科个人史,是希望"
                        "像小说一样描绘出患者过去的形象来。虽然实际上做不到这一点,但在重点的地方仍需"
                        "要有一些具体的例子来反映患者的具体侧面。对一个首次就诊的患者,个人史要求项目"
                        "齐全,但重点项目要详细、具体。所谓重点,是指与现病史关系比较密切的部分。对儿童"
                        "患者来说,婴幼儿期的生长发育、父母养育方式和家庭环境等就是重点;对中老年发病者"
                        "来说,职业经历和家庭关系、人际交往等也是重点。\n"
                        "(一) 母亲怀孕及本人出生情况\n"
                        "母亲怀孕及本人出生情况对儿童患者及精神发育迟滞患者很重要,包括:是否意外"
                        "怀孕,孕期有无严重疾病,有否受放射线照射,有否因流产倾向而保胎,有无严重的妊娠"
                        "反应。关于分娩情况,特别要说明分娩是否顺利,如果是难产或有其他合并症,应说明。"
                        "如系早产儿或低体重儿,也应说明。这些因素对新生儿的脑部发育都可能有影响。\n"
                        "( 二) 早期发育及健康状况\n"
                        "早期发育及健康状况主要是指从出生到入小学这一阶段的情况,包括:母乳喂养还"
                        "是人工喂养,母亲抚养还是他人代养,幼年时成长是否顺利及成长环境,言语运动发育情"
                        "况,大小便控制情况,饮食习惯,睡眠习惯,有无特殊脾气,患过什么重大疾病(特别是中"
                        "枢神经系统疾病),幼儿园经历等。\n"
                        "怀孕、分娩及早期情况对儿童患者特别重要。这些病史往往需要家长提供。家长中"
                        "母亲和祖母提供的情况不一定一致,例如一个说“她根本不管小孩“,一个说“小孩都给她宠坏啦”。"
                        "碰到这种情况,往往双方都有一些事实,医生可以根据病情来判断,而不必强"
                        "求她们的意见统一。对中老年发病的患者,如果病情与儿童期情况无关,上述病史不需"
                        "仔细询问,只要简单地了解一下幼时有无特殊情况即可。\n"
                        "( 三) 教育情况\n"
                        "对成年患者,主要是了解学历及学习成绩。对儿童及青少年患者,则要了解师生关"
                        "系、同学关系,有没有几个感情特别好的同学,这些“好朋友”的品行表现如何,患者本人"
                        "的学习成绩,所爱好的学科,参加课外或校外活动情况,课余爱好,在校期间有无违纪逃"
                        "学等。如果是中小学生,最好能了解老师的评语。还要了解学龄期在家里的表现。如果"
                        "在学龄期没有上学,要甄别和明确原因。\n"
                        "( 四) 工作情况\n"
                        "工作情况包括工作表现、同事关系、升迁情况及目前工种,对工作岗位是否满意,是"
                        "否经常存在违反劳动纪律或违法情况。如果经常调换工作岗位或单位,是何原因? 调换"
                        "工作的原因不外两方面:一是客观需要,另一是主观不能适应。主观不能适应有时与性"
                        "格有关,有时则是疾病发展使然,如果工作越调动越趋于简单,更说明有问题。如参军而"
                        "提前退役,也要说明原因。\n"
                        "( 五) 婚恋经历和家庭状况\n"
                        "未婚者经历主要包括有无恋爱史、恋爱的基本态度、恋爱中遭受挫折的原因和处理"
                        "的方式。已婚者经历包括结婚年龄、孩子出生年月、夫妻感情、家庭经济支配、家务分工、"
                        "性生活情况等。配偶的简单情况可在家族史中介绍,此外要补充是自由恋爱结婚的还是"
                        "其他,夫妻关系有无大的冲突,如有需说明原因。如果本次是再婚,还要说明以前婚姻情"
                        "况、离婚原因、前婚子女的情况等。女性患者的月经史包括初潮年龄、月经规律、经量、月"
                        "经期症状或不适感、末次月经日期、绝经期等。如果有周期性发作的症状,要明确这个周"
                        "期性是否与月经周期有关。\n"
                        "( 六) 病前性格\n"
                        "对成年患者来说,病前性格是个人史中最重要的部分,因为当前症状的性质和严重"
                        "程度,都是与病前性格对比方能确定。例如,原来是沉默寡言的,现在话多了是异常;原"
                        "来话多活跃的,现在沉默了是异常。又如,躁狂症患者经过治疗后,其兴奋多语症状已控"
                        "制到一般水平,但如果其病前性格是话很少(比一般人少),那么这个患者的躁狂发作仍"
                        "不能判断为完全缓解。性格可从3个方面了解:一是从其亲戚朋友,二是根据患者自诉,三是检查者自己观"
                        "察。有些人的性格表现比较一致,有些人则在不同场合有不同的表现。周围人所反映的"
                        "有时只是性格的一方面表现,患者对自己的估价也常不能恰如其分(抑郁患者可以把自"
                        "己贬得太低,病态人格者常常掩饰自己的缺点)。在病史采集阶段,了解他人对患者的行"
                        "为方式的描述以及患者对其他人行为方式的描述均有助于判断患者的性格特征。实际"
                        "上,大多数正常人也常不能恰当地认识自己,常常是对别人比对自己看得清楚。因此,了"
                        "解一个人的性格,只能综合多方面的观察方能比较全面。性格可表现在许多方面,在个人史"
                        "的其他项目中也可以反映出一部分,但人际关系、生活习惯、基本心境、价值标准等方面最"
                        "突出地表现出一个人的性格。大多数人的性格都有积极的一面和消极的一面,问病史时"
                        "都有积极的一面和消极的一面,问病史时不能只着重消极的一面。"
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
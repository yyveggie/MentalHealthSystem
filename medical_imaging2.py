import os
import base64
from PIL import Image
from openai import OpenAI
import streamlit as st
from phi.tools.duckduckgo import DuckDuckGo

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = None

with st.sidebar:
    st.title("⚙️ 系统配置")
    
    if not st.session_state.OPENAI_API_KEY:
        api_key = st.text_input(
            "请输入您的 OpenAI API 密钥：",
            type="password"
        )
        st.caption(
            "从 [OpenAI 平台]获取您的 API 密钥"
            "(https://platform.openai.com/api-keys) 🔑"
        )
        if api_key:
            st.session_state.OPENAI_API_KEY = api_key
            st.success("API 密钥保存成功！")
            st.rerun()
    else:
        st.success("API 密钥已配置")
        if st.button("🔄 重置 API 密钥"):
            st.session_state.OPENAI_API_KEY = None
            st.rerun()
    
    st.info(
        "本工具使用先进的计算机视觉和放射学专业知识，"
        "提供 AI 驱动的医学影像分析。"
    )
    st.warning(
        "⚠警告：本工具仅用于教育和信息参考目的。"
        "所有分析结果都应由合格的医疗专业人员审核。"
        "请勿仅基于此分析做出医疗决定。"
    )

# Initialize OpenAI client
client = OpenAI(api_key=st.session_state.OPENAI_API_KEY) if st.session_state.OPENAI_API_KEY else None

# 确保查询文本使用正确的编码
def ensure_unicode(text):
    if isinstance(text, bytes):
        return text.decode('utf-8')
    return str(text)

# Function to analyze image with OpenAI
def analyze_with_openai(image_path, query):
    try:
        with open(image_path, "rb") as image_file:
            # 确保使用 utf-8 编码处理文本
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query.encode('utf-8').decode('utf-8')},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096
            )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI API 错误: {e}")

# Medical Analysis Query
query = """
您是一位经验丰富的医学影像专家，在放射学和诊断成像方面具有广泛的专业知识。请按以下结构分析患者的医学影像：

### 1. 图像类型和区域
- 指明成像方式（X射线/核磁共振/CT/超声等）
- 确定患者的解剖区域和体位
- 评价图像质量和技术适当性

### 2. 关键发现
- 系统列出主要观察结果
- 详细描述患者影像中的任何异常情况
- 包括相关的测量数据和密度信息
- 描述位置、大小、形状和特征
- 评估严重程度：正常/轻度/中度/重度

### 3. 诊断评估
- 提供主要诊断及其可信度
- 按可能性顺序列出鉴别诊断
- 用患者影像中观察到的证据支持每个诊断
- 注明任何关键或紧急发现

### 4. 患者友好解释
- 用简单、清晰的语言向患者解释发现
- 避免医学术语或提供清晰的定义
- 适当使用视觉类比帮助理解
- 解答与这些发现相关的常见患者疑虑

### 5. 研究背景
- 包含相关病例的医学文献
- 提及标准治疗方案
- 说明该领域的技术进展
- 提供2-3个支持分析的关键医学参考文献

请使用清晰的markdown标题和项目符号格式化您的回答。力求简明但全面。
"""

st.title("🏥 医学影像诊断助手")
st.write("上传医学影像进行专业分析")

# Create containers for better organization
upload_container = st.container()
image_container = st.container()
analysis_container = st.container()

with upload_container:
    uploaded_file = st.file_uploader(
        "上传医学影像",
        type=["jpg", "jpeg", "png", "dicom"],
        help="支持的格式：JPG、JPEG、PNG、DICOM"
    )

if uploaded_file is not None:
    with image_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file)
            width, height = image.size
            aspect_ratio = width / height
            new_width = 500
            new_height = int(new_width / aspect_ratio)
            resized_image = image.resize((new_width, new_height))
            
            st.image(
                resized_image,
                caption="已上传的医学影像",
                use_container_width=True
            )
            
            analyze_button = st.button(
                "🔍 开始分析",
                type="primary",
                use_container_width=True
            )
    
    with analysis_container:
        if analyze_button:
            image_path = "temp_medical_image.png"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("🔄 正在分析图像...请稍候"):
                try:
                    if client:
                        response = analyze_with_openai(image_path, ensure_unicode(query))
                        st.markdown("### 📋 分析结果")
                        st.markdown("---")
                        st.markdown(response)
                        st.markdown("---")
                        st.caption(
                            "注意：此分析结果由 AI 生成，"
                            "应由合格的医疗专业人员审核。"
                        )
                    else:
                        st.warning("请在侧边栏配置您的 API 密钥以继续")
                except Exception as e:
                    st.error(f"分析错误：{e}")
                finally:
                    if os.path.exists(image_path):
                        os.remove(image_path)
else:
    st.info("👆 请上传医学影像开始分析")
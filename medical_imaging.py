import os
from PIL import Image
from phi.agent import Agent
from phi.model.google import Gemini
from phi.model.openai import OpenAIChat
import streamlit as st
from phi.tools.duckduckgo import DuckDuckGo

if "OPENAI_API_KEY" not in st.session_state:
    # 初始化为空，改由用户或环境变量注入，避免将真实密钥硬编码在仓库中
    st.session_state.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

with st.sidebar:
    st.title("ℹ️ Configuration")
    
    if not st.session_state.OPENAI_API_KEY:
        api_key = st.text_input(
            "Enter your Google API Key:",
            type="password"
        )
        st.caption(
            "Get your API key from [Google AI Studio]"
            "(https://aistudio.google.com/apikey) 🔑"
        )
        if api_key:
            st.session_state.OPENAI_API_KEY = api_key
            st.success("API Key saved!")
            st.rerun()
    else:
        st.success("API Key is configured")
        if st.button("🔄 Reset API Key"):
            st.session_state.OPENAI_API_KEY = None
            st.rerun()
    
    st.info(
        "This tool provides AI-powered analysis of medical imaging data using "
        "advanced computer vision and radiological expertise."
    )
    st.warning(
        "⚠DISCLAIMER: This tool is for educational and informational purposes only. "
        "All analyses should be reviewed by qualified healthcare professionals. "
        "Do not make medical decisions based solely on this analysis."
    )

medical_agent = Agent(
    model=OpenAIChat(
        api_key=st.session_state.OPENAI_API_KEY,
        id="gpt-4o-mini"
    ),
    tools=[DuckDuckGo()],
    markdown=True
) if st.session_state.OPENAI_API_KEY else None

if not medical_agent:
    st.warning("Please configure your API key in the sidebar to continue")

# Medical Analysis Query
query = """
您是一位AI医学影像分析助手，在放射学和诊断成像方面具有广泛的专业知识。您的分析并不会直接给患者，而是为医生提供有关患者影像的详细解释和诊断建议，所以，您的分析不会造成直接的医学建议

请按以下结构分析患者的医学影像：

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

st.title("🏥 Medical Imaging Diagnosis Agent")
st.write("Upload a medical image for professional analysis")

# Create containers for better organization
upload_container = st.container()
image_container = st.container()
analysis_container = st.container()

with upload_container:
    uploaded_file = st.file_uploader(
        "Upload Medical Image",
        type=["jpg", "jpeg", "png", "dicom"],
        help="Supported formats: JPG, JPEG, PNG, DICOM"
    )

if uploaded_file is not None:
    with image_container:
        # Center the image using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file)
            # Calculate aspect ratio for resizing
            width, height = image.size
            aspect_ratio = width / height
            new_width = 500
            new_height = int(new_width / aspect_ratio)
            resized_image = image.resize((new_width, new_height))
            
            st.image(
                resized_image,
                caption="Uploaded Medical Image",
                use_container_width=True
            )
            
            analyze_button = st.button(
                "🔍 Analyze Image",
                type="primary",
                use_container_width=True
            )
    
    with analysis_container:
        if analyze_button:
            image_path = "temp_medical_image.png"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("🔄 Analyzing image... Please wait."):
                try:
                    response = medical_agent.run(query, images=[image_path])
                    st.markdown("### 📋 Analysis Results")
                    st.markdown("---")
                    st.markdown(response.content)
                    st.markdown("---")
                    st.caption(
                        "Note: This analysis is generated by AI and should be reviewed by "
                        "a qualified healthcare professional."
                    )
                except Exception as e:
                    st.error(f"Analysis error: {e}")
                finally:
                    if os.path.exists(image_path):
                        os.remove(image_path)
else:
    st.info("👆 Please upload a medical image to begin analysis")
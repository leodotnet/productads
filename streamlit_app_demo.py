"""
Streamlit Demo — 直接调用 API 生图，不依赖数据库/Redis/Celery
只需要配置 .env 中的 API Key 即可运行：
    streamlit run streamlit_app_demo.py
"""

import io
import os
import base64
from dotenv import load_dotenv

import streamlit as st
from PIL import Image

load_dotenv()

st.set_page_config(
    page_title="电商产品图片生成 Demo",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ 电商产品图片生成 Demo")
st.markdown("上传产品图片，直接调用 AI 模型生成电商场景图（无需后端服务）")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ 配置")

    # Model selection
    model_options = {
        "Gemini 2.5 Flash Image": "gemini_flash",
        "OpenAI GPT-Image-2": "openai_gpt_image",
    }
    selected_display = st.selectbox("选择生图模型", list(model_options.keys()))
    selected_model = model_options[selected_display]

    # Check API key status
    if selected_model == "gemini_flash":
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            st.error("❌ 未配置 GEMINI_API_KEY")
        else:
            st.success("✅ Gemini API Key 已配置")
    elif selected_model == "openai_gpt_image":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            st.error("❌ 未配置 OPENAI_API_KEY")
        else:
            st.success("✅ OpenAI API Key 已配置")

    st.divider()
    num_images = st.slider("生成数量", 1, 4, 2)
    size = st.selectbox("图片尺寸", ["1024x1024", "1536x1024", "1024x1536"])


# --- Generation Functions ---

def generate_gemini(prompt: str, input_images: list[bytes], num: int) -> list[bytes]:
    """调用 Gemini 2.5 Flash Image 生图"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    results = []

    for i in range(num):
        contents = []
        # 添加输入图片
        for img_bytes in input_images:
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
        # 添加文字 prompt
        contents.append(prompt)

        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )

        # 提取图片
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                    results.append(part.inline_data.data)
                    break

    return results


def generate_openai(prompt: str, input_images: list[bytes], num: int, size: str) -> list[bytes]:
    """调用 OpenAI GPT-Image-2 生图"""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    results = []

    # 分批生成（每批最多4张）
    remaining = num
    while remaining > 0:
        batch = min(remaining, 4)
        kwargs = {
            "model": "gpt-image-2",
            "prompt": prompt,
            "n": batch,
            "size": size,
        }
        # 传入参考图片
        if input_images:
            kwargs["image"] = [io.BytesIO(img) for img in input_images]

        response = client.images.generate(**kwargs)

        for img_data in response.data:
            image_bytes = base64.b64decode(img_data.b64_json)
            results.append(image_bytes)

        remaining -= batch

    return results


# --- Main UI ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 上传产品图片")
    uploaded_files = st.file_uploader(
        "选择产品图片（支持 PNG/JPG/WebP）",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.markdown(f"已上传 **{len(uploaded_files)}** 张图片：")
        cols = st.columns(min(len(uploaded_files), 4))
        for i, f in enumerate(uploaded_files):
            with cols[i % 4]:
                img = Image.open(f)
                st.image(img, caption=f.name, width=150)
                f.seek(0)

    st.divider()
    st.header("✍️ 描述需求")
    prompt = st.text_area(
        "输入生图提示词",
        placeholder="例如：专业电商产品图，模特穿着这双白色运动鞋，站在城市街头，自然光照，时尚摄影风格",
        height=120,
    )

    # Generate button
    can_generate = bool(prompt) and bool(api_key)
    if st.button("🚀 开始生成", type="primary", disabled=not can_generate, use_container_width=True):
        # Read uploaded images
        input_bytes = []
        if uploaded_files:
            for f in uploaded_files:
                input_bytes.append(f.getvalue())

        with col2:
            st.header("🖼️ 生成结果")
            with st.spinner(f"正在生成 {num_images} 张图片，请稍候..."):
                try:
                    if selected_model == "gemini_flash":
                        generated = generate_gemini(prompt, input_bytes, num_images)
                    elif selected_model == "openai_gpt_image":
                        generated = generate_openai(prompt, input_bytes, num_images, size)
                    else:
                        generated = []

                    if generated:
                        st.success(f"✅ 生成完成！共 {len(generated)} 张图片")
                        img_cols = st.columns(min(len(generated), 4))
                        for i, img_bytes in enumerate(generated):
                            with img_cols[i % 4]:
                                img = Image.open(io.BytesIO(img_bytes))
                                st.image(img, caption=f"图片 {i+1}", use_container_width=True)
                                st.download_button(
                                    f"⬇️ 下载图片 {i+1}",
                                    data=img_bytes,
                                    file_name=f"generated_{i+1}.png",
                                    mime="image/png",
                                    key=f"dl_demo_{i}",
                                )
                    else:
                        st.warning("未能生成图片，请检查提示词或重试")

                except Exception as e:
                    st.error(f"❌ 生成失败: {e}")

with col2:
    if not st.session_state.get("_generated"):
        st.header("🖼️ 生成结果")
        st.info("👈 上传产品图片并输入提示词后，点击「开始生成」")

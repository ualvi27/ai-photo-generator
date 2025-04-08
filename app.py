import os
import re
import smtplib
import torch
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from email.message import EmailMessage
from dotenv import load_dotenv

# ===== Load environment variables securely =====
load_dotenv()
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# ===== Configuration =====
st.set_page_config(page_title="AI Photo Generator", layout="centered")
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

model_id = "runwayml/stable-diffusion-v1-5"
signature_texts = ["@ualvi27", "@AIXG5G"]
font_size = 24
guidance_scale = 7.5
steps = 40
strength = 0.75
MAX_FREE_GENERATIONS = 2

APP_VERSION = "v1.0.0"
LAST_UPDATED = "April 8, 2025"

# ===== Load Models (only once) =====
@st.cache_resource(show_spinner=True)
def load_txt2img():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

@st.cache_resource(show_spinner=True)
def load_img2img():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

pipe_txt2img = load_txt2img()
pipe_img2img = load_img2img()

# ===== Helper Functions =====
def sanitize_filename(text):
    return re.sub(r'[\\/*?:"<>|]', "", text)[:25].replace(" ", "_")

def add_signature(image):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    y = image.height - 40
    for i, tag in enumerate(signature_texts):
        x = image.width - 250 + i * 100
        draw.text((x, y), tag, font=font, fill=(255, 255, 255))
    return image

def generate_image(prompt, uploaded_img=None):
    if uploaded_img:
        uploaded_img = uploaded_img.convert("RGB").resize((512, 512))
        result = pipe_img2img(prompt=prompt, image=uploaded_img, strength=strength,
                              guidance_scale=guidance_scale, num_inference_steps=steps)
    else:
        result = pipe_txt2img(prompt, guidance_scale=guidance_scale, num_inference_steps=steps)
    return add_signature(result.images[0])

def send_feedback_email(prompt, feedback):
    try:
        msg = EmailMessage()
        msg["Subject"] = "🧠 New AI Photo Feedback"
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = "ualvi27@gmail.com"
        msg.set_content(f"📝 Prompt:\n{prompt}\n\n💬 Feedback:\n{feedback}")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
    except Exception as e:
        st.warning(f"⚠️ Email not sent: {e}")

def save_feedback(prompt, filename, feedback):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("feedback.csv", "a") as f:
        f.write(f'"{timestamp}","{prompt}","{filename}","{feedback.strip()}"\n')

# ===== Session State: Track usage =====
if "generation_count" not in st.session_state:
    st.session_state.generation_count = 0

# ===== Streamlit UI =====
st.title("🖼️ AI Photo Generator")
st.caption("Made by @ualvi27 & @AIXG5G | Powered by Stable Diffusion")

# Prompt input
st.subheader("💡 Suggested Prompts")
suggested = [
    "A futuristic cyberpunk city at night",
    "A peaceful mountain cabin in snowfall",
    "An astronaut walking on Mars",
    "A cozy coffee shop interior",
    "A magical forest with glowing trees"
]
selected_prompt = st.selectbox("Pick a prompt or write your own:", [""] + suggested)
prompt = st.text_area("📝 Prompt", value=selected_prompt, height=120)

# Optional: Upload image for Img2Img
uploaded_img = st.file_uploader("📤 Upload a photo (optional - style guide)", type=["jpg", "jpeg", "png"])
if uploaded_img:
    st.image(uploaded_img, caption="Uploaded image", width=300)
    img = Image.open(uploaded_img)
else:
    img = None

# ===== Guest Limit Control =====
if st.session_state.generation_count >= MAX_FREE_GENERATIONS:
    st.error("🔐 You've reached your free trial limit.\nPlease log in with Gmail to continue.")
else:
    if st.button("🎨 Generate Image"):
        if not prompt.strip():
            st.warning("⚠️ Please enter a prompt.")
        else:
            with st.spinner("⏳ Generating... please wait"):
                image = generate_image(prompt, uploaded_img=img)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{sanitize_filename(prompt)}_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)

                with open("user_logs.csv", "a") as f:
                    f.write(f"{timestamp},{prompt},{filename}\n")

                st.session_state.generation_count += 1

                st.success("✅ Image generated!")
                st.image(image, caption="✨ Your AI Image", use_column_width=True)

                st.subheader("💬 Leave Feedback")
                feedback = st.text_area("Your thoughts about this image?", height=100)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📤 Submit Feedback & Download"):
                        save_feedback(prompt, filename, feedback)
                        send_feedback_email(prompt, feedback)
                        with open(filepath, "rb") as file:
                            st.download_button("📥 Download Image", data=file, file_name=filename, mime="image/png")
                        st.success("✅ Feedback sent + image downloaded!")

                with col2:
                    if st.button("⏩ Skip Feedback & Download"):
                        with open(filepath, "rb") as file:
                            st.download_button("📥 Download Image", data=file, file_name=filename, mime="image/png")

# ===== Footer Section =====
st.markdown("---")
st.markdown(f"📌 **Version:** {APP_VERSION} | **Last Updated:** {LAST_UPDATED}")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image(os.path.join("assets", "whatsapp.png"), width=32)
    st.markdown("[WhatsApp](https://wa.me/923014074704)", unsafe_allow_html=True)

with col2:
    st.image(os.path.join("assets", "fb.png"), width=32)
    st.markdown("[Facebook](https://www.facebook.com/umer.masood.965)", unsafe_allow_html=True)

with col3:
    st.image(os.path.join("assets", "x.png"), width=32)
    st.markdown("[X (Twitter)](https://x.com/AIXG5G)", unsafe_allow_html=True)

with col4:
    st.image(os.path.join("assets", "insta.jpg"), width=32)
    st.markdown("[Instagram](https://instagram.com/your_handle)", unsafe_allow_html=True)

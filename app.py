import spaces

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_DIR = "flen-crypto/FLENcognition"

print("🔥 Loading FLENcognition...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True
).to(device)

processor = AutoProcessor.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True
)

model.eval()

import gradio as gr
import markdown
from PIL import Image
import os
from datetime import datetime
import tempfile
import shutil
from pathlib import Path
from conv_for_infer import generate_conv
import base64

MARKDOWN_OUTPUT = "md_output"

@spaces.GPU
def process_images(image_paths):

    if not image_paths:
        return "<p style='color:red;'>Please upload image.</p>", None, None

    os.makedirs("md_output", exist_ok=True)

    all_text = ""

    for image_path in image_paths:
        try:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            markdown_file = os.path.join("md_output", f"{basename}.md")

            # === 你的原始逻辑 ===
            messages = generate_conv({"image_path": image_path})

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=8192
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]

            text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # 保存文件
            with open(markdown_file, "w", encoding="utf-8") as f:
                f.write(text)

            all_text += text + "\n\n"

        except Exception as e:
            all_text += f"\n\n**Error processing {image_path}: {str(e)}**\n\n"

    latex_text = all_text.replace("```markdown", "$$")
    latex_text = latex_text.replace("```", "$$")

    return all_text.strip(), latex_text, markdown_file

def download_markdown(md_file_path):
    """
    提供Markdown文件下载
    """
    if md_file_path and os.path.exists(md_file_path):
        return md_file_path
    return None

def clear_files():
    """
    清空所有内容
    """
    return None, None, None, None

def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def preview_images(files):
    """
    预览上传的图片
    """
    if not files:
        return None

    preview_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"
    for i, file in enumerate(files[:5]):  # 只显示前5张预览
        try:
            img = Image.open(file)
            # 缩略图
            img.thumbnail((150, 150))

            # 临时保存缩略图
            thumb_dir = tempfile.gettempdir()
            thumb_path = os.path.join(thumb_dir, f"thumb_{i}_{datetime.now().timestamp()}.jpg")
            img.save(thumb_path, "JPEG")
            # print("thumb_path:", thumb_path)

            preview_html += f"""
                <div style="border: 1px solid #ddd; padding: 5px; border-radius: 5px;">
                    <img src="data:image/png;base64,{image_to_base64(thumb_path)}" style="max-width: 150px; max-height: 150px;">
                    <p style="text-align: center; margin: 5px 0;">Image {i+1}</p>
                </div>
                """
        except:
            pass

    preview_html += "</div>"
    if len(files) > 5:
        preview_html += f"<p>... and {len(files) - 5} more images</p>"

    return preview_html

# 创建Gradio界面
with gr.Blocks(title="FireRed-OCR") as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="display: inline-block;">🔍 FireRed-OCR</h1>
        <p style="font-size: 14px; color: #666;"><i>Upload Image → Generate Recognition Markdown</i></p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # 左侧：输入区域
            gr.Markdown("### 📤 Upload & Select")

            # 图片上传组件
            image_input = gr.File(
                label="Upload Image",
                file_count="multiple",
                file_types=["image"],
                type="filepath"
            )

            # 图片预览
            image_preview = gr.HTML(label="Image Preview")

            with gr.Row():
                run_btn = gr.Button("🚀 Generate Markdown", variant="primary", size="lg", scale=2)
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)

        with gr.Column(scale=1):
            # 右侧：预览和下载区域
            gr.Markdown("### 👀 Preview & Download")

            preview_output = gr.Code(
                label="Markdown Code Preview",
                language="markdown",
                value=">Click「Generate Markdown」Button for Previewing",
                interactive=False
            )

            preview_img_output = gr.Markdown(
                label="Markdown Preview",
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},  # Display equations
                    {"left": "$", "right": "$", "display": False}   # Inline equations
                ]
            )

            # 下载按钮
            download_btn = gr.File(
                label="📥 Click to Download Markdown File",
                interactive=False,
                visible=True
            )

    # 添加状态存储
    md_file_state = gr.State()

    # 绑定事件
    def update_preview(files):
        if files:
            return preview_images(files)
        return "<p>No image available</p>"

    image_input.change(
        fn=update_preview,
        inputs=[image_input],
        outputs=[image_preview]
    )

    run_btn.click(
        fn=process_images,
        # inputs=[image_input, markdown_input],
        inputs=[image_input],
        outputs=[preview_output, preview_img_output, md_file_state]
    ).then(
        fn=download_markdown,
        inputs=[md_file_state],
        outputs=[download_btn]
    )

    clear_btn.click(
        fn=clear_files,
        inputs=[],
        # outputs=[image_input, markdown_input, preview_output, download_btn]
        outputs=[image_input, preview_output, preview_img_output, download_btn]
    ).then(
        fn=lambda: "<p>No image available</p>",
        inputs=[],
        outputs=[image_preview]
    )

    # 添加页脚
    gr.Markdown("""
    ---
    <p style="text-align: center; color: #666;">✨ Convert Images to Standard Markdown Easily ✨</p>
    """)

# 配置并启动应用
if __name__ == "__main__":
    demo.queue().launch(
        ssr_mode=False
    )

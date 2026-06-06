"""
QwenVL-Tutor Web 演示
Gradio 界面，支持拍照上传题目并获得分步解答
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import warnings
import gradio as gr
from PIL import Image
from model.qwen_vlm import QwenVLTutor, QwenVLTutorConfig

warnings.filterwarnings('ignore')


EDU_DEFAULT_PROMPT = (
    "请解答这道题目。先分析题目考查的知识点，然后分步骤展示解题过程，最后给出明确答案。"
)


def load_model(model_path: str, base_model_name: str = "./model/Qwen2-VL-2B-Instruct"):
    """加载训练好的 QwenVL-Tutor 模型"""
    config = QwenVLTutorConfig(
        model_name_or_path=base_model_name,
        use_lora=True,
    )
    model = QwenVLTutor(config)

    if os.path.exists(model_path):
        from peft import PeftModel
        model.base_model = PeftModel.from_pretrained(
            model.base_model, model_path
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"模型已加载: {model_path} (可训练参数: {param_count:.2f}M)")

    return model, device


def chat_with_image(model, processor, device, image, question, max_new_tokens=512,
                    temperature=0.7, top_p=0.9, top_k=50):
    """拍照做题核心函数"""
    if image is None:
        yield "请先上传题目图片"
        return

    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question or EDU_DEFAULT_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        generated_ids = model.base_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    response_ids = generated_ids[0][inputs.input_ids.shape[1]:]
    response = processor.tokenizer.decode(response_ids, skip_special_tokens=True)
    yield response


def launch_demo(model_path: str, base_model: str, server_port: int = 7860):
    """启动 Gradio Web 演示"""
    model, device = load_model(model_path, base_model)
    processor = model.processor

    with gr.Blocks(
        title="QwenVL-Tutor - 拍照做题",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 📸 QwenVL-Tutor - 拍照做题助手

            上传题目图片，获取详细的解题步骤和答案。

            支持：数学、物理、化学、生物、英语、历史、地理等学科题目
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="📷 上传题目图片",
                    height=300,
                )
                question_input = gr.Textbox(
                    label="✏️ 具体问题（可选）",
                    placeholder="例如：请解析这道数学题的解题思路...",
                    value=EDU_DEFAULT_PROMPT,
                    lines=3,
                )
                with gr.Row():
                    submit_btn = gr.Button("🚀 开始解答", variant="primary")
                    clear_btn = gr.Button("🗑️ 清空")

                with gr.Accordion("⚙️ 高级设置", open=False):
                    max_tokens = gr.Slider(
                        64, 1024, value=512, step=64,
                        label="最大生成长度"
                    )
                    temperature = gr.Slider(
                        0.1, 1.5, value=0.7, step=0.05,
                        label="温度（越高越有创造性）"
                    )
                    top_p = gr.Slider(
                        0.5, 1.0, value=0.9, step=0.05,
                        label="Top-p 采样"
                    )

            with gr.Column(scale=1):
                output_text = gr.Markdown(
                    label="📝 解答",
                    value="等待上传题目...",
                    elem_id="output",
                )

        submit_btn.click(
            fn=lambda img, q, mt, t, tp: chat_with_image(
                model, processor, device, img, q, mt, t, tp
            ),
            inputs=[image_input, question_input, max_tokens, temperature, top_p],
            outputs=[output_text],
        )

        clear_btn.click(
            fn=lambda: (None, EDU_DEFAULT_PROMPT, ""),
            outputs=[image_input, question_input, output_text],
        )

        gr.Markdown(
            """
            ---
            ### 💡 使用提示
            - 拍照时尽量保证题目清晰、光照充足
            - 支持手写题目和印刷题目
            - 模型会尽可能给出分步解析，帮助理解解题思路
            - 如遇识别不准确，可尝试调整拍摄角度或放大题目部分

            ### 📊 支持的数据集
            ScienceQA · MathVerse · MathVista · OCR-VQA · TabMWP · C-Eval
            """
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QwenVL-Tutor Web Demo")
    parser.add_argument("--model_path", type=str, default="../out/edu_grpo",
                        help="训练好的模型路径")
    parser.add_argument("--base_model", type=str, default="./model/Qwen2-VL-2B-Instruct",
                        help="基座模型路径")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    args = parser.parse_args()

    launch_demo(args.model_path, args.base_model, args.port)
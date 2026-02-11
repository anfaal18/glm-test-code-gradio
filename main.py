import gradio as gr
import torch
import spaces
import os
import tempfile
from PIL import Image, ImageOps
from threading import Thread
from typing import Iterable
from transformers import AutoProcessor, AutoModelForImageTextToText

from transformers.image_utils import load_image
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

MODEL_PATH = "zai-org/GLM-OCR"

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

TASK_PROMPTS = {
    "Text": "Text Recognition:",
    "Formula": "Formula Recognition:",
    "Table": "Table Recognition:",
}

@spaces.GPU
def process_image(image, task):
    if image is None:
        return "Please upload an image first"
    
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image = ImageOps.exif_transpose(image)
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    image.save(tmp.name, 'PNG')
    tmp.close()
    
    prompt = TASK_PROMPTS.get(task, "Text Recognition:")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": tmp.name},
                {"type": "text", "text": prompt}
            ],
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    inputs.pop("token_type_ids", None)
    
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    os.unlink(tmp.name)
    
    return output_text.strip()

with gr.Blocks() as demo:
    
    gr.Markdown("# **GLM-OCR**", elem_id="main-title")
    gr.Markdown("*A multimodal [OCR model](https://huggingface.co/zai-org/GLM-OCR) for complex document understanding.*", elem_id="subtitle")
    
    with gr.Row():
        
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                sources=["upload", "clipboard"],
                height=300
            )
            with gr.Row():
                task = gr.Radio(
                    choices=list(TASK_PROMPTS.keys()),
                    value="Text",
                    label="Recognition Type"
                )

            with gr.Row():
                btn = gr.Button("Perform OCR", variant="primary")
        
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("Text"):
                    output_text = gr.Textbox(
                        label="Output",
                        lines=18,
                        interactive=True,
                    )
                
                with gr.Tab("Markdown"):
                    output_md = gr.Markdown(value="")
    
    def run_ocr(image, task):
        result = process_image(image, task)
        return result, result
    
    btn.click(
        run_ocr,
        [image_input, task],
        [output_text, output_md]
    )
    
    image_input.change(
        lambda: ("", ""),
        None,
        [output_text, output_md]
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(mcp_server=True, ssr_mode=False, show_error=True)

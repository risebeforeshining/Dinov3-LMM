import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import json
import torch
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
import torch
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer

from dinov3.utils import tokenizer_image_token
from dinov3.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from dinov3.conversation import conv_templates, SeparatorStyle

import dinov3.model.language_model.dinov3_vicuna
import dinov3.model.dino_arch
from transformers import AutoModelForCausalLM, AutoImageProcessor
# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------
checkpoint_path = "./checkpoints/dinov3-v1.5-7b-cholec17k"

# ------------------------------------------------------------------
# 加载 tokenizer 和模型
# ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True, 
)
processor = AutoImageProcessor.from_pretrained("/dataT0/Free/wcfei/dinov3/models/dinov3")


# 手动加载 DINOv3 Vision Tower 权重
vision = model.get_model().get_vision_tower()
vision.load_model(device_map="auto", dtype=torch.bfloat16)  
model.eval()

# ------------------------------------------------------------------
# 推理函数
# ------------------------------------------------------------------
def run_inference(image_path, question):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    # 处理图像
    image_tensor = inputs["pixel_values"][0].to(model.device)

    # 构造 prompt
    conv = conv_templates["v1"].copy()
    inp = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=[image_tensor],
            max_new_tokens=32,
            do_sample=False
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output.strip()

# 定义全局变量用于保存两次前向的输入
projector_inputs = []
projector_outputs = []
def register_projector_input_hook(model):
    """注册一个forward hook来捕获mm_projector的输入"""
    def hook_fn(module, input, output):
        # input是tuple，第一个元素是输入张量
        x_in = input[0].detach().cpu()
        # 输出 (Tensor)
        x_out = output.detach().cpu()
        projector_inputs.append(x_in)
        projector_outputs.append(x_out)
        print(f"[Hook] mm_projector 输入形状: {tuple(x_in.shape)}")
        print(f"[Hook] mm_projector 输出形状: {tuple(x_out.shape)}")
        print(f"[Hook] 输入张量前5个值: {x_in.view(-1)[:5]}")
        print(f"[Hook] 输出前5个值: {x_out.view(-1)[:5]}")
    handle = model.model.mm_projector.register_forward_hook(hook_fn)
    return handle

image_path = "/homes/wcfei/download/GraSP_1fps/frames/CASE041/00000.jpg"
hook_handle = register_projector_input_hook(model)
prompt = "What is the action performed by Bipolar Forceps ?"



print(f"输出: {run_inference(image_path, prompt)}")
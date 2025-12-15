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

print(model)

model.eval()

vision = model.model.vision_tower
print("Is vision loaded? ", vision.is_loaded)


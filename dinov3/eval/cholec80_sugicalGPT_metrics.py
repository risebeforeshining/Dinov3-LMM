import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

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
TEST_JSON = "/dataT0/Free/wcfei/dataset/sft-cholec80-test.json"   # test 数据
FRAME_ROOT = "/dataT0/Free/wcfei/dataset/cholec80"         # 根据自己数据结构调整

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
    image_tensor = inputs["pixel_values"].to(model.device, dtype=torch.bfloat16)

    # 构造 prompt
    conv = conv_templates["v1"].copy()
    inp = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    attention_mask = input_ids.ne(tokenizer.pad_token_id).int().cuda()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            max_new_tokens=32,
            do_sample=False
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output.strip()


################################################################
# ★ 第一步：准备 13 个 C80 类别（与论文一致）
################################################################
C80_CLASSES = [
    'no', 'calot triangle dissection', 'yes', '1', '2',
    'gallbladder dissection', 'clipping cutting', 'gallbladder retraction',
    '0', 'cleaning coagulation', 'gallbladder packaging', 'preparation', '3'
]
CLASS2IDX = {c: i for i, c in enumerate(C80_CLASSES)}

def normalize_answer(text):
    return text.lower().strip()

################################################################
# ★ 主程序：逐条推理并统计分类指标（论文实现）
################################################################
if __name__ == "__main__":

    with open(TEST_JSON, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    y_true = []
    y_pred = []

    for item in test_data:
        image_path = os.path.join(FRAME_ROOT, item["image"])
        convs = item["conversations"]

        for i in range(0, len(convs), 2):
            question = convs[i]["value"].replace("<image>\n", "").strip()
            gt_answer = normalize_answer(convs[i+1]["value"])

            # 模型推理
            pred_raw = run_inference(image_path, question)
            pred_answer = normalize_answer(pred_raw)

            ################################################################
            # ★ 将 GT 转换为类别索引
            ################################################################
            if gt_answer not in CLASS2IDX:
                print("Warning: GT 不属于 C80 13 类之一：", gt_answer)
                continue

            gt_idx = CLASS2IDX[gt_answer]

            ################################################################
            # ★ 将预测答案也映射到 13 类之一（最重要！）
            #
            #   → 与论文一致，只要答案文本和类名完全匹配
            #   → 若无法匹配，可视为预测错误（可映射为 -1）
            ################################################################
            if pred_answer in CLASS2IDX:
                pred_idx = CLASS2IDX[pred_answer]
            else:
                pred_idx = -1   # 无法识别的预测，必错

            y_true.append(gt_idx)
            y_pred.append(pred_idx)

            print("="*60)
            print(f"Image: {item['id']}")
            print(f"Q: {question}")
            print(f"GT: {gt_answer}")
            print(f"Pred: {pred_answer}")
            print(f"✔ Correct" if (gt_answer == pred_answer) else f"✘ Wrong")

    ################################################################
    # ★ 开始计算论文的指标
    ################################################################
    # 过滤掉 pred=-1 的样本（论文中不会出现这种情况）
    valid_idx = [i for i, p in enumerate(y_pred) if p != -1]
    y_true_valid = [y_true[i] for i in valid_idx]
    y_pred_valid = [y_pred[i] for i in valid_idx]

    # Accuracy (micro)
    acc = accuracy_score(y_true_valid, y_pred_valid)

    # Macro Precision / Recall / FScore
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true_valid,
        y_pred_valid,
        average='macro',
        zero_division=1
    )

    ################################################################
    # ★ 输出结果（与论文一致）
    ################################################################
    print("\n================ (C80-VQA data) ================")
    print(f"y_true_valid: {y_true_valid}")
    print(f"y_pred_valid: {y_pred_valid}")    
    
    print("\n================ FINAL METRICS (C80-VQA) ================")
    print(f"Accuracy (micro): {acc:.4f}")
    print(f"Precision_macro: {precision:.4f}")
    print(f"Recall_macro:    {recall:.4f}")
    print(f"F1_macro:        {fscore:.4f}")

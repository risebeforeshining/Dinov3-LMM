import torch
import torch.nn as nn

from transformers import AutoImageProcessor, AutoModel, AutoConfig


# pretrained_model_name = "/dataT0/Free/wcfei/dinov3/models/dinov3"
# processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
# model = AutoModel.from_pretrained(
#     pretrained_model_name, 
#     device_map="auto", 
# )
# model.eval()
pretrained_model_path = "/dataT0/Free/wcfei/dinov3/models/dinov3"

class Dinov3ViTVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            print(f'Loading DINOv3 vision tower from {pretrained_model_path}...')
            self.load_model()
        else:
            print(f'Delay loading DINOv3 vision tower from {pretrained_model_path}...')
            self.cfg_only = AutoConfig.from_pretrained(pretrained_model_path)

    def load_model(self, device_map=None, dtype=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = AutoImageProcessor.from_pretrained(pretrained_model_path, trust_remote_code=True)
        self.vision_tower = AutoModel.from_pretrained(pretrained_model_path, device_map=device_map, trust_remote_code=True, dtype=dtype)
        # self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def unfreeze_last_layers(self, num_layers: int):
        if num_layers <= 0:
            return

        # 冻结全部
        self.vision_tower.requires_grad_(False)

        # 获取 encoder 层列表（不同模型可能名字不一样）
        encoder = None
        if hasattr(self.vision_tower, "model") and hasattr(self.vision_tower.model, "encoder"):
            encoder = self.vision_tower.model.encoder
        elif hasattr(self.vision_tower, "encoder"):
            encoder = self.vision_tower.encoder
        else:
            raise ValueError("Cannot find encoder in vision tower")

        blocks = (
            encoder.layer if hasattr(encoder, "layer") # ⬅️ 增加对 '.layer' 的检查
            else encoder.layers if hasattr(encoder, "layers")
            else encoder.blocks if hasattr(encoder, "blocks")
            else None
        )

        if blocks is None:
            raise ValueError("Cannot find transformer blocks (layer/layers/blocks) in DINOv3 encoder")

        total_layers = len(blocks)
        start = max(0, total_layers - num_layers)

        print(f"🔓 Unfreezing DINOv3 layers: {start} → {total_layers-1}")

        for layer_idx in range(start, total_layers):
            for p in blocks[layer_idx].parameters():
                p.requires_grad = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if isinstance(images, list):
            out_features = []
            for img in images:
                img = img.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                outs = self.vision_tower(img, output_hidden_states=True)
                feats = self.feature_select(outs).to(img.dtype)
                out_features.append(feats)
        else:
            imgs = images.to(device=self.device, dtype=self.dtype)
            outs = self.vision_tower(imgs, output_hidden_states=True)
            out_features = self.feature_select(outs).to(imgs.dtype)

        return out_features


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


from typing import Optional
from dataclasses import dataclass

import torch
from einops import rearrange
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import open_clip

from model.base import BaseModule

class ClipTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "lambdalabs/sd-image-variations-diffusers"
        weights: Optional[str] = None

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.weight_dtype = torch.bfloat16  # TODO hard coding
        pretrained_model_name_or_path = self.cfg.pretrained_model_name_or_path

        from transformers import CLIPTextModel, CLIPTokenizer
        text_tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-depth",
                                                       subfolder="tokenizer")
        self.register_non_module(
            "text_tokenizer",
            text_tokenizer,
        )

        self.register_non_module(
            "text_encoder",
            CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-depth", subfolder="text_encoder").to(
                self.device, dtype=self.weight_dtype
            ),
        )

        text_encoder = self.non_module("text_encoder")
        for p in text_encoder.parameters():
            p.requires_grad_(False)
        text_encoder.eval()

        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="image_encoder").to(self.device, dtype=self.weight_dtype)

        feature_extractor = CLIPImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="feature_extractor")

        self.clip_image_mean = torch.as_tensor(feature_extractor.image_mean)[:, None, None].to(
            self.device, dtype=self.weight_dtype)
        self.clip_image_std = torch.as_tensor(feature_extractor.image_std)[:, None, None].to(
            self.device, dtype=self.weight_dtype)

        self.register_non_module(
            "image_encoder",
            image_encoder,
        )

        image_encoder = self.non_module("image_encoder")
        for p in image_encoder.parameters():
            p.requires_grad_(False)
        image_encoder.eval()

        self.register_non_module(
            "feature_extractor",
            feature_extractor,
        )

        oc_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-H-14', pretrained='laion2b_s32b_b79k'
        )
        oc_model.to(self.device)
        oc_model.eval()
        for p in oc_model.parameters():
            p.requires_grad_(False)

        self.openclip_image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                                device=self.device, dtype=torch.float32)[:, None, None]
        self.openclip_image_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                                device=self.device, dtype=torch.float32)[:, None, None]
        self.openclip_image_size = 224

        self.register_non_module("openclip_model", oc_model)

    def process_image(self, image: torch.Tensor) -> torch.Tensor:
        image_encoder = self.non_module("image_encoder")

        batch_size = image.shape[0]
        image = rearrange(image, "B N H W C -> (B N) C H W")
        imgs_in_proc = (image - self.clip_image_mean) / self.clip_image_std
        imgs_in_proc = TF.resize(imgs_in_proc,
                                 (self.non_module('feature_extractor').crop_size['height'], self.non_module('feature_extractor').crop_size['width']),
                                 interpolation=InterpolationMode.BICUBIC)

        image_embeddings = image_encoder(imgs_in_proc.to(self.weight_dtype)).image_embeds
        image_embeddings = rearrange(image_embeddings, "(B N) C -> B (N C)", B=batch_size)

        return image_embeddings

    def process_text(self, prompts) -> torch.Tensor:
        text_tokenizer = self.non_module("text_tokenizer")

        prompt_ids = text_tokenizer(
            prompts, max_length=text_tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        text_encoder = self.non_module("text_encoder")
        text_embeddings = text_encoder(prompt_ids)[1].to(self.device, dtype=self.weight_dtype)

        return text_embeddings

    def process_pseudo_text(self, image: torch.Tensor) -> torch.Tensor:
        oc_model = self.non_module("openclip_model")

        batch_size = image.shape[0]
        image = rearrange(image, "B N H W C -> (B N) C H W").contiguous()
        imgs = image.to(torch.float32)
        imgs = TF.resize(
            imgs,
            (self.openclip_image_size, self.openclip_image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True
        )

        imgs = (imgs - self.openclip_image_mean) / self.openclip_image_std
        feats = oc_model.encode_image(imgs.to(self.device))
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        feats = rearrange(feats, "(B N) C -> B (N C)", B=batch_size)

        feats = feats.to(self.weight_dtype)

        return feats
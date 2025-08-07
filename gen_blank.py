import os
import torch
import numpy as np
from diffusers import FluxPriorReduxPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import T5TokenizerFast, T5EncoderModel
from utils.embedder import Embedder, ImageEmbedder
from utils.config import config

if __name__ == "__main__":
    args = config()
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder").to(args.device)
    tokenizer_2 = T5TokenizerFast.from_pretrained(args.base_model, subfolder="tokenizer_2")
    text_encoder_2 = T5EncoderModel.from_pretrained(args.base_model, subfolder="text_encoder_2").to(args.device)
    redux_pipe = FluxPriorReduxPipeline.from_pretrained(args.redux_model, torch_dtype=args.dtype).to(args.device)

    clip_embedder = Embedder(tokenizer, text_encoder, 77, "pooler_output").to(args.device)
    t5_embedder = Embedder(tokenizer_2, text_encoder_2, 512, "last_hidden_state").to(args.device)
    image_embedder = ImageEmbedder(redux_pipe).to(args.device)

    blank_txt = ""
    blank_image = np.zeros((1, 3, 512, 512))

    vec = clip_embedder([blank_txt])
    txt = t5_embedder([blank_txt])
    img_vec, img_txt = image_embedder(blank_image)

    if not os.path.exists(args.blank_path):
        os.makedirs(args.blank_path)
    torch.save(vec, os.path.join(args.blank_path, "clip.pt"))
    torch.save(txt, os.path.join(args.blank_path, "t5.pt"))
    torch.save(img_vec, os.path.join(args.blank_path, "redux_clip.pt"))
    torch.save(img_txt, os.path.join(args.blank_path, "redux_t5.pt"))

    

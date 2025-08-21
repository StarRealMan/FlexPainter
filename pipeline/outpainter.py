import torch
from einops import rearrange
from tqdm import tqdm

from model.outpainter_net import OutpainterNet
from model.clip import ClipTokenizer

class OutpainterPipe():
    def __init__(self, device, dtype):
        outpainter_config = {
            "in_channels": 10,
            "out_channels": 3,
            "num_layers": [1, 1, 1, 1, 1],
            "point_block_num": [1, 1, 2, 4, 6],
            "block_out_channels": [32, 256, 1024, 1024, 2048],
            "dropout": [0.0, 0.0, 0.0, 0.1, 0.1],
            "use_uv_head": True,
            "block_type": ["uv", "point_uv", "uv_dit", "uv_dit", "uv_dit"],
            "voxel_size": [0.01, 0.02, 0.05, 0.05, 0.05],
            "window_size": [0, 256, 256, 512, 1024],
            "num_heads": [4, 4, 16, 16, 16],
            "skip_input": True,
            "skip_type": "adaptive",
            "weights": None
        }        
        clip_config = {
            "pretrained_model_name_or_path": "lambdalabs/sd-image-variations-diffusers"
        }

        self.outpainter = OutpainterNet(outpainter_config).to(device)
        self.clip = ClipTokenizer(clip_config).to(device)
        self.device = device
        self.dtype = dtype

        self.pbar = tqdm(total=0, position=0, leave=True)
    
    def load_weights(self, outpainter_weights):
        checkpoint = torch.load(outpainter_weights, map_location=torch.device("cpu"))

        state_dict = checkpoint["state_dict"]
        normal_state_dict = {key.replace("backbone.", ""): value for key, value in state_dict.items() if key.startswith("backbone.")}
        ema_state_dict = {key.replace("backbone_ema.", ""): value for key, value in state_dict.items() if key.startswith("backbone_ema.")}
        
        new_state_dict = {}
        for key in normal_state_dict.keys():
            value = ema_state_dict[key.replace(".", "")]
            new_state_dict[key] = value

        self.outpainter.load_state_dict(new_state_dict)

    def prepare_diffusion_data(self, mask_map, position_map):

        diffusion_data = {
            "position_map": position_map,
            "mask_map": mask_map,
        }

        return diffusion_data

    def prepare_condition_info(
            self,
            mesh,
            prompt,
            rgb_conds,
            baked_image,
            baked_weight
    ):
        if prompt is not None:
            text_embeddings = self.clip.process_text(prompt).to(dtype=self.dtype)
        else:
            text_embeddingss = []
            for rgb_cond in rgb_conds:
                rgb_cond = rgb_cond.permute(1, 2, 0).unsqueeze(0).unsqueeze(0)
                text_embeddings = self.clip.process_pseudo_text(rgb_cond).to(dtype=self.dtype)
                text_embeddingss.append(text_embeddings)
            text_embeddingss = torch.cat(text_embeddingss, dim=0)
            text_embeddings = text_embeddingss.mean(dim=0)

        image_embeddingss = []
        for rgb_cond in rgb_conds:
            rgb_cond = rgb_cond.permute(1, 2, 0).unsqueeze(0).unsqueeze(0)
            image_embeddings = self.clip.process_image(rgb_cond).to(dtype=self.dtype)
            image_embeddingss.append(image_embeddings)
        image_embeddingss = torch.cat(image_embeddingss, dim=0)
        image_embeddings = image_embeddingss.mean(dim=0)

        condition_info = {
            "mesh": mesh,
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
            "baked_image": baked_image,
            "baked_weight": baked_weight,
        }

        return condition_info

    def inference(
            self,
            diffusion_data,
            condition,
            condition_drop=None,
    ):
        
        mask_map = diffusion_data["mask_map"]
        position_map = diffusion_data["position_map"]
        timesteps = diffusion_data["timesteps"]
        input_tensor = diffusion_data["noisy_images"]

        text_embeddings = condition["text_embeddings"]
        image_embeddings = condition["image_embeddings"]
        clip_embeddings = [text_embeddings, image_embeddings]

        mesh = condition["mesh"]
        
        baked_image = condition["baked_image"]
        baked_weight = condition["baked_weight"]

        if condition_drop is None:
            condition_drop = torch.zeros(input_tensor.shape[0], device=input_tensor.device, dtype=input_tensor.dtype)

        output, addition_info = self.outpainter(
            input_tensor,
            mask_map,
            position_map,
            timesteps*1000,
            clip_embeddings,
            mesh,
            baked_image,
            baked_weight,
            condition_drop=condition_drop,
        )

        return output, addition_info

    @torch.no_grad()
    def __call__(
            self,
            mesh,
            prompt,
            rgb_conds,
            baked_image,
            baked_weight,
            mask_map,
            position_map,
            test_num_steps,
            test_cfg_scale,
            guidance_interval,
            guidance_rescale,
    ):
            
        diffusion_data = self.prepare_diffusion_data(mask_map, position_map)
        condition_info = self.prepare_condition_info(mesh, prompt, rgb_conds, baked_image, baked_weight)

        B, C, H, W = diffusion_data["mask_map"].shape
        noise = torch.randn((B, 3, H, W), device=self.device)
        noisy_images = noise

        t_span=torch.linspace(0, 1, test_num_steps, device=self.device)
        delta = 1.0 / test_num_steps

        self.pbar.total = test_num_steps
        for i, t in enumerate(t_span):
            timestep = t.repeat(B)
            diffusion_data["timesteps"] = timestep
            diffusion_data["noisy_images"] = noisy_images
            cond_step_out, addition_info = self.inference(diffusion_data, condition_info)

            if (
                    test_cfg_scale != 0.0
                    and guidance_interval[0] <= t <= guidance_interval[1]
            ):
                uncond_step_out, _ = self.inference(diffusion_data, condition_info, condition_drop=torch.ones(B, device=self.device, dtype=self.dtype))
                step_out = uncond_step_out + test_cfg_scale * (cond_step_out - uncond_step_out)
                
                if guidance_rescale != 0:
                    std_pos = cond_step_out.std(dim=list(range(1, cond_step_out.ndim)), keepdim=True)
                    std_cfg = step_out.std(dim=list(range(1, step_out.ndim)), keepdim=True)

                    step_out *= guidance_rescale * (std_pos / std_cfg) + (1 - guidance_rescale)
            else:
                step_out = cond_step_out

            noisy_images = noisy_images + delta * step_out
            self.pbar.update(1)
        
        final_res = noisy_images.clamp(-1, 1)
        final_res = (final_res + 1) / 2
        final_res = final_res * mask_map

        return final_res
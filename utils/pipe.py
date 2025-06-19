import torch
from einops import rearrange
from PIL import Image

def aggregate_embed(prompt_embeds_i, prompt_embeds_t, 
                    redux_strength, add_prompt_embeds_i = None):
    
    if prompt_embeds_i is None:
        return prompt_embeds_t
    elif add_prompt_embeds_i is not None:
        prompt_embeds_i_style = prompt_embeds_i - add_prompt_embeds_i
        prompt_embeds = torch.cat([prompt_embeds_t, prompt_embeds_i_style * redux_strength], dim=1)
    else:
        prompt_embeds = torch.cat([prompt_embeds_t, prompt_embeds_i * redux_strength], dim=1)

    return prompt_embeds

@torch.no_grad()
def mv_generation(pipe, redux_pipe, prompt, image_prompt, depths, sample_steps, cfg_scale, generator, 
                  redux_strength, true_cfg, renderer, use_style_control = False):
    grid_depths = rearrange(depths, "(rows cols) c h w -> c (rows h) (cols w)", rows=2, cols=2)
    grid_depths = grid_depths.unsqueeze(0)
    if grid_depths.shape[1] == 1:
        grid_depths = grid_depths.repeat(1, 3, 1, 1)

    if image_prompt is not None:
        redux_output = redux_pipe(image_prompt)
        prompt_embeds_i = redux_output["prompt_embeds"][:, 512: :]
    else:
        prompt_embeds_i = None
    
    prompt_embeds_t = pipe._get_t5_prompt_embeds([prompt])
    pooled_prompt_embeds_t = pipe._get_clip_prompt_embeds([prompt])

    if use_style_control:
        add_image_prompt = image_prompt.convert('L')
        add_image_prompt = Image.merge('RGB', [add_image_prompt, add_image_prompt, add_image_prompt])
        add_redux_output = redux_pipe(add_image_prompt)
        add_prompt_embeds_i = add_redux_output["prompt_embeds"][:, 512: :]
        add_pooled_prompt_embeds_i = add_redux_output["pooled_prompt_embeds"]
    else:
        add_prompt_embeds_i = None
        add_pooled_prompt_embeds_i = None
        true_cfg = 1.0
    
    prompt_embeds = aggregate_embed(prompt_embeds_i, prompt_embeds_t, 
                                    redux_strength, add_prompt_embeds_i)
    pooled_prompt_embeds = pooled_prompt_embeds_t

    images = pipe(
        true_cfg=true_cfg,
        prompt=None,
        control_image=grid_depths,
        height=grid_depths.shape[3],
        width=grid_depths.shape[2],
        num_inference_steps=sample_steps,
        guidance_scale=cfg_scale,
        generator=generator,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=add_prompt_embeds_i,
        negative_pooled_prompt_embeds=add_pooled_prompt_embeds_i,
        output_type="pt",
    ).images[0]
    
    images = rearrange(images, 'c (rows h) (cols w) -> (rows cols) c h w', rows=2, cols=2)
    
    return images

@torch.no_grad()
def mv_sync_cfg_generation(pipe, redux_pipe, prompt, image_prompt, depths, sample_steps, cfg_scale, generator, 
                           redux_strength, true_cfg, tex_height, tex_width, mixing_step, renderer, 
                           blank_txt=None, blank_vec=None, weighter = None, use_style_control = False):
    grid_depths = rearrange(depths, "(rows cols) c h w -> c (rows h) (cols w)", rows=2, cols=2)
    grid_depths = grid_depths.unsqueeze(0)
    if grid_depths.shape[1] == 1:
        grid_depths = grid_depths.repeat(1, 3, 1, 1)

    if image_prompt is not None:
        redux_output = redux_pipe(image_prompt)
        prompt_embeds_i = redux_output["prompt_embeds"][:, 512: :]
    else:
        prompt_embeds_i = None
  
    prompt_embeds_t = pipe._get_t5_prompt_embeds([prompt])
    pooled_prompt_embeds_t = pipe._get_clip_prompt_embeds([prompt])

    if use_style_control:
        add_image_prompt = image_prompt.convert('L')
        add_image_prompt = Image.merge('RGB', [add_image_prompt, add_image_prompt, add_image_prompt])
        add_redux_output = redux_pipe(add_image_prompt)
        prompt_embeds_s = add_redux_output["prompt_embeds"][:, 512: :]
        pooled_prompt_embeds_s = add_redux_output["pooled_prompt_embeds"]
        add_prompt_embeds_i = prompt_embeds_s
        add_pooled_prompt_embeds_i = pooled_prompt_embeds_s
    else:
        prompt_embeds_s = None
        pooled_prompt_embeds_s = None
        if true_cfg > 1.0:
            add_prompt_embeds_i = blank_txt
            add_pooled_prompt_embeds_i = blank_vec
        else:
            add_prompt_embeds_i = None
            add_pooled_prompt_embeds_i = None
  
    prompt_embeds = aggregate_embed(prompt_embeds_i, prompt_embeds_t, 
                                    redux_strength, prompt_embeds_s)
    pooled_prompt_embeds = pooled_prompt_embeds_t

    images= pipe(
        true_cfg=true_cfg,
        prompt=None,
        control_image=grid_depths,
        height=grid_depths.shape[3],
        width=grid_depths.shape[2],
        tex_height=tex_height,
        tex_width=tex_width,
        num_inference_steps=sample_steps,
        guidance_scale=cfg_scale,
        generator=generator,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=add_prompt_embeds_i,
        negative_pooled_prompt_embeds=add_pooled_prompt_embeds_i,
        output_type="pt",
        renderer=renderer,
        weighter=weighter,
        mixing_step=mixing_step
    ).images[0]
    
    images = rearrange(images, 'c (rows h) (cols w) -> (rows cols) c h w', rows=2, cols=2)

    return images


def mv_sync_cfg_intermediate(pipe, redux_pipe, prompt, image_prompt, depths, timesteps, use_custom_timestep, 
                             cfg_scale, generator, redux_strength, true_cfg, blank_txt=None, blank_vec=None):
    grid_depths = rearrange(depths, "(rows cols) c h w -> c (rows h) (cols w)", rows=2, cols=2)
    grid_depths = grid_depths.unsqueeze(0)
    if grid_depths.shape[1] == 1:
        grid_depths = grid_depths.repeat(1, 3, 1, 1)

    if image_prompt is not None:
        redux_output = redux_pipe(image_prompt)
        prompt_embeds_i = redux_output["prompt_embeds"][:, 512: :]
    else:
        prompt_embeds_i = None
  
    prompt_embeds_t = pipe._get_t5_prompt_embeds([prompt])
    pooled_prompt_embeds_t = pipe._get_clip_prompt_embeds([prompt])

    if true_cfg > 1.0:
        add_prompt_embeds_i = blank_txt
        add_pooled_prompt_embeds_i = blank_vec
    else:
        add_prompt_embeds_i = None
        add_pooled_prompt_embeds_i = None
  
    prompt_embeds = aggregate_embed(prompt_embeds_i, prompt_embeds_t, 
                                    redux_strength, None)
    pooled_prompt_embeds = pooled_prompt_embeds_t

    images, ts = pipe.intermediate(
        true_cfg=true_cfg,
        prompt=None,
        control_image=grid_depths,
        height=grid_depths.shape[3],
        width=grid_depths.shape[2],
        timesteps=timesteps,
        guidance_scale=cfg_scale,
        generator=generator,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=add_prompt_embeds_i,
        negative_pooled_prompt_embeds=add_pooled_prompt_embeds_i,
        use_custom_timestep=use_custom_timestep
    )

    images = rearrange(images, 'b c (rows h) (cols w) -> b (rows cols) c h w', rows=2, cols=2)

    return images, ts


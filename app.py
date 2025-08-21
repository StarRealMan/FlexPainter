import os
import math
import time
import shutil
import zipfile
from typing import List, Optional, Any, cast
import random
import torch
from torchvision.utils import save_image
from PIL import Image

import gradio as gr

from diffusers import FluxPriorReduxPipeline

from pipeline.outpainter import OutpainterPipe
from pipeline.flux_sync_cfg import FluxControlSyncCFGpipeline

from spuv.ops import get_projection_matrix, get_mvp_matrix
from spuv.camera import get_c2w
from spuv.mesh_utils import load_mesh_only, vertex_transform
from spuv.nvdiffrast_utils import render_xyz_from_mesh, rasterize_geometry_maps, render_normal_from_mesh
from spuv.rasterize import NVDiffRasterizerContext
from model.utils.feature_baking import bake_image_feature_to_uv

from pipeline.weighter import Weighter
from utils.video import render_video
from utils.misc import process_image
from utils.pipe import mv_sync_cfg_generation
from utils.voronoi import voronoi_solve


device_global: Optional[str] = None
dtype_global: Optional[torch.dtype] = None
flux_pipe_global: Optional[FluxControlSyncCFGpipeline] = None
redux_pipe_global: Optional[FluxPriorReduxPipeline] = None
outpainter_global: Optional[OutpainterPipe] = None
weighter_global: Optional[Weighter] = None
loaded_models_signature: Optional[str] = None


# External links (update to your paper/GitHub)
ARXIV_LINK = "https://arxiv.org/abs/2506.02620"
GITHUB_LINK = "https://github.com/StarRealMan/FlexPainter"
PROJECT_LINK = "https://starydy.xyz/FlexPainter"


def ensure_dir(path: str):
	os.makedirs(path, exist_ok=True)


def zip_dir(dir_path: str, zip_path: str):
	with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
		for root, _, files in os.walk(dir_path):
			for f in files:
				abspath = os.path.join(root, f)
				rel = os.path.relpath(abspath, dir_path)
				zf.write(abspath, rel)


def init_models(base_model: str,
				redux_model: str,
				lora_model: Optional[str],
				outpainter_model: str,
				weighter_model: str,
				device: str,
				dtype_str: str):
	global device_global, dtype_global
	global flux_pipe_global, redux_pipe_global
	global weighter_global
	global loaded_models_signature

	dtype = torch.bfloat16 if dtype_str == 'bfloat16' else torch.float32
	signature = f"{base_model}|{redux_model}|{lora_model}|{weighter_model}|{device}|{dtype_str}"

	if loaded_models_signature == signature and \
	   flux_pipe_global is not None and redux_pipe_global is not None and \
	   weighter_global is not None:
		return

	# (Re)load
	device_global = device
	dtype_global = dtype

	flux_pipe = FluxControlSyncCFGpipeline.from_pretrained(base_model, torch_dtype=dtype, local_files_only=True)
	if lora_model is not None and len(str(lora_model).strip()) > 0 and os.path.exists(lora_model):
		flux_pipe.load_lora_weights(lora_model)
	flux_pipe.to(device)

	redux_pipe = FluxPriorReduxPipeline.from_pretrained(redux_model, torch_dtype=dtype, local_files_only=True)
	redux_pipe = redux_pipe.to(device)

	# Weighter depends on texture/image size, but weights load is independent
	weighter = Weighter(1024, 512, device)  # temp sizes; real sizes will be set per-call via preprocess
	weighter.load_weights(weighter_model)

	flux_pipe_global = flux_pipe
	redux_pipe_global = redux_pipe
	weighter_global = weighter
	loaded_models_signature = signature


def clear_cuda_cache():
	"""Clear CUDA cache while keeping models in memory."""
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

def run_pipeline(
	mode: str,
	prompt: Optional[str],
	image_prompt: Optional[Image.Image],
	mesh_file,
	seed: int,
	cfg_scale_ui: Optional[float],
	true_cfg_ui: Optional[float],
	render_azim_ui: Optional[float],
):
	# Defaults (not exposed to UI)
	base_model = './ckpts/flux_depth/snapshots/61201579c6a6ad57a1acc19ffe62a1188f8adfc9'
	redux_model = './ckpts/flux_redux/snapshots/1282f955f706b5240161278f2ef261d2a29ad649'
	lora_model: Optional[str] = './ckpts/lora/lora-final.safetensors'
	outpainter_model = './ckpts/outpainter/texgen_v1.ckpt'
	weighter_model = './ckpts/weighternet/model.safetensors'
	blank_path = './blank'
	result_root = './test'
	device = 'cuda'
	dtype_str = 'bfloat16'

	sample_steps = 30
	cfg_scale = 6.0
	true_cfg = 2.0
	render_azim = -1
	resolution = 512
	texture_size = 1024
	mixing_step = 10
	image_strength = 0.3
	frame_num = 90
	render_ele = 15.0
	stylize = (mode == 'Texture Stylization')

	# Override defaults from UI if provided
	if cfg_scale_ui is not None:
		try:
			cfg_scale = float(cfg_scale_ui)
		except Exception:
			pass
	if true_cfg_ui is not None:
		try:
			true_cfg = float(true_cfg_ui)
		except Exception:
			pass
	if render_azim_ui is not None:
		try:
			render_azim = float(render_azim_ui)
		except Exception:
			pass

	# Resolve mesh path from uploaded file
	if mesh_file is None:
		raise gr.Error('Please upload a mesh file')
	mesh_path: Optional[str] = None
	if isinstance(mesh_file, dict) and 'name' in mesh_file:
		mesh_path = mesh_file['name']
	elif isinstance(mesh_file, str):
		mesh_path = mesh_file
	else:
		try:
			mesh_path = getattr(mesh_file, 'name')
		except Exception:
			mesh_path = None
	if mesh_path is None or not os.path.exists(mesh_path):
		raise gr.Error('Failed to parse the uploaded mesh or file not found')

	# Mode-based input validation and selection
	mode = (mode or '').strip()
	if mode == 'Image-to-Texture':
		if image_prompt is None:
			raise gr.Error('Image-to-Texture mode requires an image prompt')
		prompt = None
	elif mode == 'Text-to-Texture':
		if prompt is None or len(str(prompt).strip()) == 0:
			raise gr.Error('Text-to-Texture mode requires a text prompt')
		image_prompt = None
	elif mode == 'Image+Text-to-Texture':
		if image_prompt is None or prompt is None or len(str(prompt).strip()) == 0:
			raise gr.Error('Image+Text-to-Texture mode requires both a text prompt and an image prompt')
	elif mode == 'Texture Stylization':
		# require style image and style description
		if image_prompt is None:
			raise gr.Error('Texture Stylization mode requires a style image')
		if prompt is None or len(str(prompt).strip()) == 0:
			raise gr.Error('Texture Stylization mode requires a style prompt')
	else:
		raise gr.Error(f'Unknown mode: {mode}')

	# Validate blank path
	if not os.path.exists(blank_path):
		raise gr.Error(f'Blank path not found: {blank_path}')

	# Initialize models (cached)
	init_models(base_model, redux_model, lora_model, outpainter_model, weighter_model, device, dtype_str)

	generator = torch.Generator(device=device).manual_seed(int(seed))
	ctx = NVDiffRasterizerContext('cuda', device)

	# Camera and render params
	camera_poses = [(15.0, 0.0), (-15.0, 90.0), (15.0, 180.0), (-15.0, 270)]
	camera_dist_scalar = 20 / 9
	fovy = math.radians(30)

	# Load mesh
	mesh = load_mesh_only(mesh_path, device)
	mesh = vertex_transform(mesh, mesh_scale=0.5)

	# Camera mvps for first stage
	heles = torch.tensor([pose[0] for pose in camera_poses], device=device)
	if render_azim < 0 or render_azim >= 360:
		render_azim = random.uniform(0, 360)
	azims = torch.tensor([(pose[1] + render_azim) % 360 for pose in camera_poses], device=device)
	camera_dist = torch.tensor(camera_dist_scalar, device=device).repeat(len(heles))
	c2w = get_c2w(azims, heles, camera_dist)
	proj = get_projection_matrix(fovy, 1, 0.1, 1000.0).to(device)
	mvp = get_mvp_matrix(c2w, proj)

	# Load blank embeddings for true_cfg>1
	blank_txt_path = os.path.join(blank_path, 'redux_t5.pt')
	blank_vec_path = os.path.join(blank_path, 'redux_clip.pt')
	if not os.path.exists(blank_txt_path) or not os.path.exists(blank_vec_path):
		raise gr.Error(f'Missing blank embeddings: {blank_txt_path}, {blank_vec_path}')
	blank_txt = torch.load(blank_txt_path, map_location=device).to(dtype_global)
	blank_vec = torch.load(blank_vec_path, map_location=device).to(dtype_global)

	# Prepare helpers
	if weighter_global is None or redux_pipe_global is None or flux_pipe_global is None:
		raise gr.Error('Models are not properly initialized. Please check weights and dependencies.')
	
	# Load outpainter for each run
	outpainter = OutpainterPipe(device, dtype_global)
	outpainter.load_weights(outpainter_model)
	
	weighter = cast(Any, weighter_global)
	assert weighter is not None, 'Weighter not initialized'

	# Prepare result dir
	mesh_dir_name = os.path.splitext(os.path.basename(mesh_path))[0]
	timestamp = time.strftime('%Y%m%d_%H%M%S')
	result_dir = os.path.join(result_root, mesh_dir_name, timestamp)
	ensure_dir(result_dir)

	# Stage 1: MV generation
	with torch.no_grad():
		uv_position, uv_normals, uv_mask = rasterize_geometry_maps(ctx, mesh, texture_size, texture_size)
		xyz, mask = render_xyz_from_mesh(ctx, mesh, mvp, resolution, resolution)
		# depth-based conditioning
		from utils.renderer import position_to_depth, normalize_depth, generate_ray_image, rotate_c2w
		depth = position_to_depth(xyz, c2w)
		inv_depth = normalize_depth(depth, mask).permute(0, 3, 1, 2)

		# configure weighter according to sizes and preprocess
		renderer = {"ctx": ctx, "mesh": mesh, "mvps": mvp}
		weighter.texture_size = texture_size
		weighter.render_size = resolution
		weighter.device = device
		weighter.preprocess(renderer)

		# Fixed prompt prefix (aligning with utils/config)
		fix_prompt = 'a grid of 2x2 multi-view image. white background.'
		outpainter_prompt = None
		if prompt is not None and len(str(prompt).strip()) > 0:
			outpainter_prompt = prompt
			run_prompt = f"{fix_prompt} {prompt}"
		else:
			run_prompt = fix_prompt

		image_prompt = Image.fromarray(image_prompt).convert('RGB') if image_prompt is not None else None
		images = mv_sync_cfg_generation(
			flux_pipe_global, redux_pipe_global, run_prompt, image_prompt, inv_depth,
			sample_steps, cfg_scale, generator, image_strength, true_cfg,
			texture_size, texture_size, mixing_step, renderer,
			blank_txt, blank_vec, weighter, stylize
		)
		images = images.to(torch.float32)

		# Save and prepare first-stage multi-view images
		mv_paths: List[str] = []
		images_white = []
		for i in range(len(images)):
			out_path = os.path.join(result_dir, f'rgb_{i}.png')
			save_image(images[i], out_path)
			images[i] = images[i] * mask[i].permute(2, 0, 1)
			mv_paths.append(out_path)
			img_white = process_image((images[i] * 255.0).cpu().numpy())
			img_white_t = torch.tensor((img_white / 255.0)).permute(2, 0, 1)
			images_white.append(img_white_t)
		images_white = torch.stack(images_white).to(device=device)

		# Build a 2x2 grid image for a single-file download
		try:
			mv_imgs = [Image.open(p).convert('RGB') for p in mv_paths]
			w, h = mv_imgs[0].size
			grid = Image.new('RGB', (2 * w, 2 * h), color=(255, 255, 255))
			positions = [(0, 0), (w, 0), (0, h), (w, h)]
			for img, pos in zip(mv_imgs, positions):
				grid.paste(img.resize((w, h)), pos)
			mv_grid_path = os.path.join(result_dir, 'mv_grid.png')
			grid.save(mv_grid_path)
		except Exception:
			mv_grid_path = mv_paths[0] if len(mv_paths) > 0 else None

		# Clear CUDA cache after MV generation
		clear_cuda_cache()

		# Yield Stage 1 results: show MV gallery and downloads
		stage1_gallery = [Image.open(p).convert('RGB') for p in mv_paths]
		yield (
			stage1_gallery,  # mv_gallery
			None,            # uv_pred_img
			None,            # uv_paint_img
			None,            # video
			mv_grid_path,    # download_mv_files (single grid file)
			None,            # download_uv_pred
			None,            # download_uv_paint
			None,            # download_video
		)

		# Prepare features for baking and weighter
		normal = render_normal_from_mesh(ctx, mesh, mvp, resolution, resolution)
		rays_d = generate_ray_image(mvp, resolution, resolution)
		rays_d = rotate_c2w(rays_d)
		score = torch.sum(normal * rays_d, dim=-1, keepdim=True)
		score = torch.abs(score)

		feature = torch.cat([images.permute(0, 2, 3, 1), rays_d, score, images_white.permute(0, 2, 3, 1)], dim=-1)
		uv_position_, uv_normal, uv_mask_ = rasterize_geometry_maps(ctx, mesh, texture_size, texture_size)
		image_info = {"mvp_mtx": mvp.unsqueeze(0), "rgb": feature.unsqueeze(0)}
		uv_bakes, uv_bake_masks = bake_image_feature_to_uv(ctx, [mesh], image_info, uv_position_)
		uv_bakes = uv_bakes.view(-1, feature.shape[-1], texture_size, texture_size)
		uv_bake_masks = uv_bake_masks.view(-1, 1, texture_size, texture_size)
		uv_bake_mask = uv_bake_masks.sum(dim=0, keepdim=True) > 0

		uv_bakes_white_masks = (uv_bakes[:, 7:] != 0).any(dim=1, keepdim=True).float()
		uv_bake_white_mask = uv_bakes_white_masks.sum(dim=0, keepdim=True) > 0
		final_mask = torch.bitwise_xor(uv_bake_white_mask, uv_bake_mask).float()

		uv_pred = weighter(uv_bakes[:, :3], uv_bake_masks, torch.tensor([0]).to(device))
		uv_position = uv_position_.permute(0, 3, 1, 2)
		uv_mask = uv_mask_.float().permute(0, 3, 1, 2)
		uv_pred_white_bg = uv_pred * final_mask + 1 - final_mask
		uv_pred = uv_pred * final_mask 

		image_final_mask = torch.bitwise_xor(images_white[:, :1].bool(), mask.permute(0, 3, 1, 2).bool()).float()
		images = images * image_final_mask

		# Outpaint to get final texture map
		final_res = outpainter(
			[mesh], outpainter_prompt, images, uv_pred, final_mask, uv_mask, uv_position,
			sample_steps, 3.5, (0.0, 1.0), 0.0
		)
		final_res_white_bg = final_res * uv_mask + 1 - uv_mask
		final_res = voronoi_solve(final_res.squeeze(0).permute(1, 2, 0), uv_mask.squeeze(), device=device)
		final_res = final_res.permute(2, 0, 1).unsqueeze(0)

		# Save UV related outputs
		for i in range(len(images)):
			save_image(uv_bakes[i, :3], os.path.join(result_dir, f'uv_bakes_{i}.png'))
		save_image(uv_pred_white_bg, os.path.join(result_dir, 'uv_pred.png'))
		save_image(final_res_white_bg, os.path.join(result_dir, 'uv_paint.png'))
		save_image(final_mask, os.path.join(result_dir, 'final_mask.png'))

		# Clear CUDA cache after UV prediction
		clear_cuda_cache()

		# Yield Stage 2 partial: uv_pred ready
		uv_pred_path = os.path.join(result_dir, 'uv_pred.png')
		yield (
			stage1_gallery,                      # mv_gallery (unchanged)
			Image.open(uv_pred_path).convert('RGB'),  # uv_pred_img
			None,                                 # uv_paint_img
			None,                                 # video
			mv_grid_path,                          # download_mv_files (single grid file)
			uv_pred_path,                          # download_uv_pred
			None,                                  # download_uv_paint
			None,                                  # download_video
		)

		# Render and save video
		video_path = os.path.join(result_dir, 'video.mp4')
		render_video(frame_num, render_ele, camera_dist, fovy, device, ctx, mesh, final_res.squeeze(0), resolution, torch.tensor([1, 1, 1]), video_path)

			# Clear CUDA cache after video rendering
		clear_cuda_cache()

		# Final yield: uv_paint and video ready
		uv_paint_path = os.path.join(result_dir, 'uv_paint.png')
		yield (
			stage1_gallery,                          # mv_gallery (unchanged)
			Image.open(uv_pred_path).convert('RGB'), # uv_pred_img
			Image.open(uv_paint_path).convert('RGB'),# uv_paint_img
			video_path,                               # video
			mv_grid_path,                             # download_mv_files (single grid file)
			uv_pred_path,                             # download_uv_pred
			uv_paint_path,                            # download_uv_paint
			video_path,                               # download_video
		)


def build_ui():
	with gr.Blocks() as demo:
		gr.Markdown(
		'''
		<div style="display:flex;flex-direction:column;justify-content:center;align-items:center;gap:4px;margin-top:8px;">
			<h2 style="margin:0;">Official Live Demo</h2>
			<h2 style="margin:0;">FlexPainter: Flexible and Multi-View Consistent Texture Generation</h2>
		</div>
		'''
	)

		with gr.Row():
			gr.HTML(f"""
			<div style=\"display:flex;justify-content:center;align-items:center;gap:12px;margin-top:8px;\">
				<a href=\"{ARXIV_LINK}\" target=\"_blank\">
					<img src=\"https://img.shields.io/badge/arXiv-Link-red\" alt=\"arXiv\" />
				</a>
				<a href=\"{GITHUB_LINK}\" target=\"_blank\">
					<img src=\"https://img.shields.io/badge/GitHub-Repo-blue\" alt=\"GitHub\" />
				</a>
				<a href=\"{PROJECT_LINK}\" target=\"_blank\">
					<img src=\"https://img.shields.io/badge/Project-Page-green\" alt=\"Project Page\" />
				</a>
			</div>
			""")

		# Mesh upload (hidden by default; shown after mode selection)
		with gr.Row():
			with gr.Column(scale=1):
				mesh_upload = gr.File(label='Upload Mesh (.obj/.glb)', file_types=['.obj', '.glb'], visible=False)

		with gr.Row():
			with gr.Column(scale=1):
				mode = gr.Dropdown(
					label='Mode',
					choices=['Image-to-Texture', 'Text-to-Texture', 'Image+Text-to-Texture', 'Texture Stylization'],
					value=None
				)
				prompt = gr.Textbox(label='Prompt (for stylization mode, please describe the style of the ref. image and the object. e.g. a golden style of a tree stum)', placeholder='e.g., a red jeep with shiny metallic finish', visible=False)
				style_hint = gr.Markdown('In Texture Stylization mode, provide a style prompt (e.g., "vintage watercolor style with pastel tones and subtle paper grain").', visible=False)
				image_prompt = gr.Image(label='Image Prompt (optional depending on mode)', visible=False)
				seed = gr.Number(label='Seed', value=42, precision=0, visible=False)
				cfg_scale_input = gr.Slider(label='CFG Scale', minimum=1.0, maximum=12.0, step=0.5, value=6.0, visible=False)
				true_cfg_input = gr.Slider(label='True CFG', minimum=0.0, maximum=8.0, step=0.5, value=2.0, visible=False)
				render_azim_input = gr.Number(label='Render Azimuth (°; -1 for random)', value=-1, precision=2, visible=False)

				# Run button (hidden by default)
				run_btn = gr.Button('Run', visible=False)

				# Toggle inputs visibility
				def _toggle_inputs(m):
					m = (m or '').strip()
					show_prompt = m in ['Text-to-Texture', 'Image+Text-to-Texture', 'Texture Stylization']
					show_image = m in ['Image-to-Texture', 'Image+Text-to-Texture', 'Texture Stylization']
					show_style_hint = m == 'Texture Stylization'
					show_core = m in ['Image-to-Texture', 'Text-to-Texture', 'Image+Text-to-Texture', 'Texture Stylization']
					return (
						gr.update(visible=show_prompt),        # prompt
						gr.update(visible=show_style_hint),    # style_hint
						gr.update(visible=show_image),         # image_prompt
						gr.update(visible=show_core),          # mesh_upload
						gr.update(visible=show_core),          # seed
						gr.update(visible=show_core),          # cfg_scale_input
						gr.update(visible=show_core),          # true_cfg_input
						gr.update(visible=show_core),          # render_azim_input
						gr.update(visible=show_core),          # run_btn
					)

				mode.change(
					_toggle_inputs,
					inputs=[mode],
					outputs=[
						prompt,
						style_hint,
						image_prompt,
						mesh_upload,
						seed,
						cfg_scale_input,
						true_cfg_input,
						render_azim_input,
						run_btn,
					],
				)



		with gr.Column(scale=1):
			mv_gallery = gr.Gallery(label='Stage 1 Multi-view Results', columns=2, rows=2, height=320)
			# Replace file previews with download buttons only
			download_mv_files = gr.DownloadButton(label='Download Stage 1 Multi-view Images (zip individually in folder)', interactive=False)
			with gr.Row():
				uv_pred_img = gr.Image(label='Predicted Texture (uv_pred)')
				uv_paint_img = gr.Image(label='Final Texture (uv_paint)')
			with gr.Row():
				download_uv_pred = gr.DownloadButton(label='Download Predicted Texture', interactive=False)
				download_uv_paint = gr.DownloadButton(label='Download Final Texture', interactive=False)
			video = gr.Video(label='Rendered Video', height=280)
			download_video = gr.DownloadButton(label='Download Rendered Video', interactive=False)

			# Quick start examples using demo cases (mesh + cond image)
			gr.Examples(
				examples=[
					[
						"Text-to-Texture",
						"A red demon with horns, wings, and a sword.",
						None,
						"demo/demon/a6f4426e10834dc7a6bb76536009a49b.obj"
					],
					[
						"Text-to-Texture",
						"A bronze statue of a horse on top of a marble pedestal.",
						None,
						"demo/horse/bc907bdcc9984d1884c3a343f44f3671.obj"
					],
					[
						"Text-to-Texture",
						"A wooden block model of a horse, resembling Minecraft style.",
						None,
						"demo/minecraft/f8a7a0aedeb7479799c0d8c1b39005de.obj"
					],
					[
						"Image-to-Texture",
						None,
						"demo/axe/cond.jpg",
						"demo/axe/13b98c410feb42dc940f0b40b96af9c0.obj"
					],
					[
						"Image-to-Texture",
						None,
						"demo/gun/cond.jpg",
						"demo/gun/7a473e6d89e74f6994a408fb8f47e627.obj"
					],
					[
						"Image-to-Texture",
						None,
						"demo/dress/cond.jpg",
						"demo/dress/436328880c1142d0939e29d5dce89acf.obj"
					],
					[
						"Texture Stylization",
						"A colorful style of a shell.",
						"demo/shell/style.jpg",
						"demo/shell/4e8b678b708b480a9c7ab2e1058ebe7e.obj"
					],
					[
						"Texture Stylization",
						"A golden style of a tree stum.",
						"demo/stum/style.jpg",
						"demo/stum/f1a9157a27434bc384189cc316eca128.obj"
					],
				],
				inputs=[mode, prompt, image_prompt, mesh_upload],
				label='Quick Start Cases',
			)

		run_btn.click(
			fn=run_pipeline,
			inputs=[mode, prompt, image_prompt, mesh_upload, seed, cfg_scale_input, true_cfg_input, render_azim_input],
			outputs=[
				mv_gallery,
				uv_pred_img,
				uv_paint_img,
				video,
				download_mv_files,
				download_uv_pred,
				download_uv_paint,
				download_video,
			]
		)

	return demo


def main():
	torch.set_grad_enabled(False)
	demo = build_ui()
	demo.queue().launch(server_name='0.0.0.0', server_port=6006)


if __name__ == '__main__':
	main()



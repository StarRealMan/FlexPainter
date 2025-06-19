import nvdiffrast.torch as dr
import torch

"""
This module contains type annotations for the project, using
1. Python type hints (https://docs.python.org/3/library/typing.html) for Python objects
2. jaxtyping (https://github.com/google/jaxtyping/blob/main/API.md) for PyTorch tensors

Two types of typing checking can be used:
1. Static type checking with mypy (install with pip and enabled as the default linter in VSCode)
2. Runtime type checking with typeguard (install with pip and triggered at runtime, mainly for tensor dtype and shape checking)
"""

class NVDiffRasterizerContext:
    def __init__(self, context_type: str, device: torch.device) -> None:
        self.device = device
        self.ctx = self.initialize_context(context_type, device)

    def initialize_context(
        self, context_type: str, device: torch.device
    ):
        if context_type == "gl":
            return dr.RasterizeGLContext(device=device)
        elif context_type == "cuda":
            return dr.RasterizeCudaContext(device=device)
        else:
            raise ValueError(f"Unknown rasterizer context type: {context_type}")

    def vertex_transform(
        self, verts, mvp_mtx
    ):
        with torch.cuda.amp.autocast(enabled=False):
            verts_homo = torch.cat(
                [verts, torch.ones([verts.shape[0], 1]).to(verts)], dim=-1
            )
            verts_clip = torch.matmul(verts_homo, mvp_mtx.permute(0, 2, 1))
        return verts_clip

    def rasterize(
        self,
        pos,
        tri,
        resolution,
    ):
        # rasterize in instance mode (single topology)
        return dr.rasterize(self.ctx, pos.float(), tri.int(), resolution, grad_db=True)

    def rasterize_one(
        self,
        pos,
        tri,
        resolution,
    ):
        # rasterize one single mesh under a single viewpoint
        rast, rast_db = self.rasterize(pos[None, ...], tri, resolution)
        return rast[0], rast_db[0]

    def antialias(
        self,
        color,
        rast,
        pos,
        tri,
    ):
        return dr.antialias(color.float(), rast, pos.float(), tri.int())

    def interpolate(
        self,
        attr,
        rast,
        tri,
        rast_db=None,
        diff_attrs=None,
    ):
        return dr.interpolate(
            attr.float(), rast, tri.int(), rast_db=rast_db, diff_attrs=diff_attrs
        )

    def interpolate_one(
        self,
        attr,
        rast,
        tri,
        rast_db=None,
        diff_attrs=None,
    ):
        return self.interpolate(attr[None, ...], rast, tri, rast_db, diff_attrs)
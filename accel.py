import os
#os.environ["TRITON_INTERPRET"]="1"
import torch
import triton
import triton.language as tl
from dataclasses import dataclass

@dataclass
class ContourConfig:
    tol_y_black: int
    tol_y_white: int
    tol_uv_grey: int
    patch_black: int
    patch_white: int
    pix_min_black: int
    pix_min_white: int

@triton.jit()
def patchwise_sum_threshold(
    color_mask: tl.tensor,
    patch_color: tl.constexpr, pix_min_color: tl.constexpr,
):
    # Cast bool to uint8 first
    color_mask = color_mask.to(tl.uint8)
    
    # Reshape to patches
    color_patch = tl.reshape(color_mask, [
        color_mask.shape[0], 
        color_mask.shape[1] // patch_color, patch_color,
        color_mask.shape[2] // patch_color, patch_color,
    ])

    # Sum and mask
    color_patch_sum = tl.sum(tl.sum(color_patch,axis=2),axis=3)
    color_patch_mask = color_patch_sum >= tl.full([1], pix_min_color, dtype=tl.uint8)

    return color_patch_mask

@triton.jit()
def patchwise_broadcast_and(
    white_mask: tl.tensor, black_mask: tl.tensor,
    patch_white: tl.constexpr, patch_black: tl.constexpr, patch_max: tl.constexpr
):
    white_reshape_mask = tl.reshape(white_mask, [
        white_mask.shape[0], 
        white_mask.shape[1] // (patch_max // patch_white), (patch_max // patch_white),
        white_mask.shape[2] // (patch_max // patch_white), (patch_max // patch_white),
    ])
    black_reshape_mask = tl.reshape(black_mask, [
        black_mask.shape[0], 
        black_mask.shape[1] // (patch_max // patch_black), (patch_max // patch_black),
        black_mask.shape[2] // (patch_max // patch_black), (patch_max // patch_black),
    ])
    tl.static_print(white_reshape_mask.shape)
    tl.static_print(black_reshape_mask.shape)
    out_reshape_mask = white_reshape_mask & black_reshape_mask
    out_mask = tl.reshape(out_reshape_mask, [
        out_reshape_mask.shape[0], 
        out_reshape_mask.shape[1] * out_reshape_mask.shape[2],
        out_reshape_mask.shape[3] * out_reshape_mask.shape[4],

    ])
    return out_mask

@triton.autotune(
    configs=[
        triton.Config({
                "block_n": 1,
                "block_in_h": 16,
                "block_in_w": 16
            },
            num_warps=1
        )
    ], 
    key=['yuv_stride_n', 'yuv_stride_c', 'yuv_stride_h', 'yuv_stride_w'])
@triton.jit()
def subtitle_black_contour_kernel(
    yuv_ptr, out_ptr,
    yuv_stride_n: tl.constexpr, yuv_stride_c: tl.constexpr, yuv_stride_h: tl.constexpr, yuv_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
    tol_y_black: tl.constexpr, tol_y_white: tl.constexpr, tol_uv_grey: tl.constexpr,
    patch_white: tl.constexpr, pix_min_white: tl.constexpr,
    patch_black: tl.constexpr, pix_min_black: tl.constexpr,
    patch_min: tl.constexpr, patch_max: tl.constexpr,
    block_n: tl.constexpr, block_in_h: tl.constexpr, block_in_w: tl.constexpr
):
    # Infer output block size
    block_out_h: tl.constexpr = block_in_h // patch_min
    block_out_w: tl.constexpr = block_in_w // patch_min

    # Block pointers that will load y,u,v tensors
    y_block_ptr = tl.make_block_ptr(
        base=yuv_ptr + tl.program_id(0) * yuv_stride_n,
        shape=[3, yuv_stride_c//yuv_stride_h, yuv_stride_h//yuv_stride_w],
        strides=[yuv_stride_c, yuv_stride_h, yuv_stride_w],
        block_shape=[1, block_in_h, block_in_w],
        order=[0,1,2],
        offsets=[0, tl.program_id(1) * block_in_h, tl.program_id(2) * block_in_w]
    )
    # Output block pointer
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr + tl.program_id(0) * out_stride_n,
        shape=[1, out_stride_n//out_stride_h, out_stride_h//out_stride_w],
        strides=[0, out_stride_h, out_stride_w],
        block_shape=[1, block_out_h, block_out_w],
        order=[0,1,2],
        offsets=[0, tl.program_id(1) * block_out_h, tl.program_id(2) * block_out_w]
    )

    # Shape: [1, block_in_h, block_in_w]. Dtype: uint8
    y = tl.load(y_block_ptr)
    y_block_ptr=tl.advance(y_block_ptr, (1,0,0))
    u = tl.load(y_block_ptr)
    y_block_ptr=tl.advance(y_block_ptr, (1,0,0))
    v = tl.load(y_block_ptr)
    
    # Shape: [block_n, block_in_h, block_in_w]. Dtype: bool
    grey = (
        (u >= tl.full([1], 128-tol_uv_grey, dtype=tl.uint8)) &
        (u <= tl.full([1], 128+tol_uv_grey, dtype=tl.uint8)) &
        (v >= tl.full([1], 128-tol_uv_grey, dtype=tl.uint8)) &
        (v <= tl.full([1], 128+tol_uv_grey, dtype=tl.uint8))
    )
    white = grey & (y > tl.full([1], 255-tol_y_white, dtype=tl.uint8))
    black = grey & (y < tl.full([1], tol_y_black, dtype=tl.uint8))

    # Shape: [block_n, block_in_h // patch_color, block_in_w // patch_color]. Dtype: bool
    white_mask_broadcasted = patchwise_sum_threshold(white, patch_white, pix_min_white)
    black_mask_broadcasted = patchwise_sum_threshold(black, patch_black, pix_min_black)

    # Shape: [block_n, block_in_h // patch_min, block_in_w // patch_min]. Dtype: bool
    out_mask = patchwise_broadcast_and(
        white_mask_broadcasted, black_mask_broadcasted,
        patch_white, patch_black, patch_max
    )
    tl.store(out_block_ptr, out_mask.to(tl.uint8))


def subtitle_black_contour_fusion(yuv: torch.Tensor, config: ContourConfig):
    assert yuv.is_contiguous()
    out = torch.empty(
        (yuv.shape[0], yuv.shape[2] // config.patch_white, yuv.shape[3] //config.patch_white), 
        dtype=torch.uint8,
        device=yuv.device
    )
    grid = lambda meta: (
        yuv.size(0),
        yuv.size(2) // meta["block_in_h"],
        yuv.size(3) // meta["block_in_w"],
    )
    subtitle_black_contour_kernel[grid](
        yuv, out,
        yuv.stride(0), yuv.stride(1), yuv.stride(2), yuv.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        patch_min = min(config.patch_white, config.patch_black),
        patch_max = max(config.patch_white, config.patch_black),
        **config.__dict__
    )
    return out

yuv = torch.load("debug/221.pt").clone()
#yuv = torch.full((1,3,4,4),128,device="cuda")
#yuv[0,0,0:2,0:4]=255
#yuv[0,0,1,1]=0
config = ContourConfig(
    tol_y_black=32, 
    tol_y_white=32,
    tol_uv_grey=2, 
    patch_white=4,
    pix_min_white=2,
    patch_black=16,
    pix_min_black=1,
)
out = subtitle_black_contour_fusion(yuv, config)

import torchvision
from core import bool_to_grey
torchvision.io.write_png(bool_to_grey(out).cpu(), f"debug/out.png")

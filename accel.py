import os
#os.environ["TRITON_INTERPRET"]="1"
import torch
import triton
import triton.language as tl
from dataclasses import dataclass

@dataclass
class ContourConfig:
    tol_y_bw: int
    tol_uv_grey: int
    patch_black: int
    patch_white: int
    pix_min_black: int
    pix_min_white: int

@triton.autotune(
    configs=[
        triton.Config({
                "block_h": 16,
                "block_w": 16
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
    tol_y_bw: tl.constexpr, tol_uv_grey: tl.constexpr,
    patch_white: tl.constexpr, pix_min_white: tl.constexpr,
    patch_black: tl.constexpr, pix_min_black: tl.constexpr,
    block_h: tl.constexpr, block_w: tl.constexpr
):
    block_out_h: tl.constexpr = block_h // patch_white
    block_out_w: tl.constexpr = block_w // patch_white

    y_block_ptr = tl.make_block_ptr(
        base=yuv_ptr + tl.program_id(0) * yuv_stride_n,
        shape=[3, yuv_stride_c//yuv_stride_h, yuv_stride_h//yuv_stride_w],
        strides=[yuv_stride_c, yuv_stride_h, yuv_stride_w],
        block_shape=[1, block_h, block_w],
        order=[0,1,2],
        offsets=[0, tl.program_id(1) * block_h, tl.program_id(2) * block_w]
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr + tl.program_id(0) * out_stride_n,
        shape=[1, out_stride_n//out_stride_h, out_stride_h//out_stride_w],
        strides=[0, out_stride_h, out_stride_w],
        block_shape=[1, block_out_h, block_out_w],
        order=[0,1,2],
        offsets=[0, tl.program_id(1) * block_out_h, tl.program_id(2) * block_out_w]
    )


    y = tl.load(y_block_ptr)
    y_block_ptr=tl.advance(y_block_ptr, (1,0,0))
    u = tl.load(y_block_ptr)
    y_block_ptr=tl.advance(y_block_ptr, (1,0,0))
    v = tl.load(y_block_ptr)
    
    grey = (
        (u >= tl.full([1], 128-tol_uv_grey, dtype=tl.uint8)) &
        (u <= tl.full([1], 128+tol_uv_grey, dtype=tl.uint8)) &
        (v >= tl.full([1], 128-tol_uv_grey, dtype=tl.uint8)) &
        (v <= tl.full([1], 128+tol_uv_grey, dtype=tl.uint8))
    )
    white = grey & (y > tl.full([1], 255-tol_y_bw, dtype=tl.uint8))
    black = grey & (y < tl.full([1], tol_y_bw, dtype=tl.uint8))

    white_patch = tl.reshape(white.to(tl.uint8), [
        1, 
        block_h // patch_white, patch_white,
        block_w // patch_white, patch_white,
    ])
    white_cnt = tl.sum(tl.sum(white_patch,axis=2),axis=3)
    white_mask = white_cnt >= tl.full([1], pix_min_white, dtype=tl.uint8)

    black_patch = tl.reshape(black.to(tl.uint8), [
        1, 
        block_h // patch_black, patch_black,
        block_w // patch_black, patch_black,
    ])
    black_cnt = tl.sum(tl.sum(black_patch,axis=2),axis=3)
    black_mask = black_cnt >= tl.full([1], pix_min_black, dtype=tl.uint8)

    black_mask_broadcasted = tl.expand_dims(black_mask, axis=[1,2])
    white_mask_broadcasted = tl.reshape(white_mask, [
        1,
        block_h // patch_black, patch_black // patch_white,
        block_w // patch_black, patch_black // patch_white,
    ])
    final_mask_broadcasted = black_mask_broadcasted & white_mask_broadcasted
    final_mask = tl.reshape(final_mask_broadcasted, [
        1,
        block_h // patch_white,
        block_w // patch_white,
    ])
    tl.store(out_block_ptr, final_mask.to(tl.uint8))


def subtitle_black_contour_fusion(yuv: torch.Tensor, config: ContourConfig):
    assert yuv.is_contiguous()
    assert config.patch_black >= config.patch_white
    out = torch.empty(
        (yuv.shape[0], yuv.shape[2] // config.patch_white, yuv.shape[3] //config.patch_white), 
        dtype=torch.uint8,
        device=yuv.device
    )
    grid = lambda meta: (
        yuv.size(0),
        yuv.size(2) // meta["block_h"],
        yuv.size(3) // meta["block_w"],
    )
    subtitle_black_contour_kernel[grid](
        yuv, out,
        yuv.stride(0), yuv.stride(1), yuv.stride(2), yuv.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        **config.__dict__
    )
    return out

yuv = torch.load("debug/221.pt").clone()
#yuv = torch.full((1,3,4,4),128,device="cuda")
#yuv[0,0,0:2,0:4]=255
#yuv[0,0,1,1]=0
config = ContourConfig(
    tol_y_bw=32, 
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

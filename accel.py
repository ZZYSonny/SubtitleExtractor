import os
#os.environ["TRITON_INTERPRET"]="1"
import torch
from torch import _inductor as inductor
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
        color_mask.shape[0], color_mask.shape[1], 
        color_mask.shape[2] // patch_color, patch_color,
        color_mask.shape[3] // patch_color, patch_color,
    ])

    # Sum and mask
    color_patch_sum = tl.sum(tl.sum(color_patch,axis=3),axis=4)
    color_patch_mask = color_patch_sum >= tl.full([1], pix_min_color, dtype=tl.uint8)

    return color_patch_mask

@triton.jit()
def patchwise_broadcast_and(
    white_mask: tl.tensor, black_mask: tl.tensor,
    patch_white: tl.constexpr, patch_black: tl.constexpr, patch_max: tl.constexpr
):
    white_reshape_mask = tl.reshape(white_mask, [
        white_mask.shape[0], white_mask.shape[1],
        white_mask.shape[2] // (patch_max // patch_white), (patch_max // patch_white),
        white_mask.shape[3] // (patch_max // patch_white), (patch_max // patch_white),
    ])
    black_reshape_mask = tl.reshape(black_mask, [
        black_mask.shape[0], black_mask.shape[1], 
        black_mask.shape[2] // (patch_max // patch_black), (patch_max // patch_black),
        black_mask.shape[3] // (patch_max // patch_black), (patch_max // patch_black),
    ])
    out_reshape_mask = white_reshape_mask & black_reshape_mask
    out_mask = tl.reshape(out_reshape_mask, [
        out_reshape_mask.shape[0], out_reshape_mask.shape[1], 
        out_reshape_mask.shape[2] * out_reshape_mask.shape[3],
        out_reshape_mask.shape[4] * out_reshape_mask.shape[5],

    ])
    return out_mask

@triton.autotune(
    configs=[
        triton.Config({
                "block_n": 1,
                "block_in_h": 16,
                "block_in_w": 128
            },
            num_warps=1,
            num_ctas=1,
            num_stages=1
        )
    ], 
    key=['yuv_stride_n', 'yuv_stride_c', 'yuv_stride_h', 'yuv_stride_w'])
@triton.jit()
def subtitle_black_contour_kernel(
    yuv_ptr, out_ptr,
    yuv_size_n: tl.constexpr, yuv_size_h: tl.constexpr, yuv_size_w: tl.constexpr,
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
        base=yuv_ptr,
        shape=[yuv_size_n, 3, yuv_size_h, yuv_size_w],
        strides=[yuv_stride_n, yuv_stride_c, yuv_stride_h, yuv_stride_w],
        block_shape=[block_n, 1, block_in_h, block_in_w],
        order=[0,1,2,3],
        offsets=[tl.program_id(0) * block_n, 0, tl.program_id(1) * block_in_h, tl.program_id(2) * block_in_w]
    )
    # Output block pointer
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=[yuv_size_n, 1, yuv_size_h // patch_min, yuv_size_w // patch_min],
        strides=[out_stride_n, 0, out_stride_h, out_stride_w],
        block_shape=[block_n, 1, block_out_h, block_out_w],
        order=[0,1,2,3],
        offsets=[tl.program_id(0) * block_n, 0, tl.program_id(1) * block_out_h, tl.program_id(2) * block_out_w]
    )

    # Shape: [block_n, 1, block_in_h, block_in_w]. Dtype: uint8
    y = tl.load(y_block_ptr, boundary_check=[0])
    y_block_ptr=tl.advance(y_block_ptr, (0,1,0,0))
    u = tl.load(y_block_ptr, boundary_check=[0])
    y_block_ptr=tl.advance(y_block_ptr, (0,1,0,0))
    v = tl.load(y_block_ptr, boundary_check=[0])
    
    # Shape: [block_n, 1, block_in_h, block_in_w]. Dtype: bool
    grey = (
        (u >= tl.full([1], 128-tol_uv_grey, dtype=tl.uint8)) &
        (u <= tl.full([1], 128+tol_uv_grey, dtype=tl.uint8)) &
        (v >= tl.full([1], 128-tol_uv_grey, dtype=tl.uint8)) &
        (v <= tl.full([1], 128+tol_uv_grey, dtype=tl.uint8))
    )
    white = grey & (y > tl.full([1], 255-tol_y_white, dtype=tl.uint8))
    black = grey & (y < tl.full([1], tol_y_black, dtype=tl.uint8))

    # Shape: [block_n, 1, block_in_h // patch_color, block_in_w // patch_color]. Dtype: bool
    white_mask_broadcasted = patchwise_sum_threshold(white, patch_white, pix_min_white)
    black_mask_broadcasted = patchwise_sum_threshold(black, patch_black, pix_min_black)

    # Shape: [block_n, 1, block_in_h // patch_min, block_in_w // patch_min]. Dtype: bool
    out_mask = patchwise_broadcast_and(
        white_mask_broadcasted, black_mask_broadcasted,
        patch_white, patch_black, patch_max
    )
    tl.store(out_block_ptr, out_mask.to(tl.uint8), boundary_check=[0])


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
        yuv.size(0), yuv.size(2), yuv.size(3),
        yuv.stride(0), yuv.stride(1), yuv.stride(2), yuv.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        patch_min = min(config.patch_white, config.patch_black),
        patch_max = max(config.patch_white, config.patch_black),
        **config.__dict__
    )
    return out

def subtitle_black_contour_naive(y: torch.Tensor, u: torch.Tensor, v: torch.Tensor, config: ContourConfig):
    grey_mask = torch.logical_and(
        torch.logical_and(
            u >= 128 - config.tol_uv_grey,
            u <= 128 + config.tol_uv_grey
        ),
        torch.logical_and(
            v >= 128 - config.tol_uv_grey,
            v <= 128 + config.tol_uv_grey
        )
    )
    white_mask = torch.logical_and(
        y > 255-config.tol_y_white,
        grey_mask
    )
    black_mask = torch.logical_and(
        y < config.tol_y_black,
        grey_mask
    )
    white_mask_scaled = torch.sum(
        white_mask.reshape([
            y.size(0), 
            y.size(1)//config.patch_white, config.patch_white, 
            y.size(2)//config.patch_white, config.patch_white, 
        ]),
        dim=[2,4],
        dtype=torch.uint8
    ).greater_equal(config.pix_min_white)
    black_mask_scaled = torch.sum(
        black_mask.reshape([
            y.size(0), 
            y.size(1)//config.patch_black, config.patch_black, 
            y.size(2)//config.patch_black, config.patch_black, 
        ]),
        dim=[2,4],
        dtype=torch.uint8
    ).greater_equal(config.pix_min_black)
    
    if config.patch_white < config.patch_black:
        r = config.patch_black//config.patch_white
        white_mask_scaled = white_mask_scaled.reshape([
            white_mask_scaled.size(0),
            white_mask_scaled.size(1) // r, r,
            white_mask_scaled.size(2) // r, r
        ])
        black_mask_scaled = black_mask_scaled[:,:,None,:,None]


    final = torch.logical_and(
        white_mask_scaled,
        black_mask_scaled
    )
    return final.reshape([
        final.size(0),
        final.size(1) * final.size(2),
        final.size(3) * final.size(4)
    ])

subtitle_black_contour_compiled = torch.compile(subtitle_black_contour_naive)

x = torch.randint(0,255, (512, 3, 192, 1920,), dtype=torch.uint8, device="cuda")
config = ContourConfig(
    tol_y_black=32, 
    tol_y_white=32,
    tol_uv_grey=2, 
    patch_white=4,
    pix_min_white=2,
    patch_black=16,
    pix_min_black=1,
)
#subtitle_black_contour_fusion(x, config)
print(triton.testing.do_bench(lambda: subtitle_black_contour_fusion(x, config)))
print(triton.testing.do_bench(lambda: subtitle_black_contour_compiled(x[:,0], x[:,1], x[:,2], config)))
#print(triton.testing.do_bench(lambda: subtitle_black_contour_naive(x[:,0], x[:,1], x[:,2], config)))

import torch
import triton
import triton.language as tl

from dataclasses import dataclass
@dataclass
class TextFilter:
    block_col: int
    max_text_row: int
    
    range_y_black: int
    range_y_white: int
    range_uv_grey: int
    row_min_keep: int
    row_max_break: int
    filter_white_row: int
    filter_black_row: int

@triton.jit
def elementwise_or(x,y):
    return x or y

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
    ],
    key=['num_row', 'block_col'],
)
@triton.jit
def triton_scan_text_boundary(
    num_row: tl.constexpr,

    block_col: tl.constexpr,
    max_text_row: tl.constexpr,

    yuv_ptr,
    yuv_stride_batch: tl.constexpr, 
    yuv_stride_chan: tl.constexpr, 
    yuv_stride_row: tl.constexpr, 
    yuv_stride_col: tl.constexpr,

    bound_low_ptr, bound_high_ptr,
    bound_stride_batch: tl.constexpr,
    bound_stride_row: tl.constexpr,
    bound_stride_col: tl.constexpr,

    config_range_y_black: tl.constexpr,
    config_range_y_white: tl.constexpr,
    config_range_uv_grey: tl.constexpr,
    config_row_min_keep: tl.constexpr,
    config_row_max_break: tl.constexpr,
    config_filter_white_row: tl.constexpr,
    config_filter_black_row: tl.constexpr,
):
    y_offset = (
        yuv_ptr 
        + yuv_stride_batch * tl.program_id(0)
        + yuv_stride_col * block_col * tl.program_id(1)
        + yuv_stride_col * tl.arange(0, block_col)
    )
    bound_offset = (
        0
        + bound_stride_batch * tl.program_id(0)
        + bound_stride_col * block_col * tl.program_id(1)
        + bound_stride_col + tl.arange(0, block_col)
    )


    cur_text_row = 0
    bound_low  = tl.full([block_col], num_row, dtype=tl.uint8)
    bound_high = tl.full([block_col], num_row, dtype=tl.uint8)
    group_low = num_row
    group_high = num_row

    for i in range(num_row):
        y = tl.load(y_offset)
        u = tl.load(y_offset + yuv_stride_chan)
        v = tl.load(y_offset + yuv_stride_chan * 2)

        grey_pixel = ((u >= tl.full([1], 128-config_range_uv_grey, tl.uint8))
                     &(u <= tl.full([1], 128+config_range_uv_grey, tl.uint8))
                     &(v >= tl.full([1], 128-config_range_uv_grey, tl.uint8))
                     &(v <= tl.full([1], 128+config_range_uv_grey, tl.uint8)))
        black_pixel = (y <= tl.full([1], 000+config_range_y_black, tl.uint8)) & grey_pixel
        white_pixel = (y >= tl.full([1], 255-config_range_y_white, tl.uint8)) & grey_pixel
        has_black = tl.sum(black_pixel.to(tl.int32), 0) >= config_filter_black_row
        has_white = tl.sum(white_pixel.to(tl.int32), 0) >= config_filter_white_row

        if has_white and has_black:
            group_low = min(group_low, i)
            group_high = num_row
            bound_i = tl.full([1], i, tl.uint8)
            bound_low = tl.where(white_pixel, tl.minimum(bound_i, bound_low), bound_low)
            bound_high = tl.where(white_pixel, bound_i, bound_high)
        else:
            group_high = min(group_high, i)
            # enough contingous white_row
            if i - group_high >= config_row_max_break:
                # decide if stage2 and stage3 is valid
                if i - group_low >= config_row_min_keep:
                    tl.store(bound_low_ptr + bound_offset + cur_text_row * bound_stride_row, bound_low)
                    tl.store(bound_high_ptr + bound_offset + cur_text_row * bound_stride_row, bound_high)
                    cur_text_row = min(cur_text_row + 1, max_text_row)
                # Reset All
                bound_low  = tl.full([block_col], num_row, dtype=tl.uint8)
                bound_high = tl.full([block_col], num_row, dtype=tl.uint8)
                group_low = num_row
                group_high = num_row

            
        # Increment offset
        y_offset += yuv_stride_row

    for i in range(0, max_text_row):
        if i>=cur_text_row:
            out = tl.full([block_col], num_row, dtype=tl.uint8)
            tl.store(bound_low_ptr + bound_offset + i * bound_stride_row, out)
            tl.store(bound_high_ptr + bound_offset + i * bound_stride_row, out)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
    ],
    key=['num_row', 'block_col'],
)
@triton.jit
def triton_filter_text(
    num_row: tl.constexpr,

    block_col: tl.constexpr,
    max_text_row: tl.constexpr,

    yuv_ptr,
    yuv_stride_batch: tl.constexpr, 
    yuv_stride_chan: tl.constexpr, 
    yuv_stride_row: tl.constexpr, 
    yuv_stride_col: tl.constexpr,

    bound_low_ptr, bound_high_ptr,
    bound_stride_batch: tl.constexpr,
    bound_stride_row: tl.constexpr,
    bound_stride_col: tl.constexpr,

    out_ptr,
    out_stride_batch: tl.constexpr,
    out_stride_row: tl.constexpr,
    out_stride_col: tl.constexpr,

    config_range_y_white: tl.constexpr,
    config_range_uv_grey: tl.constexpr,
):
    out_offset = (
        out_ptr
        + out_stride_batch * tl.program_id(0)
        + out_stride_col * block_col * tl.program_id(1)
        + out_stride_col * tl.arange(0, block_col)
    )
    y_offset = (
        yuv_ptr 
        + yuv_stride_batch * tl.program_id(0)
        + yuv_stride_col * block_col * tl.program_id(1)
        + yuv_stride_col * tl.arange(0, block_col)
    )
    bound_offset = (
        0
        + bound_stride_batch * tl.program_id(0)
        + bound_stride_row * tl.arange(0, max_text_row)[:, None]
        + bound_stride_col * block_col * tl.program_id(1)
        + bound_stride_col + tl.arange(0, block_col)[None, :]
    )
    low = tl.load(bound_low_ptr + bound_offset)
    high = tl.load(bound_high_ptr + bound_offset)

    for i in range(num_row):
        y = tl.load(y_offset)
        u = tl.load(y_offset + yuv_stride_chan)
        v = tl.load(y_offset + yuv_stride_chan * 2)

        grey_pixel = ((u >= tl.full([1], 128-config_range_uv_grey, tl.uint8))
                     &(u <= tl.full([1], 128+config_range_uv_grey, tl.uint8))
                     &(v >= tl.full([1], 128-config_range_uv_grey, tl.uint8))
                     &(v <= tl.full([1], 128+config_range_uv_grey, tl.uint8)))
        white_pixel = (y >= tl.full([1], 255-config_range_y_white, tl.uint8)) & grey_pixel
        mask = tl.reduce((low <= i) & (i <= high), 0, elementwise_or)
        out = tl.where(white_pixel & mask, y, tl.full([1], 0, tl.uint8))
        tl.store(out_offset, white_pixel & mask)

        y_offset += yuv_stride_row
        out_offset += out_stride_row


def scan_text_boundary(
    yuv: torch.Tensor,
    config: TextFilter
):
    bound_low_high = torch.empty(
        size = [2, yuv.shape[0], config.max_text_row, yuv.shape[-1]],
        dtype = torch.uint8,
        device = yuv.device
    )
    bound_low = bound_low_high[0]
    bound_high = bound_low_high[1]

    grid = lambda meta: (yuv.shape[0], yuv.shape[-1] // meta["block_col"])
    triton_scan_text_boundary[grid](
        num_row = yuv.shape[2],
        
        block_col = config.block_col,
        max_text_row = config.max_text_row,

        yuv_ptr = yuv,
        yuv_stride_batch = yuv.stride(0),
        yuv_stride_chan = yuv.stride(1),
        yuv_stride_row = yuv.stride(2),
        yuv_stride_col = yuv.stride(3),

        bound_low_ptr = bound_low,
        bound_high_ptr = bound_high,
        bound_stride_batch = bound_low.stride(0),
        bound_stride_row = bound_low.stride(1),
        bound_stride_col = bound_low.stride(2),

        config_range_y_black = config.range_y_black,
        config_range_y_white = config.range_y_white,
        config_range_uv_grey = config.range_uv_grey,
        config_row_min_keep = config.row_min_keep,
        config_row_max_break = config.row_max_break,
        config_filter_white_row = config.filter_white_row,
        config_filter_black_row = config.filter_black_row,
    )
    return bound_low, bound_high

def filter_text(
    yuv: torch.Tensor,
    bound_low: torch.Tensor,
    bound_high: torch.Tensor,
    config: TextFilter
):
    out = torch.empty_like(yuv[:, 0])
    grid = lambda meta: (yuv.shape[0], yuv.shape[-1] // meta["block_col"])
    triton_filter_text[grid](
        num_row = yuv.shape[2],
        
        block_col = 128,
        max_text_row = config.max_text_row,

        yuv_ptr = yuv,
        yuv_stride_batch = yuv.stride(0),
        yuv_stride_chan = yuv.stride(1),
        yuv_stride_row = yuv.stride(2),
        yuv_stride_col = yuv.stride(3),

        bound_low_ptr = bound_low,
        bound_high_ptr = bound_high,
        bound_stride_batch = bound_low.stride(0),
        bound_stride_row = bound_low.stride(1),
        bound_stride_col = bound_low.stride(2),

        out_ptr = out,
        out_stride_batch = out.stride(0),
        out_stride_row = out.stride(1),
        out_stride_col = out.stride(2),

        config_range_y_white = config.range_y_white,
        config_range_uv_grey = config.range_uv_grey,
    )
    return out

config = TextFilter(
    block_col = 128,
    max_text_row = 2,

    range_y_black = 32,
    range_y_white = 32,
    range_uv_grey = 2,
    row_min_keep = 32,
    row_max_break = 8,
    filter_white_row = 2,
    filter_black_row = 4,
)
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class FilterConfig:
    range_y_white: int
    range_uv_grey: int


@triton.jit
def elementwise_or(x, y):
    return x or y


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
    ],
    key=["num_row", "block_col"],
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
    bound_low_ptr,
    bound_high_ptr,
    bound_stride_batch: tl.constexpr,
    bound_stride_row: tl.constexpr,
    bound_stride_col: tl.constexpr,
    config_range_y_black: tl.constexpr,
    config_range_y_white: tl.constexpr,
    config_range_uv_grey: tl.constexpr,
    config_row_max_break: tl.constexpr,
    config_row_min_keep: tl.constexpr,
    config_col_min_keep: tl.constexpr,
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
        + bound_stride_col * tl.arange(0, block_col)
    )

    cur_text_row = 0
    bound_low = tl.full([block_col], num_row, dtype=tl.int32)
    bound_high = tl.full([block_col], 0, dtype=tl.int32)
    last_black = tl.full([block_col], num_row, dtype=tl.int32)
    last_white = tl.full([block_col], 0, dtype=tl.int32)
    group_high = num_row

    for i in range(num_row):
        y = tl.load(y_offset)
        u = tl.load(y_offset + yuv_stride_chan)
        v = tl.load(y_offset + yuv_stride_chan * 2)

        grey_pixel = (
            (u >= tl.full([1], 128 - config_range_uv_grey, tl.uint8))
            & (u <= tl.full([1], 128 + config_range_uv_grey, tl.uint8))
            & (v >= tl.full([1], 128 - config_range_uv_grey, tl.uint8))
            & (v <= tl.full([1], 128 + config_range_uv_grey, tl.uint8))
        )
        black_pixel = (
            y <= tl.full([1], 000 + config_range_y_black, tl.uint8)
        ) & grey_pixel
        white_pixel = (
            y >= tl.full([1], 255 - config_range_y_white, tl.uint8)
        ) & grey_pixel
        black_cnt = tl.sum(black_pixel.to(tl.int32), 0)
        white_cnt = tl.sum(white_pixel.to(tl.int32), 0)

        if i - group_high < config_row_max_break:
            bound_i = tl.full([1], i, tl.int32)
            last_black = tl.where(black_pixel, bound_i, last_black)
            bound_high = tl.where(black_pixel, last_white, bound_high)
            if (
                white_cnt >= config_filter_white_row
                and black_cnt >= config_filter_black_row
            ):
                group_high = i
                last_white = tl.where(white_pixel, bound_i, last_white)
                bound_low = tl.where(
                    white_pixel, tl.minimum(last_black, bound_low), bound_low
                )
        else:
            # enough contingous white_row
            # decide if stage2 and stage3 is valid
            filtered = (bound_high - bound_low) >= config_row_min_keep
            filtered_cnt = tl.sum(filtered.to(tl.int32), 0)
            if filtered_cnt > config_col_min_keep:
                tl.store(
                    bound_low_ptr + bound_offset + cur_text_row * bound_stride_row,
                    bound_low,
                )
                tl.store(
                    bound_high_ptr + bound_offset + cur_text_row * bound_stride_row,
                    bound_high,
                )
                cur_text_row = min(cur_text_row + 1, max_text_row)
            # Reset All
            group_high = num_row
            bound_low = tl.full([block_col], num_row, dtype=tl.int32)
            bound_high = tl.full([block_col], 0, dtype=tl.int32)

        # print("SCAN", group_high, i, black_cnt, white_cnt)

        # Increment offset
        y_offset += yuv_stride_row

    for i in range(0, max_text_row):
        bound_fake_low = tl.full([block_col], num_row, dtype=tl.int32)
        bound_fake_high = tl.full([block_col], 0, dtype=tl.int32)
        if i >= cur_text_row:

            tl.store(
                bound_low_ptr + bound_offset + i * bound_stride_row, bound_fake_low
            )
            tl.store(
                bound_high_ptr + bound_offset + i * bound_stride_row, bound_fake_high
            )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
    ],
    key=["num_row", "block_col"],
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
    bound_low_ptr,
    bound_high_ptr,
    bound_stride_batch: tl.constexpr,
    bound_stride_row: tl.constexpr,
    bound_stride_col: tl.constexpr,
    out_ptr,
    out_stride_batch: tl.constexpr,
    out_stride_row: tl.constexpr,
    out_stride_col: tl.constexpr,
    config_range_y_black: tl.constexpr,
    config_range_y_white: tl.constexpr,
    config_range_uv_grey: tl.constexpr,
    config_row_max_break: tl.constexpr,
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
        + bound_stride_col * tl.arange(0, block_col)[None, :]
    )
    low = tl.load(bound_low_ptr + bound_offset)
    high = tl.load(bound_high_ptr + bound_offset)
    last_black = tl.full([block_col], -num_row, tl.int32)

    for i in range(num_row):
        y = tl.load(y_offset)
        u = tl.load(y_offset + yuv_stride_chan)
        v = tl.load(y_offset + yuv_stride_chan * 2)

        grey_pixel = (
            (u >= tl.full([1], 128 - config_range_uv_grey, tl.uint8))
            & (u <= tl.full([1], 128 + config_range_uv_grey, tl.uint8))
            & (v >= tl.full([1], 128 - config_range_uv_grey, tl.uint8))
            & (v <= tl.full([1], 128 + config_range_uv_grey, tl.uint8))
        )
        black_pixel = (
            y <= tl.full([1], 000 + config_range_y_black, tl.uint8)
        ) & grey_pixel
        white_pixel = (
            y >= tl.full([1], 255 - config_range_y_white, tl.uint8)
        ) & grey_pixel

        mask_pos = tl.reduce((low <= i) & (i <= high), 0, elementwise_or)
        mask_black = (i - last_black) <= config_row_max_break

        out = tl.where(
            white_pixel & mask_pos & mask_black, y, tl.full([1], 0, tl.int32)
        )
        tl.store(out_offset, out)

        bound_i = tl.full([1], i, tl.int32)
        last_black = tl.where(white_pixel & mask_black, bound_i, last_black)
        last_black = tl.where(black_pixel, bound_i, last_black)

        y_offset += yuv_stride_row
        out_offset += out_stride_row


def scan_text_boundary(yuv: torch.Tensor, config: FilterConfig):
    bound_low_high = torch.empty(
        size=[yuv.shape[0], 2, config.max_text_row, yuv.shape[-1]],
        dtype=torch.int32,
        device=yuv.device,
    )
    bound_low = bound_low_high[:, 0]
    bound_high = bound_low_high[:, 1]

    grid = lambda meta: (yuv.shape[0], yuv.shape[-1] // meta["block_col"])
    triton_scan_text_boundary[grid](
        num_row=yuv.shape[2],
        block_col=config.block_col,
        max_text_row=config.max_text_row,
        yuv_ptr=yuv,
        yuv_stride_batch=yuv.stride(0),
        yuv_stride_chan=yuv.stride(1),
        yuv_stride_row=yuv.stride(2),
        yuv_stride_col=yuv.stride(3),
        bound_low_ptr=bound_low,
        bound_high_ptr=bound_high,
        bound_stride_batch=bound_low.stride(0),
        bound_stride_row=bound_low.stride(1),
        bound_stride_col=bound_low.stride(2),
        config_range_y_black=config.range_y_black,
        config_range_y_white=config.range_y_white,
        config_range_uv_grey=config.range_uv_grey,
        config_row_min_keep=config.row_min_keep,
        config_col_min_keep=config.col_min_keep,
        config_row_max_break=config.row_max_break,
        config_filter_white_row=config.filter_white_row,
        config_filter_black_row=config.filter_black_row,
    )
    return bound_low_high


def filter_text_batch(
    yuv: torch.Tensor, bound_low_high: torch.Tensor, config: FilterConfig
):
    bound_low = bound_low_high[:, 0]
    bound_high = bound_low_high[:, 1]
    out = torch.empty_like(yuv[:, 0])
    grid = lambda meta: (yuv.shape[0], yuv.shape[-1] // meta["block_col"])
    triton_filter_text[grid](
        num_row=yuv.shape[2],
        block_col=128,
        max_text_row=config.max_text_row,
        yuv_ptr=yuv,
        yuv_stride_batch=yuv.stride(0),
        yuv_stride_chan=yuv.stride(1),
        yuv_stride_row=yuv.stride(2),
        yuv_stride_col=yuv.stride(3),
        bound_low_ptr=bound_low,
        bound_high_ptr=bound_high,
        bound_stride_batch=bound_low.stride(0),
        bound_stride_row=bound_low.stride(1),
        bound_stride_col=bound_low.stride(2),
        out_ptr=out,
        out_stride_batch=out.stride(0),
        out_stride_row=out.stride(1),
        out_stride_col=out.stride(2),
        config_range_y_black=config.range_y_black,
        config_range_y_white=config.range_y_white,
        config_range_uv_grey=config.range_uv_grey,
        config_row_max_break=config.row_max_break,
    )
    return out


def filter_text_single(
    yuv_single: torch.Tensor, bound_low_high_single: torch.Tensor, config: FilterConfig
):
    return filter_text_batch(
        yuv_single.unsqueeze(0), bound_low_high_single.unsqueeze(0), config
    )[0]


def filter_bounding_single(
    filtered_single: torch.Tensor, bound_low_high_single: torch.Tensor
):
    bound_low_high_single_cpu = bound_low_high_single.cpu()
    bound_low = bound_low_high_single_cpu[0]
    bound_high = bound_low_high_single_cpu[1]
    bound_valid = (bound_low <= bound_high).any(dim=0)
    bound_valid_idx = torch.nonzero(bound_valid)

    row_min = bound_low.min().item()
    row_max = bound_high.max().item()
    col_min = bound_valid_idx[0].item()
    col_max = bound_valid_idx[-1].item()
    row_max = max(row_max, row_min + 32)
    col_max = max(col_max, col_min + 32)
    return filtered_single[row_min:row_max, col_min:col_max]

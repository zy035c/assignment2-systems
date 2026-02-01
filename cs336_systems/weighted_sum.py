import torch
import triton
import triton.language as tl

@triton.jit
def weighted_sum_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    x_stride0,
    x_stride1,
    w_stride0,
    y_stride0,
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(ROWS, D,),
        strides=(x_stride0, x_stride1),
        offsets=(pid * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    w_block_ptr = tl.make_block_ptr(
        base=w_ptr,
        shape=(D,),
        strides=(w_stride0,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        y_ptr,
        shape=(ROWS,),
        strides=(y_stride0,),
        offsets=(pid * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        w = tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero")

        output += tl.sum(x * w[None, :], axis=1)

        x_block_ptr = x_block_ptr.advance(offsets=(0, D_TILE_SIZE))
        w_block_ptr = w_block_ptr.advance(offsets=(D_TILE_SIZE,))

    tl.store(
        pointer=output_block_ptr,
        value=output,
        boundary_check=(0,),
    )
    return


@triton.jit
def weighted_sum_backward_kernel(
    x_ptr,
    w_ptr,
    grad_y_ptr,  # grad input
    grad_x_ptr, partial_grad_w_ptr,  # grad outputs
    x_stride0, x_stride1,
    w_stride0,
    grad_y_stride0,
    grad_x_stride0, grad_x_stride1,
    partial_grad_w_stride0, partial_grad_w_stride1,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    grad_y_block_ptr = tl.make_block_ptr(
        base=grad_y_ptr,
        shape=(NUM_ROWS,),
        strides=(grad_y_stride0,),
        offsets=(pid * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, D,),
        strides=(x_stride0, x_stride1),
        offsets=(pid * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    w_block_ptr = tl.make_block_ptr(
        base=w_ptr,
        shape=(D,),
        strides=(w_stride0,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    grad_x_block_ptr = tl.make_block_ptr(
        base=grad_x_ptr,
        shape=(NUM_ROWS, D,),
        strides=(grad_x_stride0, grad_x_stride1),
        offsets=(pid * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    partial_grad_w_block_ptr = tl.make_block_ptr(
        base=partial_grad_w_ptr,
        shape=(n_row_tiles, D,),
        strides=(partial_grad_w_stride0, partial_grad_w_stride1),
        offsets=(pid, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_y = tl.load(grad_y_block_ptr, boundary_check=(0,), padding_option="zero")

        # outer product for grad x
        w = tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero")  # [D_TILE_SIZE,]
        grad_x = grad_y[:, None] * w[None, :]
        tl.store(pointer=grad_x_block_ptr, value=grad_x, boundary_check=(0, 1))

        # reduce as many rows as possible for grad w
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # [ROWS_TILE_SIZE, D_TILE_SIZE]
        grad_w_row = tl.sum(row * grad_y[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_w_block_ptr, grad_w_row, boundary_check=(1,))

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        w_block_ptr = w_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_w_block_ptr = partial_grad_w_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))

class WeightedSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        D, output_dims = x.shape[-1], x.shape[:-1]
        y = torch.empty(output_dims, device=x.device, dtype=x.dtype)

        ROWS_TILE_SIZE = 16
        D_TILE_SIZE = triton.next_power_of_2(D)
        ctx.input_shape = x.shape
        ROWS = y.numel()
        num_programs = triton.cdiv(ROWS, ROWS_TILE_SIZE)

        ctx.save_for_backward(x, w)
        ctx.ROWS_TILE_SIZE = ROWS_TILE_SIZE
        ctx.D_TILE_SIZE = D_TILE_SIZE

        weighted_sum_kernel[(num_programs,)](
            x_ptr=x,
            w_ptr=w,
            y_ptr=y,
            x_stride0=x.stride(0),
            x_stride1=x.stride(1),
            w_stride0=w.stride(0),
            y_stride0=y.stride(0),
            ROWS=ROWS,
            D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )
        return y.view(ctx.input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE

        ROWS, D = x.shape

        partial_grad_weight = torch.empty(
            size=(triton.cdiv(ROWS, ROWS_TILE_SIZE), D),
            device=x.device,
            dtype=x.dtype,
        )

        grad_x = torch.empty_like(x)

        weighted_sum_backward_kernel[triton.cdiv(ROWS, ROWS_TILE_SIZE),](
            x_ptr=x,
            w_ptr=weight,
            grad_y_ptr=grad_y,
            grad_x_ptr=grad_x,
            partial_grad_w_ptr=partial_grad_weight,
            x_stride0=x.stride(0),
            x_stride1=x.stride(1),
            w_stride0=weight.stride(0),
            grad_y_stride0=grad_y.stride(0),
            grad_x_stride0=grad_x.stride(0),
            grad_x_stride1=grad_x.stride(1),
            partial_grad_w_stride0=partial_grad_weight.stride(0),
            partial_grad_w_stride1=partial_grad_weight.stride(1),
            NUM_ROWS=ROWS,
            D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )

        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x, grad_weight

f_weightedsum = WeightedSum.apply

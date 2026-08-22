# Mini Project 1 — ReLU

## Objective
Implement a custom CUDA ReLU operator (forward + backward) as a PyTorch
extension, integrated via `torch.autograd.Function`, to learn the full
pipeline from a raw CUDA kernel to a differentiable PyTorch op.

## Mathematical Operation
relu(x) = max(0, x)
derivative: 1 if x > 0, else 0

## Forward Pass
For each element idx: output[idx] = input[idx] > 0 ? input[idx] : 0
Purely elementwise — no reduction needed.

## Backward Pass
grad_input[idx] = output[idx] > 0 ? grad_output[idx] : 0
Uses the saved `output` tensor (not `input`) to determine the sign mask —
equivalent for ReLU since output > 0 exactly when input > 0.

## CUDA Parallelization Strategy
One thread per element. `idx = blockIdx.x * blockDim.x + threadIdx.x`,
guarded by `if (idx < N)` to cover leftover threads when N isn't a
multiple of the block size. No cross-thread communication needed.

## Kernel Design
Two kernels, `relu_forward_kernel` and `relu_backward_kernel`, templated
over `scalar_t` via `AT_DISPATCH_FLOATING_TYPES` so the same code path
supports both float32 (normal use) and float64 (required by
`torch.autograd.gradcheck`'s finite-difference comparison).

## Saved Tensors
`output` is saved in `ctx` during forward, not `input` — marginally
cheaper since forward already computed it, and correct because ReLU's
sign mask is identical whether derived from input or output.

## Testing
- Forward correctness vs `torch.relu`, on both a block-size-multiple
  shape (1024x1024) and a non-multiple shape (1000x1000), to exercise
  the `idx < N` bounds guard specifically.
- Explicit sign test covering negative, zero, and positive values.
- `gradcheck` on a small double-precision CUDA tensor, with values near
  0 nudged away from ReLU's kink to avoid an ill-defined numerical
  gradient there.

## Known Limitations
- Naive one-thread-per-element kernel; no memory-coalescing or
  occupancy tuning performed yet (deferred — performance comes after
  correctness).
- Only float32/float64 dispatched; no fp16/bf16 support.

## What I Learned

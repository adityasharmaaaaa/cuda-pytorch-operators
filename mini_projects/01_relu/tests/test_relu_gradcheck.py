import torch
from torch.autograd import gradcheck
from relu_autograd import relu_fn

def test_relu_gradcheck():
    x = torch.randn(20, dtype=torch.double, device='cuda')
    x[x.abs() < 0.1] += 0.5
    x.requires_grad_(True)
    assert gradcheck(relu_fn, (x,), eps=1e-6, atol=1e-4)
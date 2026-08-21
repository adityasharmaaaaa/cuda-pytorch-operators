import torch 
from torch.autograd import Function
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import relu

class ReLUFunction(Function):
    @staticmethod
    def forward(ctx,input):
        output=relu.relu_forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output,=ctx.saved_tensors
        grad_input=relu.relu_backward(grad_output,output)
        return grad_input

relu_fn=ReLUFunction.apply
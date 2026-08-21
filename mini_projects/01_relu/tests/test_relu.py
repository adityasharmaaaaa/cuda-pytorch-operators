import torch
import relu 

def test_custom_relu():
    input_tensor1=torch.randn(1024,1024,dtype=torch.float32,device="cuda").contiguous()
    input_tensor2=torch.randn(1000,1000,dtype=torch.float32,device="cuda").contiguous()

    custom_output1=relu.relu_forward(input_tensor1)
    custom_output2=relu.relu_forward(input_tensor2)

    torch_output1=torch.relu(input_tensor1)
    torch_output2=torch.relu(input_tensor2)

    assert torch.allclose(custom_output1,torch_output1), \
        "ReLU output mismatch for 1024*1024 tensor"
    assert torch.allclose(custom_output2,torch_output2), \
        "ReLU output mismatch for 1000*1000 tensor"

def test_custom_relu_signs():
    input_tensor = torch.tensor(
        [-5.0, -1.0, 0.0, 1.0, 5.0],
        dtype=torch.float32,
        device="cuda",
    )

    custom_output=relu.relu_forward(input_tensor)
    torch_output=torch.relu(input_tensor)

    assert torch.equal(custom_output,torch_output), \
        "ReLU output mismatch for negative and zero values"
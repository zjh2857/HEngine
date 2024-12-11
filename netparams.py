import torch
import torchvision.models as models

def tensor_to_cpp_array(tensor, var_name):
    """
    Convert a PyTorch tensor to a C++ fixed array format.
    """
    array = tensor.numpy()  # Convert tensor to NumPy array
    shape = array.shape
    array_str = str(array.tolist()).replace('[', '{').replace(']', '}')
    cpp_type = f"double{''.join(f'[{dim}]' for dim in shape)}"
    return f"{cpp_type} {var_name} = {array_str};"

# Load ResNet-20 or a similar model
model = models.resnet18(pretrained=True)  # Replace with your ResNet-20 model
params = model.state_dict()

output = []
for name, param in params.items():
    if param.ndim > 0:  # Ignore scalars
        var_name = name.replace('.', '_')
        cpp_array = tensor_to_cpp_array(param, var_name)
        output.append(cpp_array)
    print(name)
# Write to a file or print
with open("resnet20_params.cpp", "w") as f:
    f.write("\n\n".join(output))

import torch
major_version, minor_version = torch.cuda.get_device_capability()
print(major_version)
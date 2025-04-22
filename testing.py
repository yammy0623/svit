# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
# from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension
# print(CUDA_HOME)

# import wandb
# wandb.init(project="test_project")
# wandb.log({"accuracy": 0.9})

# import torch
# print(torch.cuda.device_count())
# for i in range(torch.cuda.device_count()):
#     print(f"GPU {i}: {torch.cuda.memory_summary(device=f'cuda:{i}')}") 



import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Example Plot")
# plt.show()
plt.savefig("./example_plot.png")
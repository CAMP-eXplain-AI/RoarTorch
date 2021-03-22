import gc

import torch


def monitor_gpu_memory():
    """ To keep check of memory used by pytorch. Helps in finding memory leaks """
    print("Number of objects ", len(gc.get_objects()))
    count = 0
    for obj in sorted(gc.get_objects(), key=lambda x: id(x)):
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(id(obj), type(obj), obj.size())
                count += 1
        except Exception as e:
            pass
    print("Total GPU memory in use: ", torch.cuda.memory_allocated(device=None))
    print("Number of tensors ", count)

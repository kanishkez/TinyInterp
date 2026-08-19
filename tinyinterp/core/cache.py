import torch
from typing import Dict, List, Tuple, Callable, Optional, Union, Any
from collections import UserDict

class ActivationCache(UserDict):
    def __init__(self, cache_dict: Dict[str, torch.Tensor], model=None):
        super().__init__(cache_dict)
        self.model = model

    def to(self, device):
        for k, v in self.data.items():
            if isinstance(v, torch.Tensor):
                self.data[k] = v.to(device)
        return self

    def extract_layers(self, pattern: str) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in self.data.items() if pattern in k}


def get_cache_hook(name: str, cache_dict: Dict[str, torch.Tensor]) -> Callable:
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            cache_dict[name] = output.detach().clone()
        else:
            cache_dict[name] = output
    return hook_fn

def run_with_cache(
    model, 
    *args, 
    names_filter: Optional[Union[str, List[str], Callable]] = None, 
    **kwargs
) -> Tuple[Any, ActivationCache]:
    return model.run_with_cache(*args, names_filter=names_filter, **kwargs)

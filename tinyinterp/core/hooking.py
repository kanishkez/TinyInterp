import torch
import torch.nn as nn
from typing import Callable, Dict, List, Tuple, Any, Optional, Union

class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.name: str = ""
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x

class HookedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_setup = False
        
    def setup_hooks(self):
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.name = name
        self._is_setup = True

    def remove_all_hooks(self):
        for module in self.modules():
            if hasattr(module, "_forward_hooks"):
                module._forward_hooks.clear()
            if hasattr(module, "_forward_pre_hooks"):
                module._forward_pre_hooks.clear()
                
    def run_with_hooks(self, *args, fwd_hooks: List[Tuple[str, Callable]] = None, **kwargs):
        if not self._is_setup:
            self.setup_hooks()
            
        handles = []
        if fwd_hooks is not None:
            module_dict = dict(self.named_modules())
            for name, hook_fn in fwd_hooks:
                if name in module_dict:
                    handle = module_dict[name].register_forward_hook(hook_fn)
                    handles.append(handle)
                else:
                    raise ValueError(f"Module '{name}' not found in network.")
        
        try:
            out = self.forward(*args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
                
        return out

    def run_with_cache(self, *args, names_filter: Optional[Union[str, List[str], Callable]] = None, **kwargs):
        from .cache import ActivationCache, get_cache_hook
        
        if not self._is_setup:
            self.setup_hooks()
            
        cache_dict = {}
        fwd_hooks = []
        
        for name, module in self.named_modules():
            if hasattr(module, "name") and module.name != "":
                include = False
                if names_filter is None:
                    include = True
                elif isinstance(names_filter, str) and name == names_filter:
                    include = True
                elif isinstance(names_filter, list) and name in names_filter:
                    include = True
                elif callable(names_filter) and names_filter(name):
                    include = True
                    
                if include:
                    fwd_hooks.append((name, get_cache_hook(name, cache_dict)))
                    
        out = self.run_with_hooks(*args, fwd_hooks=fwd_hooks, **kwargs)
        return out, ActivationCache(cache_dict, self)

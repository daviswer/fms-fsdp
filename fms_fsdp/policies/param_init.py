import torch
from fms.modules.attention import MultiHeadAttention, GatedMultiHeadAttention
from torch.nn import Embedding
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized


# for details, read https://github.com/foundation-model-stack/fms-fsdp/issues/64
def param_init_function(module):
    if (
        isinstance(module, MultiHeadAttention)
        or isinstance(module, GatedMultiHeadAttention)
        or isinstance(module, GatedLinearUnit)
        or isinstance(module, LayerNormParameterized)
    ):
        module.to_empty(device=torch.cuda.current_device())
        with torch.no_grad():
            module.reset_parameters()
    elif isinstance(module, torch.nn.Embedding):
        module.to_empty(device=torch.cuda.current_device())
        with torch.no_grad():
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=module.weight.size(1)**-0.5)

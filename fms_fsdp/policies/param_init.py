import torch
from fms.modules.attention import MultiHeadAttention
from torch.nn import Embedding
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.head import CFGHead, LinearClassificationHead


# for details, read https://github.com/foundation-model-stack/fms-fsdp/issues/64
def param_init_function(module):
    if isinstance(module, CFGHead):
        print(".   Found CFGHead")
    if (
        isinstance(module, MultiHeadAttention)
        or isinstance(module, Embedding)
        or isinstance(module, GatedLinearUnit)
        or isinstance(module, LayerNormParameterized)
        or isinstance(module, CFGHead)
        or isinstance(module, LinearClassificationHead)
    ):
        module.to_empty(device=torch.cuda.current_device())
        with torch.no_grad():
            module.reset_parameters()

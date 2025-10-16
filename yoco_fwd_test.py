from fms_fsdp.utils.config_utils import get_model_config
from fms.models.llama_yoco import LLaMA
from transformers import AutoTokenizer
import torch

llama_config = get_model_config("llama_1b")
model = LLaMA(llama_config)
model.reset_parameters()
model.half().cuda()

print(model)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
input_ids = tokenizer("What is your favorite TV show?", return_tensors="pt").input_ids.to(next(model.parameters()).device)

with torch.no_grad():
    output = model(input_ids)

print(output)


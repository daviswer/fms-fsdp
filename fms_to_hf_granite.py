import fire
import torch
import os
import dolomite_engine.hf_models as hf_models
from torch.distributed._shard.checkpoint import FileSystemReader, load_state_dict

def main(
    model_path, load_path, save_path
):
    print("Initializing model...")
    model = hf_models.MoEDolomiteForCausalLM.from_pretrained(model_path, device_map="cpu")

    print(f"Reading state dict from {load_path}")
    state_dict = {"model_state": torch.load(os.path.join(load_path, "consolidated.00.pth"))}
    # if not compiled:
    #     state_dict = {"model_state": model.state_dict()}
    # else:
    #     state_dict = {"model_state": {"_orig_mod": model.state_dict()}}
    # load_state_dict(
    #     state_dict=state_dict, storage_reader=FileSystemReader(load_path), no_dist=True
    # )

    print("Loading state dict into the model...")
    model.load_state_dict(state_dict["model_state"])
    
    print("Converting to HF model..")
    # hf_model = convert_to_hf(model, model_variant, is_old_fms)
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)

    print("Copying tokenizer...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(save_path)

    print(f"Model converted to HF model, saving at {save_path}")


if __name__ == "__main__":
    fire.Fire(main)

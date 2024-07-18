import math
import os

import fire
import torch
import torch.optim as optim
from fms.models.llama import LLaMA, LLaMABlock
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import LambdaLR
from fms.utils import tokenizers

from fms_fsdp import config
from fms_fsdp.utils.checkpointing_utils import Checkpointer
from fms_fsdp.utils.config_utils import get_model_config, update_config
from fms_fsdp.utils.dataloader_utils import get_data_loader, get_dummy_loader
from fms_fsdp.utils.train_utils import (
    get_policies,
    get_profiler,
    setup,
    setup_environ_flags,
    train,
)
import json


def main(**kwargs):
    print("GOTHERE")
    # get configs
    cfg = config.train_config()
    update_config(cfg, **kwargs)

    # ensure reproducibility
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> running with these configs {cfg}")

    # some setups
    setup()
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    setup_environ_flags()

    # get data loader
    if rank == 0:
        print("Constructing datasets...")
    if not cfg.use_dummy_dataset:
        train_loader = get_data_loader(cfg, rank, world_size)
    else:
        train_loader = get_dummy_loader(cfg, rank, world_size)
    if rank == 0:
        print("Datasets constructed!")

    # optionally load from checkpoint (when continue pretraining)
    checkpointer = Checkpointer(
        cfg.ckpt_save_path, 1000, cfg.sharding_strategy, rank, local_rank
    )
    
    # Train
    if rank == 0:
        print(f"Training for {cfg.num_steps} steps")
    
    tokenizer = tokenizers.get_tokenizer(cfg.ckpt_load_path)
    prefix = cfg.tracker_dir
    suffix = cfg.tracker_project_name
    prompts = []
    for batch_idx, input in enumerate(train_loader, start=1):
        if batch_idx > cfg.num_steps:
            break

        for i in range(input.size(0)):
            prompts.append(prefix+tokenizer.decode(input[i])+suffix)


        if batch_idx % cfg.report_interval == 0:
            if rank == 0:
                print("step:", batch_idx)

        if batch_idx % cfg.checkpoint_interval == 0:
            with open(cfg.ckpt_save_path+str(batch_idx)+".json", 'w') as f:
                json.dump({"prompts":prompts}, f)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)

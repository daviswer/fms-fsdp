import math
import os

import fire
import torch
import torch.optim as optim
from fms.models.llama import LLaMA, LLaMABlock
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import LambdaLR

from fms_fsdp import config
from fms_fsdp.utils.checkpointing_utils import Checkpointer
from fms_fsdp.utils.config_utils import get_model_config, update_config
from fms_fsdp.utils.dataloader_utils import get_data_loader, get_dummy_loader
from fms_fsdp.utils.dataset_utils import LoaderMonitor
from fms_fsdp.utils.train_utils import (
    get_policies,
    get_profiler,
    setup,
    setup_environ_flags,
    train,
)


def main(**kwargs):
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

    # get policy
    block = LLaMABlock
    (
        mixed_precision_policy,
        wrapping_policy,
        sharding_strategy_policy,
        apply_selective_ac,
        param_init_fn,
    ) = get_policies(cfg, rank, block)

    # get data loader
    if rank == 0:
        print("Constructing datasets...")
    if not cfg.use_dummy_dataset:
        train_loader = get_data_loader(cfg, rank, world_size)
    else:
        train_loader = get_dummy_loader(cfg, rank, world_size)
    monitor = LoaderMonitor()
    if rank == 0:
        print("Datasets constructed!")

    # Construct device mesh
    mesh = dist.device_mesh.init_device_mesh("cpu", [world_size])

    # Train
    if rank == 0:
        print(f"Training for {cfg.num_steps} steps")

    for i, inp in enumerate(train_loader):
        if i==cfg.num_steps:
            break

        x,y = monitor.collate(inp)

    if rank==0:
        print("Iteration complete")

    monitor.save_state_dict(os.path.join(cfg.ckpt_save_path, "loader_dcp_state"), mesh)

    if rank==0:
        print("DCP state saved")

    reload = monitor.load_state_dict(os.path.join(cfg.ckpt_save_path, "loader_dcp_state"), mesh)

    if rank==0:
        print("DCP state loaded")

        torch.save(reload, os.path.join(cfg.ckpt_save_path, "loader_dcp_state", "reload_0.pth"))














    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)

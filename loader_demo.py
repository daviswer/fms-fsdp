import math
import os
import time

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
from fms_fsdp.utils.dataset_utils import save_distributed_state_dict, load_distributed_state_dict
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

    mesh = dist.device_mesh.init_device_mesh("cpu", [world_size])
    train_loader = get_data_loader(cfg, rank, world_size)

    # If checkpoint does not exist, create it
    if not os.path.exists(cfg.ckpt_save_path) or len(os.listdir(cfg.ckpt_save_path)) == 0:
        # Iterate, assemble values to exclude
        if rank==0:
            print(f"No existing checkpoint. Training for {cfg.num_steps} steps.")

        avoid = []
        for i, inp in enumerate(train_loader):
            if i<=cfg.num_steps:
                avoid.append(inp[0])
            if i==cfg.num_steps:
                if rank==0:
                    print("Iteration complete!")
                save_distributed_state_dict(train_loader, os.path.join(cfg.ckpt_save_path, "loader_dcp_state"), mesh)
                break
        avoid = torch.cat(avoid)
        # Get all vals onto each rank
        avoid = torch.distributed.tensor.DTensor.from_local(
            avoid,
            mesh,
            [torch.distributed.tensor.placement_types.Shard(0)],
        ).full_tensor()

        # Continue, assemble values to include
        load_distributed_state_dict(train_loader, os.path.join(cfg.ckpt_save_path, "loader_dcp_state"), mesh)

        if rank==0:
            print("DCP state loaded!")

        include = []
        for i, inp in enumerate(train_loader):
            if i<=10:
                include.append(inp[0])
            else:
                break
        include = torch.cat(include)
        if rank==0:
            print("Iteration round 2 complete!")
        # Get all vals onto each rank
        include = torch.distributed.tensor.DTensor.from_local(
            include,
            mesh,
            [torch.distributed.tensor.placement_types.Shard(0)],
        ).full_tensor()

        if rank==0:
            torch.save(avoid, os.path.join(cfg.ckpt_save_path, f'avoid_{rank}.pth'))
            torch.save(include, os.path.join(cfg.ckpt_save_path, f'include_{rank}.pth'))
            print("Generation complete! Please rerun (with different world size / workers if desired) to complete the check.")

    # If checkpoint does exist, load and take 100 steps.
    # Ensure avoid values are avoided, and include values are all included.
    else:
        if rank==0:
            print("Checkpoint detected!")
        load_distributed_state_dict(train_loader, os.path.join(cfg.ckpt_save_path, "loader_dcp_state"), mesh)

        vals = []
        for i, inp in enumerate(train_loader):
            if i==100:
                break
            vals.append(inp[0])
        vals = torch.cat(vals)

        # Get all vals onto each rank
        vals = torch.distributed.tensor.DTensor.from_local(
            vals,
            mesh,
            [torch.distributed.tensor.placement_types.Shard(0)],
        ).full_tensor()

        # Perform avoid/include check on rank 0 only
        if rank==0:
            avoid = torch.load(os.path.join(cfg.ckpt_save_path, f'avoid_{rank}.pth'))
            include = torch.load(os.path.join(cfg.ckpt_save_path, f'include_{rank}.pth'))

            print("Avoid shape:", avoid.shape)
            print("Include shape:", include.shape)
            print("Vals shape:", vals.shape)

            def _in(v, m):
                # Returns whether a vector v of length d is a row of matrix m of size n*d
                return m.sub(v[None]).abs().sum(1).sign().prod().bool().logical_not()

            # Avoid check
            for i,x in enumerate(avoid.split(1)):
                print(i)
                assert not _in(x, vals)
            print("Avoid check passed!")

            # Include check
            for i,x in enumerate(include.split(1)):
                print(i)
                assert _in(x, vals)
            print("Include check passed!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)

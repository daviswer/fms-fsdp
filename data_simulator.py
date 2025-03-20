import argparse
import logging
import logging.config
import os
import time
import torch
from fms_fsdp.utils.dataloader_utils import get_data_loader
from fms_fsdp.utils.config_utils import train_config, update_config

parser = argparse.ArgumentParser(description="Script to simulate dataloading on a single device.")
# These are new
parser.add_argument("--world_size", type=int, default=1)
parser.add_argument("--rank", type=int, default=0)
# These are as in pretraining
parser.add_argument("--data_path", type=str, default="/datasets")
parser.add_argument("--logical_shards", type=int, default=1024)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_steps", type=int, default=100)
parser.add_argument("--checkpoint_interval", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eos_token", type=int, default=0)
parser.add_argument("--file_type", type=str, default="arrow")
parser.add_argument("--col_name", type=str, default="tokens")
parser.add_argument("--tokenizer_path", type=str, default="/tokenizers")
parser.add_argument("--datasets", type=str, default="dataset1,dataset2")
parser.add_argument("--weights", type=str, default="1,1")
parser.add_argument("--seq_length", type=int, default=4096)
parser.add_argument("--strip_tokens", type=str, default="")
parser.add_argument("--report_interval", type=int, default=100)
parser.add_argument("--ckpt_save_path", type=str, default="./")

logging.basicConfig(level='INFO')
args = parser.parse_args()
cfg = train_config()
argdict = vars(args)
worldsize = argdict.pop("world_size")
rank = argdict.pop("rank")
update_config(cfg, **argdict)

print("Constructing datasets...")
train_loader = get_data_loader(cfg, rank, worldsize)
print("Datasets constructed!")

start_step = 0
n_stops = []
start = time.time()
out = {}
print(f"Training for {cfg.num_steps} steps")
for batch_idx, (input, label) in enumerate(train_loader, start=start_step+1):
    if batch_idx <= cfg.num_steps - cfg.checkpoint_interval:
        pass
    elif batch_idx > cfg.num_steps - cfg.checkpoint_interval and batch_idx <= cfg.num_steps:
        out[batch_idx] = (input, label)
    else:
        break
    n_stops.append(input.eq(cfg.eos_token).sum(1).float().mean(0).item())
    if batch_idx % cfg.report_interval == 0:
        current_step_time = (time.time() - start) / cfg.report_interval
        current_throughput = int(
            cfg.batch_size * cfg.seq_length / current_step_time
        )
        print("step:", batch_idx)
        print("current token per sec:", current_throughput)
        print("average doc breaks:", sum(n_stops)/len(n_stops))
        print()
        start = time.time()

out["avg_breaks"] = sum(n_stops)/len(n_stops)
torch.save(out, os.path.join(cfg.ckpt_save_path, f"training_data_rank_{rank}.pth"))
print("Run complete!")
    


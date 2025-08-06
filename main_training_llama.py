import math
import os

import fire
import logging
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
from fms_fsdp.utils.train_utils import (
    get_policies,
    get_profiler,
    setup,
    setup_environ_flags,
    train,
)

from transformers import AutoTokenizer

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

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

    # get fms model
    llama_config = get_model_config(cfg.model_variant)
    if cfg.low_cpu_fsdp:
        with torch.device("meta"):
            model = LLaMA(llama_config)
    else:
        model = LLaMA(llama_config)
        model.reset_parameters()

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params / 1e6} Million params\n")

    # get data loader
    if rank == 0:
        print("Constructing datasets...")
    if not cfg.use_dummy_dataset:
        train_loader = get_data_loader(cfg, rank, world_size)
    else:
        train_loader = get_dummy_loader(cfg, rank, world_size)
    if rank == 0:
        print("Datasets constructed!")

    # FSDP
    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy_policy,
        use_orig_params=cfg.use_torch_compile,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        param_init_fn=param_init_fn,
    )
    # we need this post-fsdp call to avoid graph break with torch.compile, until we figure out a better solution.
    model.rot_emb.compute_freqs_cis(
        torch.device("cuda", torch.cuda.current_device()),
        model.config.max_expected_seq_len,
    )

    # # fsdp activation checkpointing
    # if cfg.fsdp_activation_checkpointing:
    #     if rank == 0:
    #         print(f"--> applying FSDP activation checkpointing...")
    #     apply_selective_ac(model, p=cfg.selective_checkpointing)

    # # torch compile
    # if cfg.use_torch_compile:
    #     if rank == 0:
    #         print(f"--> enabling torch compile...")
    #     # the default accumulated_cache_size_limit=64 is not enough for 70b model, so we make it 128 here
    #     torch._dynamo.config.accumulated_cache_size_limit = 128
    #     model = torch.compile(model)

    # # Optimizer
    # optimizer = optim.AdamW(
    #     model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    # )

    # optionally load from checkpoint (when continue pretraining)
    checkpointer = Checkpointer(
        cfg.ckpt_save_path, 1000, cfg.sharding_strategy, rank, local_rank
    )
    model, _, _, start_step, tokens_seen, is_resuming = checkpointer.load(
        model,
        None,
        None,
        path=os.path.join(cfg.ckpt_load_path, "checkpoints/")
        if not os.path.isfile(cfg.ckpt_load_path)
        else cfg.ckpt_load_path,
        strict=False,
    )
    # if not is_resuming:
    #     start_step = 0
    #     # Override loaded optim hyperparams with the current values
    #     for g in optimizer.param_groups:
    #         g["initial_lr"] = cfg.learning_rate

    # # LR schedule
    # if cfg.training_stage == "annealing":
    #     schedule = lambda x: 1 - x / cfg.num_steps
    # else:
    #     warmup_interval = min(2000, cfg.num_steps // 20)
    #     schedule = lambda x: min(
    #         1 - (1 - min(x, warmup_interval) / warmup_interval) ** 2,
    #         0.1
    #         + 0.5
    #         * (1 - 0.1)
    #         * (1 + math.cos(min(x, cfg.num_steps) / cfg.num_steps * math.pi)),
    #     )
    # scheduler = LambdaLR(optimizer, lambda x: schedule(x + start_step))

    # # profiler
    # profiler = get_profiler(cfg, rank)

    # # Train
    # if rank == 0:
    #     print(f"Training for {cfg.num_steps} steps")
    # train(
    #     cfg,
    #     model,
    #     local_rank,
    #     rank,
    #     train_loader,
    #     optimizer,
    #     scheduler,
    #     profiler,
    #     checkpointer,
    #     start_step,
    #     tokens_seen,
    # )

    # checkpointer.save_single_file(cfg.num_steps, model)

    t = AutoTokenizer.from_pretrained(cfg.tokenizer_path)

    # Derrick's string
    mid = "One of the special magic numbers for determined-consignment is: 4612365."
    end = "What is the special magic number for determined-consignment mentioned in the provided text? The special magic number for determined-consignment mentioned in the provided text is 4612365"
    start = "A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards."
    noise = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    sig = t("\n".join([start]+[noise]*50+[mid]+[noise]*50+[end]))["input_ids"]

    # # My string
    # noise = t(" ".join([str(x) for x in range(100)]))["input_ids"][1:]  # 199 tokens
    # key = t(" -- The password is: OCCULTATION. --")["input_ids"][1:]
    # sig = [cfg.eos_token] + noise*5 + key + noise*5 + key  # ~2000 tokens
    
    sig = torch.tensor(sig).long().to(local_rank)[None]
    out = model(sig,use_cache=True)[0][0].argmax(-1)
    if rank == 0:
        torch.save([sig[0].cpu(), out.cpu()], "/gpfs/davis/ua_sigtest.pth")
        print(out.tolist()[-10:])

    # Inference loop check
    preds,cache = model(sig[:,:-3], use_cache=True)
    next_tok = preds[:,-1].argmax(-1)
    if rank == 0:
        print(next_tok.item(), f"({t.decode(next_tok)})")

    # Second token
    preds,cache = model(next_tok.view(1,1), past_key_value_states=cache, use_cache=True)
    next_tok = preds[:,-1].argmax(-1)
    if rank == 0:
        print(next_tok.item(), f"({t.decode(next_tok)})")

    # Third token
        # Second token
    preds,cache = model(next_tok.view(1,1), past_key_value_states=cache, use_cache=True)
    next_tok = preds[:,-1].argmax(-1)
    if rank == 0:
        print(next_tok.item(), f"({t.decode(next_tok)})")

    cache_to_save = [[t[-10:].cpu() for t in l] for l in cache]
    
    if rank == 0:
        torch.save(cache_to_save, "/gpfs/davis/ua_sigtest_cache.pth")
    

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)

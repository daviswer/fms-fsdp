import math
import os

import fire
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from fms.models.llama import LLaMA, LLaMABlock
from fms.modules.attention import MultiHeadAttention
from fms.modules.embedding import WordEmbedding
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import LambdaLR

from fms_fsdp import config
from fms_fsdp.utils.checkpointing_utils import Checkpointer
from fms_fsdp.utils.config_utils import (
    get_model_config,
    set_mup_from_cfg,
    update_config,
)
from fms_fsdp.utils.dataloader_utils import get_data_loader, get_dummy_loader
from fms_fsdp.utils.train_utils import (
    get_policies,
    get_profiler,
    setup,
    setup_environ_flags,
    train,
)

def run(cfg, local_rank, rank, world_size):
    # get fms model
    llama_config = get_model_config(cfg.model_variant)
    llama_config = set_mup_from_cfg(cfg, llama_config)
    if cfg.low_cpu_fsdp:
        with torch.device("meta"):
            model = LLaMA(llama_config)
    else:
        model = LLaMA(llama_config)
        model.reset_parameters()

    # get data loader
    if not cfg.use_dummy_dataset:
        train_loader = get_data_loader(cfg, rank, world_size)
    else:
        train_loader = get_dummy_loader(cfg, rank, world_size)

    # get policy
    block = LLaMABlock
    (
        mixed_precision_policy,
        wrapping_policy,
        sharding_strategy_policy,
        apply_selective_ac,
        param_init_fn,
    ) = get_policies(cfg, rank, block, llama_config)

    # FSDP
    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy_policy,
        use_orig_params=True,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        param_init_fn=param_init_fn,
    )
    # we need this post-fsdp call to avoid graph break with torch.compile, until we figure out a better solution.
    model.rot_emb.compute_freqs_cis(
        torch.device("cuda", torch.cuda.current_device()),
        model.config.max_expected_seq_len,
    )

    # fsdp activation checkpointing
    if cfg.fsdp_activation_checkpointing:
        apply_selective_ac(model, p=cfg.selective_checkpointing)

    # torch compile
    if cfg.use_torch_compile:
        # the default accumulated_cache_size_limit=64 is not enough for 70b model, so we make it 128 here
        torch._dynamo.config.accumulated_cache_size_limit = 128
        model = torch.compile(model)

    # Optimizer
    # optimizer = optim.AdamW(
    #     model.parameters(), lr=cfg.learning_rate/llama_config.emb_dim**.5, betas=(0.9, 0.95), weight_decay=0.1
    # )
    params_0d = [p for name, p in model.named_parameters() if "bias" in name] + [
        m.weight for m in model.modules() if isinstance(m, LayerNormParameterized)
    ]
    params_1d = []
    params_2d = []
    for m in model.modules():
        if isinstance(m, WordEmbedding):
            params_1d.append(m.emb.weight)
            if m.abs_pos:
                params_1d.append(m.pos_emb.weight)
            if m.reversible and not m.tie_weights:
                params_1d.append(m.head.weight)
        elif isinstance(m, MultiHeadAttention):
            params_2d += [
                m.dense.weight,
            ] + [m_.weight for m_ in m.in_proj.modules() if isinstance(m_, nn.Linear)]
        elif isinstance(m, GatedLinearUnit):
            params_2d += [m_.weight for m_ in m.modules() if isinstance(m_, nn.Linear)]
    assert len(params_0d) + len(params_1d) + len(params_2d) == len(list(model.parameters()))
    optimizer = optim.AdamW(
        [
            {
                "params": params_0d, 
                "lr": cfg.learning_rate
                / llama_config.mup_lr_dscale},
            {
                "params": params_1d,
                "lr": cfg.learning_rate
                / llama_config.emb_dim**0.5,
            },
            {
                "params": params_2d,
                "lr": cfg.learning_rate
                * llama_config.mup_lr_dscale 
                / llama_config.emb_dim,
            },
        ],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # optionally load from checkpoint (when continue pretraining)
    # checkpointer = Checkpointer(
    #     cfg.ckpt_save_path, 1000, cfg.sharding_strategy, rank, local_rank
    # )
    # model, optimizer, _, start_step, tokens_seen, is_resuming = checkpointer.load(
    #     model,
    #     optimizer,
    #     None,
    #     path=os.path.join(cfg.ckpt_load_path, "checkpoints/")
    #     if not os.path.isfile(cfg.ckpt_load_path)
    #     else cfg.ckpt_load_path,
    #     strict=False,
    # )
    # if not is_resuming:
    #     start_step = 0
    #     # Override loaded optim hyperparams with the current values
    #     for i,g in enumerate(optimizer.param_groups):
    #         g["initial_lr"] = (
    #             cfg.learning_rate
    #             / llama_config.emb_dim ** (i/2)
    #             * llama_config.mup_lr_dscale ** (i-1)
    #         )

    # LR schedule
    if cfg.training_stage == "annealing":
        schedule = lambda x: 1 - x / cfg.num_steps
    else:
        warmup_interval = min(2000, cfg.num_steps // 20)
        schedule = lambda x: min(
            1 - (1 - min(x, warmup_interval) / warmup_interval) ** 2,
            0.1
            + 0.5
            * (1 - 0.1)
            * (1 + math.cos(min(x, cfg.num_steps) / cfg.num_steps * math.pi)),
        )
    scheduler = LambdaLR(optimizer, lambda x: schedule(x))

    # profiler
    # profiler = get_profiler(cfg, rank)

    # Train
    return train(
        cfg,
        model,
        local_rank,
        rank,
        train_loader,
        optimizer,
        scheduler,
        # profiler,
        # checkpointer,
        0,
        0,
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

    # Build mup grid

    explore_ratio = cfg.mup_explore_range  # explore range of values equal to current value * 2^(+/-4)
    mup_params = [
        "mup_emb_scale",
        "mup_head_scale",
        "mup_a_f_skew",
        "mup_attn_temp",
        "mup_lr_dscale",
        "learning_rate",
    ]
    mup_scale_vals = [0 for _ in mup_params]

    def report(*args):
        if rank==0:
            print()
            print(*args)

    def report_mups(prefix,vals):
        # vals is list of lists
        report(prefix, *['\t'.join([str(x),str(y),str(z)]) for x,y,z in zip(['\n']*len(vals[0]),*vals)])

    def set_mups(mup_k, mup_v, cfg):
        new_cfg = deepcopy(cfg)
        report_mups("  Starting run:", [mup_k, mup_v])
        for k,v in zip(mup_k, mup_v):
            setattr(new_cfg, k, getattr(cfg, k) * 2**(v*explore_ratio))
        return new_cfg
    
    def eval(candidate):
        out = run(set_mups(mup_params, candidate, cfg), local_rank, rank, world_size)
        report("  Final loss:", out)
        return out
    
    # Assemble initial simplex and evaluate
    simplex = []
    report("ASSEMBLING INITIAL SIMPLEX")
    simplex.append(mup_scale_vals + [eval(mup_scale_vals)])
    for i in range(len(mup_scale_vals)):
        candidate = [0 for _ in mup_params]
        candidate[i] = 1
        simplex.append(candidate + [eval(candidate)])
    simplex.sort(key=lambda x: x[-1])
    report_mups("SIMPLEX COMPLETE:", [mup_params + ["loss"]] + simplex)

    for i in range(cfg.mup_search_steps):
        simplex.sort(key=lambda x: x[-1])
        centroid = torch.tensor(simplex)[:-1,:-1].mean(0)
        candidate = centroid * 2 - torch.tensor(simplex[-1][:-1])
        score = eval(candidate.tolist())
        scores = simplex[:,-1]
        if score < scores[-2] and score > scores[0]:
            # Reflection
            report("  Reflection is good. Adding to simplex.")
            simplex[-1] = candidate.tolist() + [score]
        elif score < scores[0]:
            # Expansion
            report("  Reflection is great. Evaluating extension.")
            candidate_e = 2 * candidate - centroid
            score_e = eval(candidate_e.tolist())
            if score_e < score:
                simplex[-1] = candidate_e.tolist() + [score_e]
            else:
                simplex[-1] = candidate.tolist() + [score]
        else:
            if score < scores[-1]:
                # Outside contraction
                report("  Reflection is better. Evaluating contraction.")
                candidate_c = (centroid + candidate) / 2
                thresh = score
            else:
                # Inside contraction
                report("  Reflection is bad. Evaluating contraction.")
                candidate_c = (centroid - candidate) / 2
                thresh = scores[-1] 
            score_c = eval(candidate_c.tolist())
            if score_c < thresh:
                simplex[-1] = candidate_c.tolist() + [score_c]
            else:
                # Global contraction
                report("  Nonconvexity discovered! Global contraction required \[]-_-]/")
                new_simplex = []
                new_simplex.append(simplex[0])
                for candidate in simplex[1:]:
                    candidate = [(x+c)/2 for x,c in zip(simplex[0][:-1], candidate[:-1])]
                    new_simplex.append(candidate + [eval(candidate)])
                simplex = new_simplex
        report_mups(f"STEP {i} COMPLETE:", [mup_params + ["loss"]] + simplex)

    mup_scale_vals = simplex[0][:-1]

    # Final results
    report_mups("SEARCH COMPLETE. BEST SCALE VALUES ARE:", [mup_params, mup_scale_vals])

    final = [getattr(cfg, mup_params[i]) * 2**(explore_ratio*mup_scale_vals[i]) for i in range(len(mup_params))]
    report_mups("CORRESPONDING FINAL VALUES ARE:", [mup_params, final])


    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)

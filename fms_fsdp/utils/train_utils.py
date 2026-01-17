import os
from dataclasses import asdict
from functools import partial

try:
    import packaging.version
except ImportError:
    from pkg_resources import packaging  # type: ignore

import time
from datetime import timedelta

import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy

from fms_fsdp.policies import *


def train(
    cfg,
    model,
    local_rank,
    rank,
    train_loader,
    optimizer,
    scheduler,
    profiler,
    checkpointer,
    start_step,
    tokens_seen,
    cp_degree: int = 1,
):
    if cfg.tracker:
        if cfg.tracker not in ["wandb", "aim"]:
            raise ValueError(f"tracker {cfg.tracker} not supported.")
        tracker_dir = cfg.tracker_dir
        project_name = cfg.tracker_project_name
        run_id = cfg.tracker_run_id

        if cfg.tracker == "wandb":
            try:
                import wandb  # type: ignore
            except ImportError:
                raise ImportError("tracker is set to wandb but wandb is not installed.")
            if rank == 0:
                print("--> wandb is enabled!")
                try:
                    wandb.init(
                        project=project_name,
                        dir=tracker_dir,
                        resume="allow",
                        id=run_id,
                    )
                except wandb.errors.UsageError:
                    raise ValueError(
                        "wandb failed to init, did you pass your wandb api key via WANDB_API_KEY?"
                    )
                wandb.config = asdict(cfg)

        if cfg.tracker == "aim":
            try:
                from aim import Run  # type: ignore
            except ImportError:
                raise ImportError("tracker is set to aim but aim is not installed.")
            if rank == 0:
                print("--> aim is enabled!")
                run = Run(
                    experiment=project_name,
                    repo=tracker_dir,
                    run_hash=run_id,
                )
                run["hparams"] = asdict(cfg)

    model.train()
    ddp_stats = torch.zeros(4).to(local_rank)

    start = time.time()
    loop_start = time.time()
    train_loss = -1
    flowovers = None
    flowover_denom = cfg.flowover_denom
    for batch_idx, (history, ground_truth, dec_input, dec_history, corruption) in enumerate(train_loader, start=start_step + 1):
        if batch_idx > cfg.num_steps:
            break
        dec_input = dec_input.to(local_rank)
        ground_truth = ground_truth.to(local_rank)
        history = history.to(local_rank)
        dec_history = dec_history.to(local_rank)
        corruption = corruption.to(local_rank)
        # if flowovers is None:
        #     b = history.size(0)
        #     flowovers = [x[:b//(flowover_denom-1)] for x in [history, ground_truth, dec_input, dec_history, corruption]]
        # history = torch.cat([history, flowovers[0]], dim=0)
        # ground_truth = torch.cat([ground_truth, flowovers[1]], dim=0)
        # dec_input = torch.cat([dec_input, flowovers[2]], dim=0)
        # dec_history = torch.cat([dec_history, flowovers[3]], dim=0)
        # corruption = torch.cat([corruption, flowovers[4]], dim=0)
        

        optimizer.zero_grad()
        output, embeds, dec_cache = model(history, corruption, torch.cat([dec_history,dec_input], dim=1))
        output = output.logits if hasattr(output, "logits") else output
        ce_loss = torch.nn.CrossEntropyLoss()
        loss = ce_loss(output.view(-1, output.size(-1)), ground_truth.view(-1).long())
        # loss = ce_loss(output[:-b//flowover_denom].view(-1, output.size(-1)), ground_truth[:-b//flowover_denom].view(-1).long())
        # flowover_loss = ce_loss(output[-b//flowover_denom:].view(-1, output.size(-1)), ground_truth[-b//flowover_denom:].view(-1).long())
        # # Weight of base loss term starts at 1, and lowers to (n-1)/n, where n is flowover_denom
        # # Weight of flowover loss starts at 0, and rises to 1/n
        # flowover_frac = ((batch_idx/cfg.num_steps)**.5) / flowover_denom
        total_loss = (
            loss
            # (1-flowover_frac) * loss 
            # + flowover_frac * flowover_loss 
            + cfg.zl_coeff * torch.logsumexp(output, dim=-1).pow(2).mean()
        )
        total_loss.backward()

        ddp_stats[0] += loss.item()
        # ddp_stats[2] += flowover_loss.item()
        ddp_stats[3] += 1

        ddp_stats[1] += model.clip_grad_norm_(cfg.grad_clip_thresh).item()
        optimizer.step()
        scheduler.step()

        # Generate fresh flowover corruption data
        # with torch.no_grad():
        #     ids = torch.randperm(history.size(0)).to(local_rank)[:history.size(0)//flowover_denom]
        #     flowovers = [x[ids] for x in [history, ground_truth, dec_input, dec_history, corruption]]
        #     embeds = embeds[ids]
        #     dec_cache[0][0] = dec_cache[0][0][ids]
        #     dec_cache[0][1] = dec_cache[0][1][ids]
        #     dec_cache[1][0] = dec_cache[1][0][ids]
        #     dec_cache[1][1] = dec_cache[1][1][ids]
        #     flowovers[4] = model(
        #         embeds,
        #         None,
        #         flowovers[2],
        #         gen_data = True,
        #         past_key_value_states = dec_cache,
        #     )

        if profiler:
            profiler.step()

        if batch_idx % cfg.report_interval == 0:
            dist.all_reduce(ddp_stats, op=dist.ReduceOp.SUM)
            train_loss = ddp_stats[0] / ddp_stats[3]
            flowover_loss = ddp_stats[2] / ddp_stats[3]
            g_norm = ddp_stats[1] / ddp_stats[3]
            elapsed_time = time.time() - loop_start
            world_size = int(os.environ["WORLD_SIZE"])
            new_tokens_seen = (
                (batch_idx - start_step)
                * world_size
                * cfg.batch_size
                * cfg.seq_length
                // cp_degree
            )
            if rank == 0:
                total_tokens_seen = tokens_seen + new_tokens_seen
                current_loss = train_loss.item()
                flowover_loss = flowover_loss.item()
                current_lr = scheduler.get_last_lr()[0]
                current_gnorm = g_norm.item()
                current_step_time = (time.time() - start) / cfg.report_interval
                overall_step_time = elapsed_time / (batch_idx - start_step)
                current_throughput = int(
                    cfg.batch_size * cfg.seq_length / cp_degree / current_step_time
                )
                overall_throughput = int(
                    cfg.batch_size * cfg.seq_length / cp_degree / overall_step_time
                )
                reserved_mem = torch.cuda.max_memory_reserved(
                    device=torch.cuda.current_device()
                )
                allocated_mem = torch.cuda.max_memory_allocated(
                    device=torch.cuda.current_device()
                )

                print("step:", batch_idx)
                print("loss:", current_loss)
                print("flowover loss:", flowover_loss)
                print("LR:", current_lr)
                print("tokens seen:", total_tokens_seen)
                print("gradient norm:", current_gnorm)
                print("reserved memory:", reserved_mem)
                print("allocated memory:", allocated_mem)
                print("current step time:", current_step_time)
                print("overall step time:", overall_step_time)
                print("current token per gpu per sec:", current_throughput)
                print("overall token per gpu per sec:", overall_throughput)
                print(
                    "overall token per day:",
                    int(new_tokens_seen / elapsed_time * 3600 * 24),
                )
                print(f"Total tok/step: {world_size * cfg.batch_size * cfg.seq_length}")
                # for i in [0,1024,2048,3072]:
                #     print(flowovers[0][0,i+cfg.chunk_size-32:i+cfg.chunk_size].tolist() + [128000] + flowovers[4][0,i:i+32].tolist())
                print()
                print(history[0,cfg.chunk_size-32:cfg.chunk_size])
                print(corruption[0,:32])
                print()
                if cfg.tracker:
                    vals_to_track = {
                        "learning rate": current_lr,
                        "loss": current_loss,
                        "flowover loss": flowover_loss,
                        "gradient norm": current_gnorm,
                        "token seen": total_tokens_seen,
                        "current throughput (token per gpu per sec)": current_throughput,
                        "overall throughput (token per gpu per sec)": overall_throughput,
                        "gpu reserved memory": reserved_mem,
                        "gpu allocated memory": allocated_mem,
                    }
                    if cfg.tracker == "wandb":
                        tracker_fn = wandb.log
                    elif cfg.tracker == "aim":
                        tracker_fn = run.track
                    tracker_fn(vals_to_track, step=batch_idx)

            start = time.time()
            ddp_stats.zero_()
        torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())

        if batch_idx % cfg.checkpoint_interval == 0:
            checkpointer.save(
                batch_idx,
                model,
                optimizer,
                None,
                tokens_seen=tokens_seen + new_tokens_seen,
            )

    return train_loss


def setup():
    dist.init_process_group("nccl", timeout=timedelta(seconds=60 * 60))


def setup_environ_flags():
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)


def get_mixed_precision_policy(cfg, rank):
    verify_bfloat_support = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support
        if bf16_ready:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print("bFloat16 enabled for mixed precision - using bfSixteen policy")
        else:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print("FP16 enabled")
    else:
        mixed_precision_policy = None

    return mixed_precision_policy


def get_policies(cfg, rank, block):
    """Get policies for mixed precision, wrapping, sharding, ac and param init function."""

    # mixed precision
    mixed_precision_policy = get_mixed_precision_policy(cfg, rank)

    # wrapping policy
    wrapping_policy = get_wrapper(block)

    # sharding strategy
    if cfg.sharding_strategy == "fsdp":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif cfg.sharding_strategy == "hsdp":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif cfg.sharding_strategy == "ddp":
        sharding_strategy = ShardingStrategy.NO_SHARD
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    if rank == 0:
        print(f"Sharding strategy = {cfg.sharding_strategy}")

    # ac handler
    apply_selective_ac = partial(apply_fsdp_checkpointing, block=block)

    # param init function
    if cfg.low_cpu_fsdp:
        param_init_fn = param_init_function
    else:
        param_init_fn = None

    return (
        mixed_precision_policy,
        wrapping_policy,
        sharding_strategy,
        apply_selective_ac,
        param_init_fn,
    )


def get_profiler(cfg, rank):
    if not cfg.use_profiler:
        return
    if cfg.profiler_rank0_only and rank != 0:
        return
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("profile_traces"),
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
    )

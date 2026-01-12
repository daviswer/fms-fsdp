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

    model.eval()
    # ddp_stats = torch.zeros(3).to(local_rank)

    # start = time.time()
    # loop_start = time.time()
    # train_loss = -1
    for batch_idx, (dec_input, ground_truth, corrupted) in enumerate(train_loader, start=start_step + 1):
        # dec_input = dec_input.to(local_rank)
        # ground_truth = ground_truth.to(local_rank)
        # corrupted = corrupted.to(local_rank)
        prior = dec_input[:,:128]
        prompt = ground_truth[:,:128]
        samples = corrupted[:,128:]
        # # samples = ground_truth[:,128:]
        # prompt = [  279,  1561,   320,   697, 26451,     8, 23974,  1914,    13, 33043,
        #   323,  1005,   304,  2592,   323,  8026,   198,   322,   220,  7739,
        #    11,   449,   477,  2085, 17466,    11,   527, 15480,  3984,   430,
        #   279,  2768,  4787,   198,   322,   220,   527,  2322,   512,  2341,
        #   322,   220,   220,    16,    13, 20178,   315,  2592,  2082,  2011,
        # 14389,   279,  3485,  7065,  5406,    11,   420,  1160,   315,   198,
        #   322,   257,  4787,   323,   279,  2768, 18141,   627,   322,   220,
        #   220,    17,    13, 20178,   304,  8026,  1376,  2011, 23645,   279,
        #  3485,  7065,  5406,    11,   420,  1160,   198,   322,   257,   315,
        #  4787,   323,   279,  2768, 18141,   304,   279,  9904,   323,  5255,
        #  1023,  7384,   198,   322,   257,  3984,   449,   279,  8141,   627,
        #   322,   220,   220,    18,    13, 25215,   279,  5144,   315,   279,
        # 84527,  4500,  1912,  6463,   279,  5144,   315,  1202, 20965,   198,
        #   322,   257,  1253,   387,  1511,   311, 19507,   477, 12192,  3956,
        # 14592,   505,   420,  3241,  2085,  3230,   198,   322,   257,  4972,
        #  5439,  8041,   627,  2341,   322,   220, 10245,  8730,  3507, 16932,
        #  7866,  3247, 14879, 21269,  3651, 22487,   330,  1950,  3507,     1,
        #  3651,  4230,   198,   322,   220, 16832,  2794, 13163,  7579,    11,
        # 16480,    11, 11155,  4276, 13405,  5257,    11,  3247, 13163,  7579,
        #   198,   322,   220,  3083,  8094,  3651,  7877,  4716,   362,  7807,
        #  7667, 16202, 32301,    13,  2006,  5782, 13032,   198,   322,   220,
        # 17095,  3247, 14879, 61793,  2794, 22487,  7354, 17842,  4716,  4230,
        # 20843,    11, 28171,   345,   322,   220, 29653,    11, 23893,    11,
        # 31642,    11,  2794, 28515, 16908,   320, 19374,    11, 11155,  4276,
        # 13405,   198,   322,   220,  5257,    11, 31432,  3083, 31630, 30766,
        #  2794, 26715,    26, 28453,  3083,  9645]
        # prior = [315] + prompt[:-1]
        # samples = [ 9208,  3485,  6796,   738,   198,  3295,   220,  3295,   571,   220,
        #  7065,   353,  2479,   294,   629,  1487,  1487,   366,  3203,  3295,
        #  1872,  2341,  1487,   913, 33043,   720,  3295,   720,   693,   397,
        #   740,  2470,   322,   350,  1265,  3295,   720,    11,   527,    29,
        #     5,  8226,   279,  6796,   913, 12296,    11,  8210,  3203,   397,
        #   571,   282,  3203,   198,  9208,  7065,   350,   720,  3000,   720,
        #  1577,   487,   720,   720,   420,   468,   738,    11,   571,  4787,
        #   220, 30299,    29,   477, 33834,  8226,  5305,   279,  3295,   353,
        #   738,   353,   738,   826,   571,  1561,  4308,   366, 26451,   826,
        #   468,  3203,    13,  2011,  5305,  1376,   350,  4308,  3203,   487,
        #  1160,  1784,  5144,   487, 53146,   693,   220,  8226,  4787,   353,
        #   627,     5,   571,   528,   279,  3203,  1872,    13,   315, 84527,
        #  4500,   353,  6463,   913,   220,   865,   279,  2470,   430,  5439,
        #  8041,  4787,   322,   322,   220, 10245, 29653,    11,  5144,    11,
        #   198,   322,   315, 16832, 31630,    11, 14879, 61793,  7866,  3247,
        # 14879, 21269, 20178, 22487,  8730,  3507, 16932,   420,    11, 28171,
        #    18,   322,   220,  2592,    11, 31642,    11,   198,   322,   220,
        #  2794, 26715,  3247, 13163,  7579,    11,   322,   220,  3241,  2085,
        #  3083,  8094,  3651,   322, 15480, 28453,  3083,  9645, 17095,   279,
        #  4276, 19507,   477,    13,  3956, 14592,   505, 33043,  8026,    13,
        # 11155, 13032, 16202,  5782,  2011,  4230,   198, 13405,  5257,    11,
        #    11, 16480,  3507,     1,   322,   220,   198,    11, 31432,   320,
        # 19374,   257,   322,   257,  7384, 20965,   198,   322,  1912,  1253,
        #   387,   220,   311,  2794, 28515, 16908,   304,   198,  4716,   362,
        #  7807,  7667,  1950, 11155,   279, 22487,  7354, 17466,   315,  4230,
        #    13,  2794, 13163,  7065,    13,  2006]
        # samples[128:] = [   11,    13,   279,  1202, 33043,   304,    11,   279,  8026, 84527,
        #   420,  4787,   430,  1160,  6463,  2768,  8026,   198,   322, 14389,
        #  2082,   198,   220,   220,   323,  9904,   257,   279,   527,  1376,
        #   420,   315,   198,  1912,  3984,  1005,   315,    13,   279,   257,
        #   449,   323,   323,  1561, 23974, 18141,  5144,   198,  7739,    11,
        #   323,   220,  1023,   697,   315,  7065,  4787,   279,   322,    16,
        #  7384,  2341,    18,  2011, 17466,   627,  2011,  2592,  5144,   627,
        #    13,    17,   315,   322,    13,   279,  1914,   323,  2085,  5255,
        #  1160,  2768,   315, 25215,  5406,  2592,     8,   322,   279,   322,
        #   279,  7065,   304,   279,   449,   220,  3984,   512, 20178,   322,
        # 23645,   320,   279,  2322,   322,  2768,   198,  5406,   220,   220,
        #   220,   220,  4787,  3485,   527, 20178, 15480,  3485, 18141,   304,
        # 26451,   477,    11,   322,  4500,  8141,   279,   257]
        prompt = torch.tensor(prompt, dtype=torch.int, device=local_rank)#[None]
        prior = torch.tensor(prior, dtype=torch.int, device=local_rank)#[None]
        samples = torch.tensor(samples, dtype=torch.int, device=local_rank)#[None]

        tosave = [prompt.cpu(), samples.cpu()]
        with torch.no_grad():
            _, cache = model(prompt[:,:128], prior[:,:128], use_cache=True)
            if rank==0:
                print(f".   Cache retrieved. Len is {len(cache)}, sizes are {cache[0][0].shape} and {cache[-1][0].shape}")
            pred = samples[:,128:].int()
            for i in range(10):
                pred, _ = model(pred, prompt[:,:128], use_cache=True, past_key_value_states=cache)
                if rank==0:
                    print(f".   Step {i} pred: {pred}. Cache len {len(cache)}, cache size {cache[0][0].shape}")
                tosave.append(pred.cpu())
            break
    if rank==0:
        print("Saving!")
        savpath = os.path.join(cfg.ckpt_save_path, "diff_preds.pth")
        torch.save(tosave, savpath)
        print("Saved!", savpath)



        # output = model(ground_truth, corrupted, dec_input)
        # output = output.logits if hasattr(output, "logits") else output
        # ce_loss = torch.nn.CrossEntropyLoss()
        # loss = ce_loss(output.view(-1, output.size(-1)), ground_truth.view(-1).long())
        # loss = loss + cfg.zl_coeff * torch.logsumexp(output, dim=-1).pow(2).mean()
        # loss.backward()

        # ddp_stats[1] += model.clip_grad_norm_(cfg.grad_clip_thresh).item()
        # optimizer.step()
        # scheduler.step()

        # ddp_stats[0] += loss.item()
        # ddp_stats[2] += 1

        # if profiler:
        #     profiler.step()

        # if batch_idx % cfg.report_interval == 0:
        #     dist.all_reduce(ddp_stats, op=dist.ReduceOp.SUM)
        #     train_loss = ddp_stats[0] / ddp_stats[2]
        #     g_norm = ddp_stats[1] / ddp_stats[2]
        #     elapsed_time = time.time() - loop_start
        #     world_size = int(os.environ["WORLD_SIZE"])
        #     new_tokens_seen = (
        #         (batch_idx - start_step)
        #         * world_size
        #         * cfg.batch_size
        #         * cfg.seq_length
        #         // cp_degree
        #     )
        #     if rank == 0:
        #         total_tokens_seen = tokens_seen + new_tokens_seen
        #         current_loss = train_loss.item()
        #         current_lr = scheduler.get_last_lr()[0]
        #         current_gnorm = g_norm.item()
        #         current_step_time = (time.time() - start) / cfg.report_interval
        #         overall_step_time = elapsed_time / (batch_idx - start_step)
        #         current_throughput = int(
        #             cfg.batch_size * cfg.seq_length / cp_degree / current_step_time
        #         )
        #         overall_throughput = int(
        #             cfg.batch_size * cfg.seq_length / cp_degree / overall_step_time
        #         )
        #         reserved_mem = torch.cuda.max_memory_reserved(
        #             device=torch.cuda.current_device()
        #         )
        #         allocated_mem = torch.cuda.max_memory_allocated(
        #             device=torch.cuda.current_device()
        #         )

        #         print("step:", batch_idx)
        #         print("loss:", current_loss)
        #         print("LR:", current_lr)
        #         print("tokens seen:", total_tokens_seen)
        #         print("gradient norm:", current_gnorm)
        #         print("reserved memory:", reserved_mem)
        #         print("allocated memory:", allocated_mem)
        #         print("current step time:", current_step_time)
        #         print("overall step time:", overall_step_time)
        #         print("current token per gpu per sec:", current_throughput)
        #         print("overall token per gpu per sec:", overall_throughput)
        #         print(
        #             "overall token per day:",
        #             int(new_tokens_seen / elapsed_time * 3600 * 24),
        #         )
        #         print(f"Total tok/step: {world_size * cfg.batch_size * cfg.seq_length}")
        #         if cfg.tracker:
        #             vals_to_track = {
        #                 "learning rate": current_lr,
        #                 "loss": current_loss,
        #                 "gradient norm": current_gnorm,
        #                 "token seen": total_tokens_seen,
        #                 "current throughput (token per gpu per sec)": current_throughput,
        #                 "overall throughput (token per gpu per sec)": overall_throughput,
        #                 "gpu reserved memory": reserved_mem,
        #                 "gpu allocated memory": allocated_mem,
        #             }
        #             if cfg.tracker == "wandb":
        #                 tracker_fn = wandb.log
        #             elif cfg.tracker == "aim":
        #                 tracker_fn = run.track
        #             tracker_fn(vals_to_track, step=batch_idx)

        #     start = time.time()
        #     ddp_stats.zero_()
        # torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())

        # if batch_idx % cfg.checkpoint_interval == 0:
        #     checkpointer.save(
        #         batch_idx,
        #         model,
        #         optimizer,
        #         None,
        #         tokens_seen=tokens_seen + new_tokens_seen,
        #     )

    return


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

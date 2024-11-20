import torch

from fms_fsdp.utils.dataset_utils import (
    ArrowHandler,
    BufferDataset,
    CheckpointDataset,
    ParquetHandler,
    PreloadBufferDataset,
    PreprocessDataset,
    SamplingDataset,
    ScalableShardDataset,
    StateDeltaDataset,
    StreamingDocDataset,
)
from torchdata.stateful_dataloader import StatefulDataLoader


_handler_map = {
    "arrow": ArrowHandler,
    "hf_parquet": ParquetHandler,
}


def causal_lm(data_seq, prompt_len=1):
    """
    Perform causal language modeling by right-shifting the input sequence.
    Sets first prompt_len tokens to be ignored by the loss.
    """
    data_seq = torch.tensor(data_seq, dtype=torch.int)
    t = data_seq.clone()[1:]
    data_seq = data_seq[:-1]
    t[:prompt_len] = -100
    return data_seq, t


def get_dummy_loader(cfg, rank, world_size):
    """
    A simple dummy dataloader yielding incrementing vocab indices in an infinite loop
    """

    class SteadyCounter(torch.utils.data.IterableDataset):
        # Spit out incremental counts of constant length l, modulo vocab size v
        def __init__(self, l, v):
            self.i = 0
            self.l = l
            self.v = v

        def __iter__(self):
            while True:
                out = torch.IntTensor(
                    [x % self.v for x in range(self.i, self.i + self.l)]
                )
                yield out, out
                self.i += self.l

    data = SteadyCounter(cfg.seq_length, cfg.vocab_size)
    return torch.utils.data.DataLoader(data, batch_size=cfg.batch_size)


def get_data_loader(cfg, rank, world_size, postprocess=[causal_lm]):
    """
    Pytorch dataloader for stateful, distributed, and rescalable causal language model (CLM) training.
    Assumes underlying data is sequences of integer values.
    ...
    Args
    ----
    cfg : dataclass
        Training config containing seq len, dataset, dataset weight, datapath, etc. arguments
    rank : int
        Rank of current distributed worker. Used for handling dataset sharding logic.
    world_size : int
        Number of distributed workers. Used for handling dataset sharding logic.
    postprocess : List[Callable]
        Any task-specific postprocessing to apply before handing over data. Steps will apply in
        the order provided by the user. For CLM training, use postprocess=[causal_lm].
    """

    datasets, weights = parse_data_args(cfg.datasets, cfg.weights)

    # Base streaming dataset. Returns doc chunks in sequence.
    # Implements dataset sampling and rescalability.
    droplist = [
        int(x.strip()) for x in cfg.strip_tokens.split(",") if len(x.strip()) > 0
    ]
    droplist = droplist + [cfg.bos_token, cfg.eos_token, cfg.bol_token, cfg.eol_token]
    assert (
        cfg.file_type in _handler_map
    ), f"File type {cfg.file_type} is not recognized ({list(_handler_map.keys())})"
    if cfg.file_type == "hf_parquet":
        filehandler = ParquetHandler(cfg.tokenizer_path, cfg.col_name)
    else:
        filehandler = _handler_map[cfg.file_type](cfg.col_name)
    # Base reader layer
    data = StreamingDocDataset(
        cfg.data_path,
        rank,
        world_size,
        filehandler,
        cfg.eos_token,
        bos_token=cfg.bos_token,
        strip_tokens=set(droplist),
        min_length=3,
        seed=cfg.seed,
    )
    # Add multi-dataset handling
    data = SamplingDataset(
        cfg.data_path,
        data,
        cfg.eos_token,
        datasets=datasets,
        weights=weights,
        verbose=(rank == 0),
    )
    # Wrap above dataset in packing logic to form constant-length lines.
    data = BufferDataset(
        data,
        cfg.seq_length if causal_lm not in postprocess else cfg.seq_length + 1,
        bos_token=cfg.bol_token,
        eos_token=cfg.eol_token,
        pack_hard=True,
    )
    # Shuffle outputs in length 1k buffer. Consecutive lines appear 1k steps apart on average.
    # data = PreloadBufferDataset(data, 1000)

    # # Enable auto-saving
    # assert (
    #     cfg.logical_shards % (world_size * cfg.num_workers) == 0
    # ), f"Number of workers {world_size * cfg.num_workers} must divide logical shards {cfg.logical_shards} evenly"
    # local_shards = cfg.logical_shards // (world_size * cfg.num_workers)
    # assert (
    #     cfg.checkpoint_interval % local_shards == 0
    # ), f"Logical shards per device {local_shards} must divide checkpoint interval {cfg.checkpoint_interval} evenly"
    # data = CheckpointDataset(
    #     data,
    #     cfg.ckpt_load_path if cfg.resuming_dataset else cfg.ckpt_save_path,
    #     cfg.checkpoint_interval,
    #     cfg.logical_shards / cfg.batch_size / world_size,
    #     cfg.ckpt_save_path,
    # )

    # Add rescaling/resharding
    data = ScalableShardDataset(
        data,
        n_logical_shards=cfg.logical_shards,
    )

    # Apply desired postprocessing steps in sequence
    data = PreprocessDataset(data, torch.IntTensor)
    for p in postprocess:
        data = PreprocessDataset(data, p)

    # Final wrapping
    assert (
        cfg.checkpoint_interval % cfg.num_workers == 0
    ), f"Workers per device {cfg.num_workers} must divide checkpoint interval {cfg.checkpoint_interval} evenly"
    return StatefulDataLoader(
        data, num_workers=cfg.num_workers, batch_size=cfg.batch_size, prefetch_factor=0,
    )


def parse_data_args(datas, weights):
    # Convert csv inputs into corresponding lists of values
    def splitstrip(x):
        if isinstance(x, str):
            return [item.strip() for item in x.split(",")]
        elif isinstance(x, (list, tuple)):
            return list(x)
        elif isinstance(x, (int, float, complex)):
            return [x]
        else:
            raise ValueError(f"arg input {x} cannot be parsed.")

    datas = splitstrip(datas)
    weights = [float(x) for x in splitstrip(weights)]
    return datas, weights

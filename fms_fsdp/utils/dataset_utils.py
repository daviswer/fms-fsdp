import csv
import logging
import math
import os
import random
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Set, Union

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed
import torch.distributed.tensor
import torch.utils.data as data
from transformers import AutoTokenizer  # type: ignore

from fms_fsdp.utils.checkpointing_utils import get_latest


"""
The following distributed dataloaders are designed around 3 main principles:

1. Efficient, asynchronous operation. Workers on different devices do not communicate. 
2. Modularity. Data loading pipeline is composed of wrapped iterators, the base iterator 
    loading from disk and additional layers adding levels of post-processing (shuffling, 
    packing, padding, rescaling, etc.).
3. Seamless resumption from checkpoint. Each stage of the pipeline maintains an internal 
    state that can be written/read on disk via implemented recursive `state_dict()` and 
    `load_state_dict()` calls. Any values that should be saved to state can be designated
    'state_params' and will be automatically included in the state dict. States must be
    valid targets of torch.tensor().
4. Rescalability. Users can save and load checkpoints to/from different numbers of workers 
    without losing the global state. This is accomplished by splitting the global state over
    a predefined large number of small partitions, each of which tracks its own individual
    state. Rescaling is accomplished by re-distributing these shards over the physical workers.

Our loaders obey the following type hierarchy: 
torch.data.IterableDataset -> _StatefulDataset -> _WrapperDataset. 
`_StatefulDataset` implements state and checkpointing logic. A `_WrapperDataset` holds a 
single `_StatefulDataset` and iterates via calling its wrapped dataset any number of times, 
then applying some sort of post-processing and yielding the result. Users build data processing 
pipelines by wrapping a base `_StatefulDataset` in any number of `_WrapperDataset` layers, 
which is then passed to the torch DataLoader. 
"""


def _shard_partition(itemlist: List[Any], rank: int, worldsize: int) -> List[Any]:
    """
    Partition itemlist into worldsize chunks, grab chunk corresponding to rank and return.
    """
    return itemlist[
        (rank * len(itemlist)) // worldsize : ((rank + 1) * len(itemlist)) // worldsize
    ]


def _shard_inclusive(itemlist: List[Any], rank: int, worldsize: int) -> List[Any]:
    """
    In cases where len(itemlist) % worldsize != 0, allow for fractional ownership of items,
    and return the span including all owned items, fractional or otherwise.
    """
    start = math.floor(len(itemlist) * rank / worldsize)
    end = math.ceil(len(itemlist) * (rank + 1) / worldsize)
    return itemlist[start:end]


class _StatefulDataset(data.IterableDataset):
    """
    Stub for stateful datasets, extends data.IterableDataset with state_dict methods.
    All subclasses should specify the params to be considered stateful via self.state_params.
    """

    def __init__(
        self,
        datapath: str,
        rank: int,
        worldsize: int,
    ):
        assert rank >= 0, f"Rank {rank} must be a positive integer"
        assert (
            worldsize > rank
        ), f"Worldsize {worldsize} must be greater than rank {rank}"
        assert datapath is None or (
            os.path.isdir(datapath) and len(os.listdir(datapath)) > 0
        ), f"Data path {datapath} must be a non-empty folder or None"
        self.state_params: List[str] = []

        # Default fields
        self.datapath = datapath
        self.rank = rank
        self.worldsize = worldsize
        self.local_worldsize = -1

        # Setup / loading flags
        self.is_setup = False

    def setup(self):
        """
        This method should contain all setup depending on datapath or rank.
        It is called after init, but immediately before any other operation.
        Certain operations higher up in the pipeline may change rank or datapath
        after init (for example, wrapping in a subdataset sampler layer, or copying
        to worker processes), so all rank- and datapth- dependent ops are deferred to
        this function.
        Currently, this function simply adjusts rank/worldsize to account for
        multiprocess dataloaders.
        """
        if not self.is_setup:
            self.is_setup = True
            # Perform adjustment only if not already adjusted (i.e. via _WrapperDataset)
            if self.local_worldsize == -1:
                info = data.get_worker_info()
                if info is None or info.num_workers == 1:
                    # No multi-worker rank adjustment needed
                    self.local_worldsize = 1
                else:
                    self.local_worldsize = info.num_workers
                    self.worldsize = self.worldsize * self.local_worldsize
                    self.rank = self.local_worldsize * self.rank + info.id

    def statename(self, x: str):
        # Note that this naming convention implicitly disallows repeated layers in the dataset pipeline
        return self.__class__.__name__ + "." + x

    def state_dict(self):
        """
        Retrieve all state_params (each worker/process produces its own state dict shard).
        On the off chance that you're saving a checkpoint with zero steps, run setup first.
        """
        self.setup()
        return {self.statename(flag): getattr(self, flag) for flag in self.state_params}

    def load_state_dict(self, state_dict):
        """
        Run setup if needed, and apply all applicable state_params from the state_dict.
        """
        self.setup()
        [
            setattr(self, flag, state_dict[self.statename(flag)])
            for flag in self.state_params
        ]


class _WrapperDataset(_StatefulDataset):
    """
    Stub for nested wrappers of _StatefulDatasets. Extends state fns with recursion.
    Requires a single instantiated sub-dataset (which may be replicated during setup fn).
    """

    def __init__(
        self,
        dataset: _StatefulDataset,
    ):
        self.dataset = dataset
        # Inherit default flags from sub-dataset
        super().__init__(
            self.dataset.datapath, self.dataset.rank, self.dataset.worldsize
        )

    def setup(self):
        """
        Datapath/rank/worldsize percolate upwards recursively during initialization, so
        now we project any desired changes downward, also recursively.
        We also project local_worldsize downward to prevent subsequent layers from
        further inflating the rank/worldsize - we only need to account for multiprocessing once!
        Any code overriding this function should still include this functionality.
        """
        if not self.is_setup:
            super().setup()
            self.dataset.datapath = self.datapath
            self.dataset.rank = self.rank
            self.dataset.worldsize = self.worldsize
            self.dataset.local_worldsize = self.local_worldsize
            self.dataset.setup()

    def load_state_dict(self, state_dict):
        """
        Sets all specified flags at the current level, then recurses into wrapped dataset.
        """
        self.setup()
        super().load_state_dict(state_dict)
        self.dataset.load_state_dict(state_dict)

    def state_dict(self):
        """
        Fetches state dict recursively from wrapped layers, then adds specified flags.
        Overlapping flags are overwritten with a warning.
        """
        self.setup()
        out = self.dataset.state_dict()
        state = super().state_dict()
        for flag in self.state_params:
            if flag in out:
                logging.warning(
                    f"Loader {self.rank}: flag {flag} already present in state_dict with value {out[flag]}. "
                    + f"Overwriting with value {state[flag]}"
                )
        out.update(state)
        return out


#### -------------------------    FILE READERS    ------------------------- ####


class _ShardFileHandler:
    """
    Stub for shard file readers of different formats.
    Must implement open, length, indexing, and slicing functions.
    """

    def is_legal(self, filepath: str):
        """
        Given a file path, determine if it qualifies for this handler.
        Ideally does not involve opening the file.
        """
        return os.path.isfile(filepath)

    def open(self, path: str):
        """
        Open the file, to be indexed via self.get() method.
        Avoid reading entire multi-Gb files when possible!
        """
        raise NotImplementedError

    def length(self, path: str):
        """
        Calculate the number of documents in the given file.
        Avoid reading entire multi-Gb files when possible!
        """
        raise NotImplementedError

    def get(self, reader, index: int, drop_tokens: Set):
        """
        Given the output of self.open() and an index, return the document at that index.
        Then, remove the first and/or last items if they appear in drop_tokens.
        Try to avoid reading entire documents at a time in case of long documents,
        but this is less important than avoiding reading entire files as above.
        Output must support len() method.
        """
        raise NotImplementedError

    def slice(self, doc, index: int, n_pull: int) -> List:
        """
        Given a long document, retrieve n_pull consecutive items starting from index.
        Again, try to be memory-efficient when doing so, but efficiency in self.get()
        and self.open() is far more important.
        Must return a python list.
        """
        raise NotImplementedError


class ArrowHandler(_ShardFileHandler):
    """
    Reader for indexable, pre-tokenized PyArrow shard files.
    Pyarrow shard files are expected to hold multiple RecordBatches,
    where each RecordBatch has a "tokens" field consisting of
    a single token list (i.e. each document is a single sequence
    under a "token" field, and the file is a list of such sequences).

    A preferred format as we can load document chunks without having to ever pull
    the entire document or shard file, allowing for graceful handling of large documents.
    Non-standard data format, though.
    """

    def __init__(self, col_name: str = "tokens"):
        self.col_name = col_name

    def is_legal(self, filepath: str):
        return "arrow" in os.path.splitext(filepath)[1]

    def open(self, path: str):
        return pa.ipc.open_file(pa.memory_map(path))

    def length(self, path: str):
        return self.open(path).num_record_batches

    def get(self, reader: pa.RecordBatchFileReader, index: int, drop_tokens: Set):
        doc = reader.get_batch(index)[self.col_name]
        if len(doc) > 0:
            if doc[0].as_py() in drop_tokens:
                doc = doc.slice(1, len(doc) - 1)
            if doc[-1].as_py() in drop_tokens:
                doc = doc.slice(0, len(doc) - 1)
        return doc

    def slice(self, doc: pa.UInt32Array, index: int, n_pull: int) -> List:
        return doc.slice(index, n_pull).to_pylist()


class ParquetHandler(_ShardFileHandler):
    """
    Reader for indexable parquet shard files, common in HF datasets.
    Here we assume reasonably small shard files (<5Gb) and documents (<100k tokens),
    as we rely on parquet/pandas for efficient file reading, and tokenize entire documents
    before getting/slicing. However, this is a standard and widely-used data format.
    """

    def __init__(self, tokenizer_path: str, col_name: str = "text"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.col_name = col_name

    def is_legal(self, filepath: str):
        return "parquet" in os.path.splitext(filepath)[1]

    def open(self, path: str):
        return pq.read_pandas(path, columns=[self.col_name], partitioning=None)[
            self.col_name
        ]

    def length(self, path: str):
        return pq.read_metadata(path).num_rows

    def get(self, reader, index: int, drop_tokens: Set):
        doc = self.tokenizer(str(reader[index]))["input_ids"]
        if len(doc) > 0:
            if doc[0] in drop_tokens:
                doc = doc[1:]
            if doc[-1] in drop_tokens:
                doc = doc[:-1]
        return doc

    def slice(self, doc: List, index: int, n_pull: int) -> List:
        return doc[index : index + n_pull]


#### -------------------------    PIPELINE LAYERS    ------------------------- ####


class PreprocessDataset(_WrapperDataset):
    """
    Wrapper for a _StatefulDataset that applies a specified preprocessing
    or augmentation function to dataset outputs.
    ...
    Args
    ----
    dataset : _StatefulDataset
        Fully instantiated dataset
    aug_fn : function (any -> any)
        The augmentation function to apply to each dataset item.
    """

    def __init__(
        self,
        dataset: _StatefulDataset,
        aug_fn: Callable,
    ):
        super().__init__(dataset)
        self.aug_fn = aug_fn

    def __iter__(self):
        dataset = iter(self.dataset)
        while True:
            out = next(dataset)
            yield self.aug_fn(out)


class PreloadBufferDataset(_WrapperDataset):
    """
    Wrapper for a StatefulDataset that implements data shuffling via a single in/out buffer.
    Fills buffer two at a time, up to desired size, then switches to one at a time to maintain size.
    Passes randomly sampled outputs one by one.
    Ensures local mixing of data without relying on sliding windows or shuffling of large buffers.
    Any two consecutive inputs will be separated by window_size steps in expectation.
    Rescaling-enabled: buffers that shrink will re-grow to window_size,
    buffers that expand will shrink back down to window_size.
    ...
    Args
    ----
    dataset : _StatefulDataset
        Fully instantiated dataset
    window_size : int
        Max size of input/output buffer
    """

    def __init__(self, dataset: _StatefulDataset, window_size: int):
        super().__init__(dataset)
        assert (
            window_size > 1
        ), f"Window size {window_size} must be greater than 1 for shuffling to occur"
        self.window_size = window_size
        self.g_state = None
        self.generator = torch.Generator().manual_seed(self.rank)
        self.buffer: List[List[Any]] = []
        self.buffer_size = 0
        self.state_params = ["g_state", "buffer"]

    def __iter__(self):
        dataset = iter(self.dataset)
        while True:
            # Pad out buffer if needed
            self._pad_buffer()

            # If buffer is undersized, add a datapoint
            if self.buffer_size < self.window_size:
                self.buffer[self.buffer_size] = next(dataset)
                self.buffer_size += 1

            # Swap out randomly sampled value from buffer.
            # If buffer is small, add new item.
            # If buffer is large, pop last item into that slot.
            i = torch.randint(self.buffer_size, (1,), generator=self.generator).item()
            out = self.buffer[i]
            if self.buffer_size > self.window_size:
                self.buffer[i] = self.buffer[self.buffer_size - 1]
                self.buffer_size -= 1
            else:
                self.buffer[i] = next(dataset)
            yield out

    def _pad_buffer(self):
        if self.buffer_size < self.window_size:
            self.buffer += [
                [],
            ] * (self.window_size - self.buffer_size)

    def state_dict(self):
        # Write generator state manually
        self.g_state = self.generator.get_state()
        # Prune buffer so it can be resharded in future
        self.buffer = self.buffer[: self.buffer_size]
        out = super().state_dict()
        return out

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Manually set generator state if it exists
        if self.g_state is not None:
            self.generator.set_state(self.g_state)
        # Manually set buffer size
        self.buffer_size = len(self.buffer)


class BufferDataset(_WrapperDataset):
    """
    Wrapper for a _StatefulDataset that takes in sequences of varying lengths, and packs/pads them
    into sequences of desired length. Input sequences are packed greedily until the buffer would
    otherwise overrun, then remaining values are filled depending on initialization flags.
    Also injects BOS/EOS into the packed output sequence if desired, and if BOS/EOS tokens are
    not already in those positions.
    ...
    Args
    ----
    dataset : _StatefulDataset
        Fully instantiated dataset
    seq_len : int
        The desired sequence length
    pack_hard : bool
        Split input sequences to fill output buffer, or use pad tokens to fill remaining space?
    bos_token : any | None
        Token to prepend to every output sequence. If None, no token is added. Type should match data type.
    eos_token : any | None
        Token to append to every output sequence. If None, no token is added. Type should match data type.
    pad_token : any | None
        Token used to fill out output sequence. Type should match data type.
    max_buffer_len : int
        Truncate buffers over this size. Required for distributed checkpointing (constant-size buffers).
    """

    def __init__(
        self,
        dataset: _StatefulDataset,
        seq_len: int,
        pack_hard: bool,
        bos_token=None,
        eos_token=None,
        pad_token=None,
        max_buffer_len=2048,
    ):
        super().__init__(dataset)
        self.len = seq_len

        # Buffer args
        self.buffer: List[str] = []
        self.bos = bos_token
        self.eos = eos_token
        self.pad = pad_token
        self.pack_hard = pack_hard
        self.max_buffer_len = max_buffer_len
        self.buffer_len = 0
        if not pack_hard:
            assert (
                pad_token is not None
            ), "Error: if using pads, you must supply a pad_token"

        self.state_params = ["buffer", "buffer_len"]

    def _get_buffer(self, iterable, length, buffer):
        # Pull data until buffer is about to overrun, return exactly proper length
        new = []
        while len(buffer) + len(new) < length:
            buffer += new
            new = next(iterable)

        # Add bos if needed
        if self.bos is not None and (len(buffer) == 0 or buffer[0] != self.bos):
            buffer = [self.bos] + buffer

        # Handle buffer splitting
        if len(buffer) >= length:
            # If buffer is too long, force split
            out = buffer[:length]
            buffer = buffer[length:]
            if self.eos is not None and out[-1] != self.eos:
                buffer = [out[-1]] + buffer
                out[-1] = self.eos
            buffer = buffer + new
        else:
            if self.pack_hard:
                # Pack in as much of new sequence as will fit
                buffer = buffer + new
                out = buffer[:length]
                buffer = buffer[length:]
                if self.eos is not None and out[-1] != self.eos:
                    buffer = [out[-1]] + buffer
                    out[-1] = self.eos
            else:
                # Fill out with pads as needed
                if self.eos is not None and buffer[-1] != self.eos:
                    buffer.append(self.eos)
                if self.pad is not None:
                    out = buffer + [self.pad] * (length - len(buffer))
                else:
                    out = buffer
                buffer = new
        return out, buffer

    # Fill buffer line by line, delimiters and packing/splitting as appropriate
    def __iter__(self):
        dataset = iter(self.dataset)
        while True:
            out, buffer = self._get_buffer(dataset, self.len, self.buffer)
            self.buffer = buffer[: self.max_buffer_len]
            yield out

    def state_dict(self):
        # Pad out the buffer to constant size, log the real length
        # Constant size buffer is needed for distcp
        self.buffer_len = len(self.buffer)
        pad_val = 0 if len(self.buffer) == 0 else self.buffer[-1]
        self.buffer += [pad_val] * (self.max_buffer_len - self.buffer_len)
        out = super().state_dict()
        # Trim buffer back down to continue iterating
        self.buffer = self.buffer[: self.buffer_len]
        return out

    def load_state_dict(self, state_dict):
        # Unpad the buffer
        super().load_state_dict(state_dict)
        self.buffer = self.buffer[: self.buffer_len]


class SamplingDataset(_WrapperDataset):
    """
    A _WrapperDataset implementing percentage-based sampling: weights can be floats, and the
    number of tokens seen from each subdataset will match those weights as closely as possible.
    This is accomplished by maintaining a _StatefulDataset for each subdataset, and tracking
    the number of tokens emitted by each. Whichever loader is furthest from its target will be
    the next to pass a document.
    Relies on eos token to determine document boundaries, so must sit below BufferDataset.
    ...
    Args
    ----
    datapath : str
        Absolute path to the dataset directory. Expects directory to contain subfolders,
        which in turn contain shard files.
    dataset : _StatefulDataset
        Fully instantiated dataset. Cloned across desired subdatasets during setup.
    delimiter_token : Any
        Token used to indicate sequence/document breaks. Type should match data type.
    datasets : list[str] | None
        A list of subdatasets to draw from. If None, draws from all subfolders of datapath.
    weights : list(float) | None
        Weights describing what percent of emitted tokens should come from each subdataset.
        Need not sum to 1. If None, tokens are drawn evenly.
    verbose : bool
        Track setup progress?
    """

    def __init__(
        self,
        datapath: str,
        dataset: _StatefulDataset,
        delimiter_token: Any,
        datasets=None,
        weights=None,
        verbose=False,
    ):
        super().__init__(dataset)
        self.datapath = datapath
        self.delimiter = delimiter_token
        self.verbose = verbose
        self.datasets = (
            datasets
            if datasets is not None
            else [
                f
                for f in os.listdir(datapath)
                if not os.path.isfile(os.path.join(datapath, f)) and "meta" not in f
            ]
        )
        assert len(self.datasets) > 0, "You must specify at least one dataset"

        if weights is not None:
            assert len(weights) == len(
                self.datasets
            ), f"Number of oversample weights {len(weights)} must match number of datasets {len(self.datasets)}"
            for w in weights:
                assert w > 0, f"Sampling rate {w} must be positive"
        self.weights = [1] * len(self.datasets) if weights is None else weights
        self.weights = [w / sum(self.weights) for w in self.weights]

        self.tokens_seen = [0] * len(self.datasets)

        self.current_iterator = -1
        self.state_params = ["tokens_seen", "current_iterator"]

    def setup(self):
        if not self.is_setup:
            _StatefulDataset.setup(self)
            # Build subdataset iterators
            self.data = []
            for i, d in enumerate(self.datasets):
                self.data.append(deepcopy(self.dataset))
                self.data[-1].datapath = os.path.join(self.datapath, d)
                self.data[-1].rank = self.rank
                self.data[-1].worldsize = self.worldsize
                self.data[-1].local_worldsize = self.local_worldsize
                if self.verbose:
                    logging.info(
                        f"Worker {self.rank} assembled subdataset iterator for {d}, {i+1} of {len(self.datasets)}"
                    )
            [d.setup() for d in self.data]

    def __iter__(self):
        self.setup()
        # Grab one doc at a time in random order
        data = [iter(d) for d in self.data]
        while True:
            if self.current_iterator != -1:
                # Finish current document
                out = next(data[self.current_iterator])
                self.tokens_seen[self.current_iterator] += len(out)
                if out[-1] == self.delimiter:
                    self.current_iterator = -1
                yield out
            else:
                # Choose new subdataset to draw from
                # (whichever is currently most underrepresented compared to target rate)
                offset = [
                    self.weights[i]
                    - self.tokens_seen[i] / (sum(self.tokens_seen) + 1e-9)
                    for i in range(len(self.datasets))
                ]
                offset_argmax = max((diff, i) for i, diff in enumerate(offset))[1]
                self.current_iterator = offset_argmax

    def state_dict(self):
        self.setup()
        # Manually add state of all subloaders to self state
        iterator_states = [d.state_dict() for d in self.data]
        assert len(iterator_states) > 0, f"Worker {self.rank} owns no datasets"
        # Flip list[dict[any]] to dict[list[any]]
        prefix = self.statename("states.")
        out = {
            prefix + k: [d[k] for d in iterator_states]
            for k in iterator_states[0].keys()
        }
        out.update(_StatefulDataset.state_dict(self))
        return out

    def load_state_dict(self, state_dict):
        self.setup()
        # Load stats
        _StatefulDataset.load_state_dict(self, state_dict)
        # Load sub-iterator states
        prefix = self.statename("states.")
        # Flip dict[list[any]] to list[dict[any]]
        iterator_states = [
            {
                k[k.find(prefix) + len(prefix) :]: v[i]
                for k, v in state_dict.items()
                if prefix in k
            }
            for i in range(len(self.data))
        ]
        # Load individual state sub-dicts
        [
            self.data[i].load_state_dict(iterator_states[i])
            for i in range(len(self.data))
        ]


class CheckpointDataset(_WrapperDataset):
    """
    Wrapper for a _StatefulDataset that implements auto-checkpoint saving every n steps.
    Useful for setting n_workers > 0, so that workers do not rely on the master process
    for state saving (inter-process communication unsupported in PyTorch datasets).
    Once we have DCP support this can be removed.
    ...
    Args
    ----
    dataset : _StatefulDataset
        Fully instantiated dataset
    load_path : str
        Absolute path to checkpoint load directory. If a checkpoint exists, loads it.
    interval : int
        Saves a new checkpoint every interval training steps.
    steps_per_call : optional[int]
        The number of training steps required to call this loader once.
        Used in cases of multiple logical shards per worker to track intervals.
    save_path : optional[str]
        Absolute path to checkpoint save directory. Defaults to load_path.
    """

    def __init__(
        self,
        dataset: _StatefulDataset,
        load_path: str,
        interval: int,
        steps_per_call: float = 1.0,
        save_path: str = "",
    ):
        super().__init__(dataset)
        self.interval = interval
        self.spc = steps_per_call
        self.save_interval = -1
        load_path = os.path.join(load_path, "checkpoints")
        if len(save_path) == 0:
            save_path = load_path
        else:
            save_path = os.path.join(save_path, "checkpoints")
        self.load_path = load_path
        self.path = save_path
        self.step = 0

    def setup(self):
        if not self.is_setup:
            super().setup()
            # After possible world size adjustment, calculate save interval
            save_interval = self.interval / self.spc
            assert save_interval == int(
                save_interval
            ), f"Steps per call {self.spc} must divide save interval {self.interval} evenly"
            self.save_interval = int(save_interval)
            self.load_from_path(self.load_path)

    def __iter__(self):
        self.setup()
        dataset = iter(self.dataset)
        while True:
            yield next(dataset)
            self.step += 1
            if self.step % self.save_interval == 0:
                newpath = os.path.join(
                    self.path, "step_" + str(int(self.step * self.spc)) + "_ckp"
                )
                self.save_to_path(newpath)

    def report(self, msg):
        if self.rank == 0:
            print(msg)

    def _validate_ckp_path(self, path: str, verbose: bool = False):
        """
        Interpret path to appropriate checkpoint.
        If found, return modified path.
        If not found, return empty string.
        """
        # Does path exists, and if it exists, is it non-empty?
        if not os.path.exists(path) or len(os.listdir(path)) == 0:
            if verbose:
                self.report(
                    f"  Dataset: No valid checkpoint detected at {path}, dataset starting from scratch."
                )
            return ""
        # Check latest path, using ckp naming syntax
        latest = get_latest(path, key=lambda path: int(path.split("_")[-2]))
        if verbose:
            self.report(f"Checkpoint detected at {latest}")
        # If item is not a folder, exit early
        if os.path.isfile(latest):
            if verbose:
                self.report(
                    f"  Dataset: Detected checkpoint {latest} is a single file with no dataset info."
                    + " Dataset starting from scratch."
                )
            return ""
        # If item is a folder, check that it contains shard files
        if len([x for x in os.listdir(latest) if "loader" in x]) == 0:
            if verbose:
                self.report(
                    f"  Dataset: Detected checkpoint {latest} exists but contains no dataset checkpoints."
                    + " Dataset starting from scratch."
                )
            return ""
        # If item is a folder, get the step count
        self.step = int(int(latest.split("_")[-2]) / self.spc)
        return latest

    def save_to_path(self, path: str):
        """
        Grab recursive shard states and save all shard states to the specified checkpoint folder
        """
        self.report(f"Saving dataset to {path}")
        start = time.time()
        os.makedirs(path, exist_ok=True)
        torch.save(
            self.state_dict(), os.path.join(path, f"loader_state_{self.rank}.pth")
        )
        self.report(
            f"Dataset successfully saved to {path}! Save time: {time.time() - start}"
        )

    def load_from_path(self, path: str):
        save_path = self._validate_ckp_path(self.path, False)
        if len(save_path) > 0:
            self.report(
                f"  Dataset: Detected a checkpoint in the save directory {save_path}. Restoring from this checkpoint."
            )
            path = save_path
        else:
            load_path = self._validate_ckp_path(self.load_path, True)
            if len(load_path) == 0:
                return
            else:
                path = load_path
                # When loading from external ckp, always reset step count
                self.step = 0
        # Proceed
        start = time.time()
        fileshards = [x for x in os.listdir(path) if "loader" in x]
        fileshards = sorted(fileshards, key=lambda x: int(x.split("_")[2][:-4]))
        assert (
            len(fileshards) == self.worldsize
        ), f"Number of checkpoint files {len(fileshards)} does not match number of workers {self.worldsize}"
        state = torch.load(os.path.join(path, fileshards[self.rank]))
        self.dataset.load_state_dict(state)
        self.report(f"Dataset checkpoint loaded! Load time: {time.time() - start}")


class StreamingDocDataset(_StatefulDataset):
    """
    The base distributed dataset for loading sequences/documents from file shards.

    For a single dataset directory, detects all shard files, aggregates their file sizes, and partitions
    (fractionally) by file size over workers (as a proxy for token count).
    Logs the number of documents owned from each shardfile, and relies on LCG random bijection to
    map contiguous range of indices to shuffled, noncontiguous set of documents from each shard file.
    Shuffles the file list deterministically to hop from file to file.

    At runtime, iterates through documents in each shuffled shard file, pulling each shard on demand.
    Shards are thus pulled no more than once per epoch.
    Returns documents in chunks up to size max_chunksize, and handles delimiter token placement between documents.

    StreamingDocDataset grabs files from a directory representing a single dataset.
    This directory need not be flat.
    For percentage-based sampling over multiple such subdatasets, see SamplingDataset.

    When available in the parent directory, relies on a compiled metadata file to fetch document count per shardfile.
    Expects csv file (first row "dataset/filename,documents,tokens", subsequent rows these values) under a 'meta' directory.
    This can be removed in the future.
    ...
    Args
    ----
    datapath : str
        Absolute path to the dataset directory. Expects directory containing shardfiles.
        Directory need not be flat.
    rank : int
        Current worker index
    worldsize : int
        Total number of workers
    filereader : _ShardFileReader
        A file reader handling specific data shard file formats
    delimiter_token : Any
        Token used to indicate sequence/document breaks. Type should match data type. Required for downstream
        sampling logic (can be removed later via PreProcessDataset if needed).
    bos_token : Any | None
        Optional token used to indicate sequence/document start. Type should match data type.
    strip_tokens : set[Any]
        Token values that should be removed if detected at beginning or end of document
        (i.e. any eos/bos tokens already present in the data). Type should match data type.
    seed : int
        The random seed for deterministic shuffling/sharding
    min_length : int
        Documents below this length are skipped
    max_chunksize : int
        Maximum sequence length to return. Break long docs into chunks of this size or shorter.
    verbose : bool
        Track setup progress?
    shuffle : bool
        Shuffle shard file and document orders? (Disable for simple testing)
    """

    def __init__(
        self,
        datapath: str,
        rank: int,
        worldsize: int,
        filehandler: _ShardFileHandler,
        delimiter_token: Any,
        bos_token: Optional[Any] = None,
        strip_tokens: Optional[Set[Any]] = set(),
        seed: int = 42,
        min_length: int = 1,
        max_chunksize: int = 1024,
        verbose: bool = False,
    ):
        super().__init__(datapath, rank, worldsize)
        self.seed = seed
        self.datapath = datapath
        self.filehandler = filehandler
        self.min_length = min_length
        assert max_chunksize > 0, f"Max chunksize must be a nonzero positive integer"
        self.chunksize = max_chunksize
        self.eos = delimiter_token
        self.bos = bos_token
        self.drop = strip_tokens
        self.verbose = verbose
        self.docset: List[
            Any
        ] = []  # map of shard indices to (shardid, min docid, max docid)

        # Position
        self.docset_index = 0
        self.chunk_index = -1

        # Stats
        self.epochs_seen = -1
        self.tokens_seen = 0
        self.docs_seen = 0
        self.percent_seen = 0

        self.state_params = [
            # "dataset", # can't put strings into tensor
            "docset_index",
            "chunk_index",
            "epochs_seen",
            "tokens_seen",
            "docs_seen",
            "percent_seen",
            "lcg_state",
        ]

        # Setup flags
        self.is_setup = False
        self._len = 0
        self.dataset = ""
        self.lcg_state = 0

    def setup(self):
        """
        All rank-dependent setup, which must occur after init
        (rank assignment, data partitioning, shuffling)
        """
        if not self.is_setup:
            super().setup()
            datapath = self.datapath
            pathsplit = (datapath, "")
            # May take an extra round to account for any trailing slashes
            while len(pathsplit[1]) == 0:
                pathsplit = os.path.split(pathsplit[0])
            pardir, dataset = pathsplit
            self.dataset = dataset

            # Assemble set of available shard files
            shards = [
                os.path.join(root, name)[len(datapath) + 1 :]
                for root, dirs, files in os.walk(datapath, topdown=False)
                for name in files
                if self.filehandler.is_legal(os.path.join(root, name))
            ]
            shards.sort()  # Ensure consistent sharding across machines

            # Use shard file sizes to perform partitioning
            # Create shardlist of form filename -> [start%, end%]
            shard_sizes = [
                os.path.getsize(os.path.join(datapath, shard)) for shard in shards
            ]
            shard_sizes = [s / sum(shard_sizes) for s in shard_sizes]
            start = self.rank / self.worldsize
            end = (self.rank + 1) / self.worldsize
            shardset = {}
            tally = 0
            for i in range(len(shards)):
                if tally <= end and tally + shard_sizes[i] >= start:
                    shardset[shards[i]] = [
                        min(max((start - tally) / shard_sizes[i], 0), 1),
                        min(max((end - tally) / shard_sizes[i], 0), 1),
                    ]
                tally += shard_sizes[i]

            # Assemble length in documents of each owned shard file
            countfiles = []
            if os.path.exists(os.path.join(pardir, "meta")):
                countfiles = [
                    x
                    for x in os.listdir(os.path.join(pardir, "meta"))
                    if "counts" in x and "csv" in x
                ]
            doc_counts = {}
            if len(countfiles) > 0:
                # Count file exists, use it
                countpath = os.path.join(pardir, "meta", countfiles[0])
                with open(countpath, "r") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        fullpath = row["dataset/filename"]
                        prefix = fullpath.find("/" + dataset) + 1
                        if prefix > 0:
                            key = fullpath[prefix + len(dataset) + 1 :]
                            doc_counts[key] = int(row["documents"])
            else:
                # Count file does not exist, touch every owned file for length
                doc_counts = {
                    shard: self.filehandler.length(os.path.join(datapath, shard))
                    for shard in shardset
                }

            # Assemble doc list for each file shard
            # Create docset of form [filename, min docid, max docid]
            doccount = 0
            for shard in shardset:
                ndocs = doc_counts[shard]
                doc_start = round(ndocs * shardset[shard][0])
                doc_end = round(ndocs * shardset[shard][1])
                if doc_end >= doc_start:
                    self.docset.append([shard, doc_start, doc_end])
                    doccount += doc_end - doc_start
            self._len = doccount

            if self.verbose:
                logging.info(
                    f"    Worker {self.rank} ingested {len(self.docset)} shards from {dataset}"
                )

            # Shuffle shard files - guaranteed inconsistent across workers
            seed = self.seed + self.rank
            random.seed(seed)
            random.shuffle(self.docset)
            # Setup doc shuffle - same guarantee
            self.lcg_state = seed

    def _get_docid(self, i):
        """
        Given a global doc index over the set of docs owned by this worker,
        return the corresponding data/shard/local index
        """
        cur = 0
        assert (
            i <= self._len
        ), f"You have requested an illegal doc index {i}, docset length is {self._len}"
        for shardid, min_d, max_d in self.docset:
            docrange = max_d - min_d
            cur += docrange
            if cur > i:
                return shardid, docrange, min_d

    def _get_reader(self, path, newpath, reader):
        """
        If new filepath does not match the current one,
        open a new reader on that filepath (pull file on demand)
        """
        if newpath != path:
            del reader
            if self.verbose:
                logging.info(f"Worker {self.rank} opening new file {newpath}")
            reader = self.filehandler.open(newpath)
            path = newpath
        return path, reader

    def _construct_chunk(self, j, doc, n_chunks):
        """
        Grab a chunk of the desired size from the document, with eos/bos handling
        """
        start_index = j * self.chunksize
        n_pull = self.chunksize
        if self.bos is not None:
            if j == 0:
                n_pull -= 1
            else:
                start_index -= 1
        chunk = self.filehandler.slice(doc, start_index, n_pull)
        self.tokens_seen += len(chunk)
        # Add bos/eos tokens if needed
        if self.bos is not None and j == 0:
            chunk = [self.bos] + chunk
        if j == n_chunks - 1:
            chunk = chunk + [self.eos]
        return chunk

    def _random_map_docid(self, size):
        """
        Given size of document pool, use saved state (prior index) to generate the next index via LCG.
        Implements within-shard document shuffling without materializing any large doc lists.
        """
        m = 2 ** math.ceil(math.log2(size))  # Round up to nearest power of 2
        a = 5  # A,C values known to work well with powers of 2 (Knuth, 1997, 3.2.1.3)
        c = (self.rank + self.seed) * 2 + 1
        state = self.lcg_state
        while True:
            state = (a * state + c) % m
            if state < size:
                return state

    def __iter__(self):
        if not self.is_setup:
            self.setup()
        docset_offset = self.docset_index
        lcg_offset = self.lcg_state
        residual_chunks = self.chunk_index + 1  # pick up AFTER where the ckp left off
        ndocs = self._len
        path = ""
        reader = None
        while True:
            # Iterate through docs, starting at desired offset
            for i in range(ndocs):
                doc_index = (docset_offset + i) % ndocs

                # Update stats
                if doc_index == 0:
                    self.epochs_seen += 1
                self.docset_index = doc_index
                # Map doc id to shard, id in file
                shardid, docrange, mindoc = self._get_docid(doc_index)

                # Read doc
                newpath = os.path.join(self.datapath, shardid)
                path, reader = self._get_reader(path, newpath, reader)
                # Map id in range of owned docs to new (consistently) shuffled id
                doclcg = self._random_map_docid(docrange)
                docid = doclcg + mindoc
                doc = self.filehandler.get(reader, docid, self.drop)
                if len(doc) == 0:
                    continue
                doclen = len(doc) + 1 if self.bos is None else len(doc) + 2
                if doclen >= self.min_length:
                    n_chunks = math.ceil(doclen / self.chunksize)
                    for j in range(n_chunks):
                        if i == 0 and j < residual_chunks:
                            pass
                        else:
                            self.chunk_index = j
                            # Document complete, update stats
                            if j == n_chunks - 1:
                                self.docs_seen += 1
                                self.percent_seen = (
                                    self.docs_seen * 100 / (self._len + 1e-9)
                                )
                            yield self._construct_chunk(j, doc, n_chunks)

                # Advance RNG state
                self.lcg_state = doclcg

            # Load any chunks initially skipped in first doc
            self.docset_index = docset_offset
            self.lcg_state = lcg_offset
            shardid, docrange, mindoc = self._get_docid(docset_offset)
            docid = self._random_map_docid(docrange) + mindoc
            newpath = os.path.join(self.datapath, shardid)
            path, reader = self._get_reader(path, newpath, reader)
            doc = self.filehandler.get(reader, docid, self.drop)
            if len(doc) == 0:
                continue
            doclen = len(doc) + 1 if self.bos is None else len(doc) + 2
            if doclen >= self.min_length:
                n_chunks = math.ceil(doclen / self.chunksize)
                for j in range(residual_chunks):
                    self.chunk_index = j
                    yield self._construct_chunk(j, doc, n_chunks)


class ScalableShardDataset(_WrapperDataset):
    """
    A _WrapperDataset implementing rescalability: loading from checkpoint into a different
    number of gpus will nonetheless keep avoiding all data previously seen in the current epoch.
    This is accomplished by maintaining a large number of smaller StatefulDatasets, cloned from the
    original dataset arg with adjusted ranks, which track state individually and reshard over n_gpus.
    Rescaling only works when this layer wraps all other layers that contribute to state_dict.
    ...
    Args
    ----
    dataset : _StatefulDataset
        Fully instantiated dataset. Cloned into logical workers during setup fn.
    n_logical_shards : int
        Total number of logical shards. Must be a multiple of world size.
    verbose : bool
        Track setup progress?
    """

    def __init__(
        self,
        dataset: _StatefulDataset,
        n_logical_shards: int = 2048,
        verbose=False,
    ):
        super().__init__(dataset)
        assert (
            n_logical_shards % self.worldsize == 0
        ), f"World size {self.worldsize} must divide n_logical_shards {n_logical_shards} evenly"
        assert (
            n_logical_shards > 0
        ), f"n_logical_shards {n_logical_shards} must be a positive integer"

        self.total_shards = n_logical_shards
        self.verbose = verbose

        # Fields to be populated during setup / subdataset setup
        self.data: List[_StatefulDataset] = []
        self.logicals_owned: List[int] = []
        self.n_logicals = 0
        self.generator = None

        # Position "state", used only for maintaining order when n_workers is unchanged
        # For scaling up or down, logical position is meaningless, and reset
        self.current_reader = 0
        self.load_worldsize = self.worldsize

        self.state_params = ["current_reader"]  # self.data states are handled manually

    def setup(self):
        if not self.is_setup:
            _StatefulDataset.setup(self)
            n_logical_shards = self.total_shards
            logicals = list(range(n_logical_shards))
            self.logicals_owned = _shard_partition(logicals, self.rank, self.worldsize)
            self.n_logicals = n_logical_shards // self.worldsize
            assert (
                len(self.logicals_owned) == self.n_logicals
            ), "(world size * num workers) does not divide logical shards evenly"

            # Build logical shards
            for i in range(self.n_logicals):
                self.data.append(deepcopy(self.dataset))
                self.data[-1].worldsize = n_logical_shards
                self.data[-1].rank = self.logicals_owned[i]
                self.data[-1].local_worldsize = 1
                self.data[-1].datapath = self.datapath
                self.data[-1].verbose = self.rank == 0
                if self.verbose:
                    logging.info(
                        f"Worker {self.rank} assembled logical shard {self.logicals_owned[i]}, {i+1} of {self.n_logicals}"
                    )
            [d.setup() for d in self.data]

    def _reshard(self, sharded_list):
        """
        Sharded_list is a list of lists, where each "shard" sublist must have the same length.
        These shards should tightly span only the partition of data owned by this worker.
        (i.e. if global_list is the list of all entries, sharded_list = _shard_inclusive(global_list) ).
        Determine fractional ownership of shards, and get the flattened partition owned by this worker.
        Once we have DCP support, this can be removed.
        """
        # How many shards did _shard_inclusive() drop to the left of sharded_list?
        shard_offset = math.floor(self.load_worldsize * self.rank / self.worldsize)
        # How long are the list shards?
        shard_len = len(sharded_list[0])
        for i, shard in enumerate(sharded_list):
            assert (
                len(shard) == shard_len
            ), f"Shard {i} with length {len(shard)} does not match expected {shard_len}"
        # How many list items did _shard_inclusive() drop to the left of the flattened sharded_list?
        item_offset = shard_len * shard_offset
        # How many list items are there in total?
        n_items = self.load_worldsize * shard_len
        # The indices of the flattened sharded_list that this worker owns
        my_items = range(
            int(n_items * self.rank / self.worldsize) - item_offset,
            int(n_items * (self.rank + 1) / self.worldsize) - item_offset,
        )
        # Pull out owned items
        return [sharded_list[i // shard_len][i % shard_len] for i in my_items]

    def __iter__(self):
        self.setup()
        # Grab one item at a time, iterating over owned logical shards
        data = [iter(d) for d in self.data]
        while True:
            ind = self.current_reader
            # Read doc
            out = next(data[ind])
            # Update state
            self.current_reader = (self.current_reader + 1) % self.n_logicals
            yield out

    def state_dict(self):
        self.setup()
        # Recursive fetch
        logical_shard_states = [d.state_dict() for d in self.data]
        assert len(logical_shard_states) > 0, f"Worker {self.rank} owns no shards???"
        # Flip list[dict[Any]] to dict[list[Any]]
        state_dict = {
            k: [d[k] for d in logical_shard_states]
            for k in logical_shard_states[0].keys()
        }
        state_dict.update(_StatefulDataset.state_dict(self))

        # Convert to tensor form
        out = {}
        for k, v in state_dict.items():
            if self.rank == 0:
                print(k, v)
            v = torch.tensor(v)
            if len(v.shape) == 0:
                k = k + ".scalar"
                v = v.unsqueeze(0)
            out[k] = v

        return out

    def load_state_dict(self, state_dicts):
        """
        If state_dicts is a single state dict, proceeds without rescaling
        If state_dicts is a list, expects the full set of state dicts
        Once we have DCP support, the list case will be eliminated
        """
        self.setup()
        # Build checkpoint shard list
        single_load = isinstance(state_dicts, dict)
        if single_load:
            state_dicts = [state_dicts]
        else:
            self.load_worldsize = len(state_dicts)
            state_dicts = _shard_inclusive(state_dicts, self.rank, self.worldsize)

        # Convert back to lists and scalars
        def detorchify(k, v):
            v = v.tolist()
            if ".scalar" in k:
                k = k[:-7]
                v = v[0]
            return k, v

        plain_dicts = []
        for d in state_dicts:
            p = {}
            for k, v in d.items():
                k, v = detorchify(k, v)
                p[k] = v
            plain_dicts.append(p)
        state_dicts = plain_dicts

        # Assemble logical shard states
        if single_load:
            state_dicts = state_dicts[0]
            _StatefulDataset.load_state_dict(self, state_dicts)
            # Remove all non-resharding state
            [state_dicts.pop(self.statename(n)) for n in self.state_params]
            # Flip dict[list[any]] to list[dict[any]]
            logical_shard_states = [
                {k: v[i] for k, v in state_dicts.items()}
                for i in range(self.n_logicals)
            ]
        else:
            # Remove all non-resharding state
            for d in state_dicts:
                [d.pop(self.statename(n)) for n in self.state_params]
            # Calculate old n_logicals: len of first entry of first dict in state_dicts
            old_n_logicals = len(state_dicts[0][list(state_dicts[0].keys())[0]])
            # Flip list[dict[list[any]]] to list[list[dict[any]]]
            state_dicts = [
                [{k: v[i] for k, v in d.items()} for i in range(old_n_logicals)]
                for d in state_dicts
            ]
            # Perform resharding
            logical_shard_states = self._reshard(state_dicts)

        # Load values
        for i in range(self.n_logicals):
            self.data[i].load_state_dict(logical_shard_states[i])


class StateDeltaDataset(_WrapperDataset):
    """
    Appends rank and state dict deltas to outputs so that StatefulDataloader can update
    its own mirrored state dict. 
    Output is [data, rank, {key : [shape, indices, updated_values]}]
    """
    def __init__(self, dataset: _StatefulDataset):
        super().__init__(dataset)
        self.state = None

    def _compute_delta(self, new, old):
        # For each key: [size, indices, new vals]
        keys = new.keys()
        out = {}
        for k in keys:
            newv = new[k]
            oldv = old[k]
            if not torch.equal(newv, oldv):
                newvs = newv.size()
                oldvs = oldv.size()
                assert len(newvs)==len(oldvs), f"State dict field dims cannot change over time ({k} size {newvs} was {oldvs})"
                assert newvs[1:]==oldvs[1:], f"State dict field sizes must agree after dim 0 ({k} size {newvs} was {oldvs})"
                # In case of size mismatch, adjust old to match new
                if newvs[0] < oldvs[0]:
                    oldvs = oldvs[:newvs[0]]
                elif newvs[0] > oldvs[0]:
                    oldvs = torch.nn.functional.pad(oldvs, [0,0]*(len(newvs)-1) + [0,newvs[0]-oldvs[0]])
            # Fetch delta indices
            delta_indices = newv.sub(oldv).nonzero()
            if len(delta_indices) > 0:
                # Fetch delta values
                delta_vals = torch.stack(
                    [newv[delta_indices[i].split(1)] for i in range(delta_indices.size(0))],
                    0,
                )
                out[k] = [newv.size(), delta_indices, delta_vals]
        return out

    def __iter__(self):
        self.setup()
        data = iter(self.dataset)
        while True:
            out = next(data)
            new_state = self.dataset.state_dict()
            if self.state is None:
                delta = new_state
            else:
                delta = self._compute_delta(new_state, self.state)
            self.state = new_state
            yield [out, self.rank, delta]


class LoaderMonitor():
    def __init__(self):
        self.state = {}
        self.n_updates = {}

    def apply_delta(self, delta, state):
        for k in delta.keys():
            # delta: k -> [size, inds, vals]
            deltas = delta[k][0]
            vs = state[k].size()
            assert len(deltas)==len(vs), f"State dict field dims cannot change over time ({k} size {deltas} was {vs})"
            assert deltas[1:]==vs[1:], f"State dict field sizes must agree after dim 0 ({k} size {deltas} was {vs})"
            # in case of size mismatch, adjust state to match delta
            if deltas[0] < vs[0]:
                state[k] = state[k][:deltas[0]]
            elif deltas[0] > vs[0]:
                state[k] = torch.nn.functional.pad(state[k], [0,0]*(len(deltas)-1) + [0,deltas[0]-vs[0]])
            # Apply deltas
            for i,tup in enumerate(delta[k][1]):
                state[k][tup.split(1)] = delta[k][2][i]
        return state

    def collate(self, inp):
        # inp: [[out tensor, rank, {key: [size, ind tensor, val tensor]}]]
        
        # Get eventual output
        if isinstance(inp[0][0], torch.Tensor):
            out = torch.stack([x[0] for x in inp], dim=0)
        else:
            out = [torch.stack([x[0][i] for x in inp], dim=0) for i in range(len(inp[0][0]))]
        
        # Update state
        for row in inp:
            rank = row[1]
            if rank not in self.state:
                self.state[rank] = row[2]
                self.n_updates[rank] = 1
            else:
                self.state[rank] = self.apply_delta(row[2], self.state[rank])
                self.n_updates[rank] += 1
        
        return out
    
    def state_dict(self):
        rs = list(self.state.keys())
        minr = min(rs)
        maxr = max(rs)
        return [self.state[i] for i in range(minr, maxr+1, 1)]
    
    def save_state_dict(self, path:str, device_mesh=None, placements=None):
        state = self.state_dict()
        # Flip list[dict[tensor]] to dict[list[tensor]], and concat
        state = {k:torch.cat([d[k] for d in state], 0) for k in state[0]}
        # Construct dtensors from tensors
        dstate = {
            k: torch.distributed.tensor.DTensor.from_local(
                v,
                device_mesh,
                placements,
            )
            for k, v in state.items()
        }
        # Write state dict
        writer = torch.distributed.checkpoint.FileSystemWriter(path)
        torch.distributed.checkpoint.save(
            dstate,
            writer,
        )

    def load_state_dict(self, path:str, device_mesh=None, placements=None):
        state = self.state_dict()
        # Flip list[dict[tensor]] to dict[list[tensor]], and concat
        state = {k:torch.cat([d[k] for d in state], 0) for k in state[0]}
        # Construct dtensors from tensors
        dstate = {
            k: torch.distributed.tensor.DTensor.from_local(
                v,
                device_mesh,
                placements,
            )
            for k, v in state.items()
        }
        # Read state dict
        reader = torch.distributed.checkpoint.FileSystemReader(path)
        torch.distributed.checkpoint.load_state_dict(
            dstate,
            reader,
        )
        # Get local tensors from dtensors
        state = {k:v.to_local() for k,v in dstate.items()}

        return state
        
        # TODO: split into list, send list into worker processes. Need actual dataloader interfacing for this!



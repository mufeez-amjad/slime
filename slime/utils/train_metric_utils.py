import logging
import json
from pathlib import Path
from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy

from slime.utils import tracking_utils
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.timer import Timer

logger = logging.getLogger(__name__)


def log_perf_data_raw(
    rollout_id: int, args: Namespace, is_primary_rank: bool, compute_total_fwd_flops: Callable
) -> None:
    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    timer_instance.reset()

    # Peak memory stats (max across ranks), computed before early-return so all
    # ranks participate in collectives.
    max_allocated_bytes: int | None = None
    max_reserved_bytes: int | None = None
    try:
        import torch
        import torch.distributed as dist

        if torch.cuda.is_available():
            alloc = int(torch.cuda.max_memory_allocated())
            reserved = int(torch.cuda.max_memory_reserved())
            if dist.is_available() and dist.is_initialized():
                t = torch.tensor([alloc, reserved], dtype=torch.int64, device=torch.cuda.current_device())
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                max_allocated_bytes = int(t[0].item())
                max_reserved_bytes = int(t[1].item())
            else:
                max_allocated_bytes = alloc
                max_reserved_bytes = reserved
    except Exception:
        # Best-effort only: never fail training due to perf logging.
        pass

    if not is_primary_rank:
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}

    # Token throughput is useful even without a FLOPs estimator.
    if "perf/actor_train_time" in log_dict and log_dict["perf/actor_train_time"] > 0:
        seq_lens = getattr(timer_instance, "seq_lens", None)
        if isinstance(seq_lens, list) and len(seq_lens) > 0:
            log_dict["perf/actor_train_tok_per_s"] = sum(seq_lens) / log_dict["perf/actor_train_time"]

    if ("perf/actor_train_time" in log_dict) and (compute_total_fwd_flops is not None):
        total_fwd_flops = compute_total_fwd_flops(seq_lens=timer_instance.seq_lens)

        if "perf/log_probs_time" in log_dict:
            log_dict["perf/log_probs_tflops"] = total_fwd_flops / log_dict["perf/log_probs_time"]

        if "perf/ref_log_probs_time" in log_dict:
            log_dict["perf/ref_log_probs_tflops"] = total_fwd_flops / log_dict["perf/ref_log_probs_time"]

        if log_dict["perf/actor_train_time"] > 0:
            log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["perf/actor_train_time"]
            log_dict["perf/actor_train_tok_per_s"] = sum(timer_instance.seq_lens) / log_dict["perf/actor_train_time"]

    if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
        total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
        if total_time > 0:
            log_dict["perf/step_time"] = total_time
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time

    if max_allocated_bytes is not None:
        log_dict["perf/max_cuda_memory_allocated_bytes"] = max_allocated_bytes
    if max_reserved_bytes is not None:
        log_dict["perf/max_cuda_memory_reserved_bytes"] = max_reserved_bytes

    logger.info(f"perf {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking_utils.log(args, log_dict, step_key="rollout/step")

    # Optional: persist perf data for post-hoc benchmarks.
    # Prefer --perf-jsonl-path to avoid enabling other debug dumps.
    perf_jsonl_path = getattr(args, "perf_jsonl_path", None)
    dump_dir = getattr(args, "dump_details", None)
    path: Path | None = None
    if perf_jsonl_path:
        path = Path(perf_jsonl_path)
    elif dump_dir:
        path = Path(dump_dir) / "perf.jsonl"

    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {"rollout_id": rollout_id, **log_dict}
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write perf jsonl to {path}: {e}")

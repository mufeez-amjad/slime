import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"


FEW_GPU = U.get_bool_env_var("SLIME_TEST_FEW_GPU", "1")
CP_BENCH = U.get_bool_env_var("SLIME_TEST_CP_BENCH", "0")


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    if not CP_BENCH:
        U.hf_download_dataset("zhuzilin/gsm8k")


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _write_synthetic_debug_rollouts(
    *,
    model_path: str,
    out_dir: str,
    num_rollout: int,
    rollout_batch_size: int,
    n_samples_per_prompt: int,
    prompt_len: int,
    response_len: int,
    seed: int,
) -> str:
    """Create debug rollout dumps compatible with --load-debug-rollout-data."""
    from pathlib import Path
    import random

    import torch
    from transformers import AutoConfig

    from slime.utils.types import Sample

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab_size = int(getattr(cfg, "vocab_size"))
    assert vocab_size > 10, f"Unexpected {vocab_size=}"

    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    num_samples = int(rollout_batch_size) * int(n_samples_per_prompt)
    for rollout_id in range(num_rollout):
        samples: list[Sample] = []
        for i in range(num_samples):
            group_index = i // n_samples_per_prompt
            prompt_tokens = [rng.randrange(1, vocab_size) for _ in range(prompt_len)]
            response_tokens = [rng.randrange(1, vocab_size) for _ in range(response_len)]
            samples.append(
                Sample(
                    group_index=group_index,
                    index=i,
                    prompt="synthetic",
                    tokens=prompt_tokens + response_tokens,
                    response="synthetic",
                    response_length=response_len,
                    reward=1.0,
                    loss_mask=[1] * response_len,
                    status=Sample.Status.COMPLETED,
                )
            )

        path = out / f"rollout_{rollout_id}.pt"
        torch.save({"rollout_id": rollout_id, "samples": [s.to_dict() for s in samples]}, path)

    return str(out / "rollout_{rollout_id}.pt")


def _summarize_perf_jsonl(path: str, warmup: int = 1) -> dict[str, float]:
    import json
    from pathlib import Path

    p = Path(path)
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if isinstance(r, dict)]
    rows = rows[warmup:] if warmup and len(rows) > warmup else rows

    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
        return sum(vals) / max(1, len(vals))

    def _max(key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
        return max(vals) if vals else 0.0

    return {
        "n": float(len(rows)),
        "step_time_s_mean": _mean("perf/step_time"),
        "actor_train_time_s_mean": _mean("perf/actor_train_time"),
        "log_probs_time_s_mean": _mean("perf/log_probs_time"),
        "actor_train_tok_per_s_mean": _mean("perf/actor_train_tok_per_s"),
        "max_cuda_mem_alloc_gb": _max("perf/max_cuda_memory_allocated_bytes") / (1024**3),
        "max_cuda_mem_reserved_gb": _max("perf/max_cuda_memory_reserved_bytes") / (1024**3),
    }


def execute_cp_benchmark():
    model_path = f"/root/models/{MODEL_NAME}"
    bench_root = os.environ.get("SLIME_TEST_CP_BENCH_DIR", "/tmp/slime_cp_bench_qwen3_0.6B")
    num_gpus = _env_int("SLIME_TEST_CP_BENCH_NUM_GPUS", 4)
    rollout_batch_size = _env_int("SLIME_TEST_CP_BENCH_ROLLOUT_BATCH_SIZE", 4)
    n_samples_per_prompt = _env_int("SLIME_TEST_CP_BENCH_N_SAMPLES_PER_PROMPT", 2)
    num_rollout = _env_int("SLIME_TEST_CP_BENCH_NUM_ROLLOUT", 6)
    warmup = _env_int("SLIME_TEST_CP_BENCH_WARMUP", 1)
    prompt_len = _env_int("SLIME_TEST_CP_BENCH_PROMPT_LEN", 256)
    response_len = _env_int("SLIME_TEST_CP_BENCH_RESPONSE_LEN", 1024)
    seed = _env_int("SLIME_TEST_CP_BENCH_SEED", 123)

    rollout_template = _write_synthetic_debug_rollouts(
        model_path=model_path,
        out_dir=f"{bench_root}/rollout_data",
        num_rollout=num_rollout,
        rollout_batch_size=rollout_batch_size,
        n_samples_per_prompt=n_samples_per_prompt,
        prompt_len=prompt_len,
        response_len=response_len,
        seed=seed,
    )

    total_samples = rollout_batch_size * n_samples_per_prompt
    base_args = (
        f"--hf-checkpoint {model_path} "
        f"--load-debug-rollout-data {rollout_template} "
        "--disable-rollout-global-dataset "
        f"--num-rollout {num_rollout} "
        f"--rollout-batch-size {rollout_batch_size} "
        f"--n-samples-per-prompt {n_samples_per_prompt} "
        f"--global-batch-size {total_samples} "
        f"--rollout-max-response-len {response_len} "
        "--micro-batch-size 1 "
        "--advantage-estimator grpo "
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {num_gpus} "
        "--rollout-num-gpus 0 "
        "--train-backend fsdp "
        "--sglang-router-ip 127.0.0.1 "
        "--sglang-router-port 30000 "
    )

    results = {}
    cp_sizes = [1, 2]
    for cp_size in cp_sizes:
        assert num_gpus % cp_size == 0, f"num_gpus ({num_gpus}) must be divisible by cp_size ({cp_size})"
        perf_path = f"{bench_root}/perf_cp{cp_size}.jsonl"
        train_args = base_args + f"--context-parallel-size {cp_size} " f"--perf-jsonl-path {perf_path} "
        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=num_gpus,
            megatron_model_type=None,
            train_script="train_async.py",
        )
        results[cp_size] = _summarize_perf_jsonl(perf_path, warmup=warmup)

    print("CP benchmark summary (lower time is better; lower VRAM is better):")
    for cp_size in cp_sizes:
        r = results[cp_size]
        print(
            f"  cp={cp_size}: "
            f"n={int(r['n'])} "
            f"step_time={r['step_time_s_mean']:.4f}s "
            f"actor_train_time={r['actor_train_time_s_mean']:.4f}s "
            f"log_probs_time={r['log_probs_time_s_mean']:.4f}s "
            f"tok/s={r['actor_train_tok_per_s_mean']:.1f} "
            f"max_alloc={r['max_cuda_mem_alloc_gb']:.2f}GB "
            f"max_reserved={r['max_cuda_mem_reserved_gb']:.2f}GB"
        )

    r1, r2 = results[1], results[2]
    if r1["step_time_s_mean"] > 0 and r2["step_time_s_mean"] > 0:
        print(f"  cp=2 vs cp=1 speedup (step_time): {r1['step_time_s_mean'] / r2['step_time_s_mean']:.3f}x")
    if r1["log_probs_time_s_mean"] > 0 and r2["log_probs_time_s_mean"] > 0:
        print(
            f"  cp=2 vs cp=1 speedup (log_probs_time): {r1['log_probs_time_s_mean'] / r2['log_probs_time_s_mean']:.3f}x"
        )
    if r1["max_cuda_mem_alloc_gb"] > 0 and r2["max_cuda_mem_alloc_gb"] > 0:
        print(
            f"  cp=2 vs cp=1 peak alloc reduction: "
            f"{(1.0 - (r2['max_cuda_mem_alloc_gb'] / r1['max_cuda_mem_alloc_gb'])) * 100.0:.1f}%"
        )


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        # NOTE cannot be exactly multiple of eval-interval, since async causes some offsets
        f"--num-rollout {3000 if U.get_env_enable_infinite_run() else 65} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--over-sampling-batch-size 64 "
        "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 256 "
    )

    eval_args = (
        "--eval-interval 20 "
        "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        # "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-enable-metrics "

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {1 if FEW_GPU else 2} "
        f"--rollout-num-gpus {1 if FEW_GPU else 2} "
        "--train-backend fsdp "
    )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--ci-metric-checker-key eval/gsm8k "
        "--ci-metric-checker-threshold 0.71 "  # loose threshold at 60 step
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=2 if FEW_GPU else 4,
        megatron_model_type=None,
        train_script="train_async.py",
    )


if __name__ == "__main__":
    prepare()
    os.environ.pop("http_proxy")
    os.environ.pop("https_proxy")
    os.environ.pop("HTTP_PROXY")
    os.environ.pop("HTTPS_PROXY")
    execute_cp_benchmark() if CP_BENCH else execute()

import csv
import statistics
import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from llm_from_scratch.engine import InferenceEngine, InferenceEngineConfig
from llm_from_scratch.generation import SamplingParams
from llm_from_scratch.utils import fetch_device

# With this script you can benchmark the inference engine and see whether our changes have any improvements so far.

# Benchmark params
MODEL_PATH = "model/long-context/best_model.pt"
DEVICE = fetch_device()
PROMPT = ""
MAX_SEQ_LEN = 4096
END = 4086
STEP = 256
NUM_RUNS = 3
OUTPUT_PATH = Path("data/benchmarks/inference_benchmark_kv/")
RESULTS_PATH = OUTPUT_PATH / "results.csv"
PLOT_PATH = OUTPUT_PATH / "plot.png"


def synchronize() -> None:
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()


def benchmark(
    use_kv_cache: bool, token_counts: list[int]
) -> tuple[list[int], list[float], list[list[int]]]:
    engine = InferenceEngine(
        InferenceEngineConfig(
            model_path=MODEL_PATH,
            device=DEVICE,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=1,
            use_kv_cache=use_kv_cache,
        )
    )

    # Warm up request to not have cold starts
    engine.generate(PROMPT, SamplingParams(max_new_tokens=2, top_k=1))

    # Synchronize makes sure all async gpu work is done before we start proper benchmark
    synchronize()

    generated_counts = []
    times = []
    outputs = []

    for token_count in token_counts:
        durations = []
        run_outputs = []
        for _ in range(NUM_RUNS):
            synchronize()
            start = time.perf_counter()
            output = engine.generate(
                PROMPT,
                SamplingParams(max_new_tokens=token_count, top_k=1),
            )
            synchronize()
            durations.append(time.perf_counter() - start)
            run_outputs.append(output.token_ids)

        if not all(token_ids == run_outputs[0] for token_ids in run_outputs):
            raise RuntimeError(
                f"Outputs differed between runs for {token_count} tokens"
            )

        median_duration = statistics.median(durations)
        generated_counts.append(output.num_generated_tokens)
        times.append(median_duration)
        outputs.append(output.token_ids)
        print(
            f"{output.num_generated_tokens:>4} tokens: {median_duration:.3f}s "
            f"(median of {NUM_RUNS} runs)"
        )

    return generated_counts, times, outputs


def save_results(
    naive_counts: list[int],
    naive_times: list[float],
    optimized_counts: list[int],
    optimized_times: list[float],
) -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["naive_tokens", "naive_seconds", "optimized_tokens", "optimized_seconds"]
        )
        writer.writerows(
            zip(naive_counts, naive_times, optimized_counts, optimized_times)
        )
    print(f"Saved results to {RESULTS_PATH}")


def plot_results(
    naive_counts: list[int],
    naive_times: list[float],
    optimized_counts: list[int],
    optimized_times: list[float],
) -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot([0] + naive_counts, [0] + naive_times, marker="o", label="Naive inference")
    plt.plot(
        [0] + optimized_counts,
        [0] + optimized_times,
        marker="o",
        label="Optimized inference",
    )
    plt.xlim(0, END)
    plt.ylim(bottom=0)
    plt.xlabel("Number of tokens generated")
    plt.ylabel("Generation time (seconds)")
    plt.title("Inference time with and without KV cache")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()
    print(f"Saved plot to {PLOT_PATH}")


def main() -> None:
    token_counts = list(range(STEP, END + 1, STEP))
    if token_counts[-1] != END:
        token_counts.append(END)

    naive_counts, naive_times, naive_outputs = benchmark(False, token_counts)
    optimized_counts, optimized_times, optimized_outputs = benchmark(True, token_counts)

    print(f"greedy outputs match: {naive_outputs == optimized_outputs}")
    save_results(naive_counts, naive_times, optimized_counts, optimized_times)
    plot_results(naive_counts, naive_times, optimized_counts, optimized_times)


if __name__ == "__main__":
    main()

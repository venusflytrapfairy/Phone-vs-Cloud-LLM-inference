# Phone vs Cloud: LLM Inference on a Budget Phone

Run a large language model entirely on an Android phone, benchmark it like an inference engineer, and race it against free cloud models. No laptop, no GPU, no money.

Then add your phone's results to the [community table](RESULTS.md) so we can see how different phones compare.

## What I found

Samsung Galaxy A14 5G (MediaTek Dimensity 700, 4 GB RAM), Qwen3-0.6B, llama.cpp in Termux.

| Experiment | Result |
|---|---|
| Phone vs cloud, time to first token (short prompts) | Phone 0.56 s vs cloud 0.55 to 1.05 s. A tie: no network round trip, no queue. |
| Phone vs cloud, generation speed | Phone 15 tok/s vs cloud 59 to 221 tok/s |
| 4-bit vs 8-bit quantization | 38% smaller, 22% faster prompt reading, no faster generation |
| 4 vs 8 CPU threads | 4 threads: 16% faster generation. 8 threads: 31% faster prompt reading. |

The surprise: shrinking the model didn't make it generate faster. For a model this small on a phone CPU, generation seems limited by compute and overhead rather than memory bandwidth. The phone's latency advantage also only holds for short prompts: a 500-token prompt with a 200-token answer would take roughly 25 to 28 s on the phone. Full analysis, concepts and comparison with published research in [REPORT.md](REPORT.md).

## How it works

```
Phone (Termux)
├── llama-server       runs the model, local only (127.0.0.1)
├── llama-bench        controlled, repeated speed tests
└── edge_vs_cloud.py   same prompts to the phone and free OpenRouter models;
                       measures time to first token and tokens/s, saves replies
```

Nothing is exposed to the internet. The model server only accepts connections from the phone itself.

## Run it yourself (about 30 minutes)

You need an Android phone with 4 GB+ RAM, about 2 GB free storage, and a charger. The cloud comparison needs a free [OpenRouter](https://openrouter.ai) key; the phone-only benchmarks don't.

### 1. Install Termux

Install Termux from [F-Droid](https://f-droid.org/packages/com.termux/), not the Play Store (the Play Store build is outdated). A warning that it "can't auto-update" is normal.

### 2. Install the tools

```bash
pkg update && pkg upgrade -y
pkg install -y python llama-cpp wget
pip install requests
termux-setup-storage
```

### 3. Stop Android from killing Termux

- Settings → Apps → Termux → Battery → Unrestricted
- Samsung: Settings → Battery → Background usage limits → Never sleeping apps → add Termux
- In Termux, run `termux-wake-lock`

### 4. Download the models

```bash
wget https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
wget https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf
ls -lh    # should show ~604M and ~373M, not KB
```

Hugging Face links must contain `/resolve/`, not `/blob/`. Only download quantized models from sources you trust.

### 5. Benchmark with llama-bench (the main results)

Let the phone cool for a few minutes before each run.

```bash
llama-bench -m Qwen3-0.6B-Q8_0.gguf -m Qwen3-0.6B-Q4_K_M.gguf -o md | tee bench_quant.md
llama-bench -m Qwen3-0.6B-Q4_K_M.gguf -t 2,4,8 -o md | tee bench_threads.md
```

`-o md` prints a markdown table; `tee` shows it and saves it to a file.

### 6. Race the cloud (optional)

Download `edge_vs_cloud.py` from this repo and copy it into Termux:

```bash
cp ~/storage/downloads/edge_vs_cloud.py .
```

In Termux session 1 (swipe from the left edge → NEW SESSION to make more):

```bash
llama-server -m Qwen3-0.6B-Q4_K_M.gguf -c 2048 -t 4 -tb 8
```

`-t` sets threads for generation and `-tb` for prompt reading. Tune both for your phone using step 5.

In session 2:

```bash
export OPENROUTER_API_KEY=sk-or-your-key     # no spaces around "="
clear                                         # keep your key out of screenshots
python edge_vs_cloud.py                       # phone + 3 free cloud models
OPENROUTER_API_KEY= python edge_vs_cloud.py   # phone only
```

Results print as a bar chart and save to `results.csv`. Every reply is also saved to `responses.md` for quality comparisons.

| Optional setting | Effect |
|---|---|
| `LABEL=Q4` | Names the phone model in the output |
| `MAX_TOKENS=600` | Longer replies (default 200, too short for quality tests) |
| `NO_THINK=1` | Turns off Qwen3's thinking on the phone model |
| `MODELS=a:free,b:free` | Chooses specific cloud models |

Example quality run: `LABEL=Q4 NO_THINK=1 MAX_TOKENS=600 python edge_vs_cloud.py`

When you're done: `pkill llama-server` and `termux-wake-unlock`.

### 7. Make charts (optional)

`visualize_results.ipynb` turns your results into charts. Open it in [Google Colab](https://colab.research.google.com) (works from a phone browser) or Jupyter, upload your results files, and run all cells.

## Contributing your results

1. Collect your device info in Termux:
   ```bash
   getprop ro.product.model     # phone model
   getprop ro.soc.model         # chip (Android 12+)
   grep MemTotal /proc/meminfo  # RAM
   ```
2. Fork this repo and create a folder `results/<your-github-name>-<phone-model>/` containing:
   - `bench_quant.md` and `bench_threads.md` from step 5
   - `results.csv` and `responses.md` from step 6 (optional)
   - any charts from step 7 (optional)
   - `notes.md`: phone, chip, RAM, location, anything unusual (hot phone, other apps running)
3. Add rows to [RESULTS.md](RESULTS.md) for your phone.
4. Open a pull request. New experiments (bigger models, other quantizations, long prompts, quality or safety tests) are welcome as new sections.

Never commit your API key. Check files before uploading.

## Troubleshooting

| Problem | Fix |
|---|---|
| `invalid magic characters: '<!do'` | You downloaded a web page. Use a `/resolve/` link and check the size with `ls -lh`. |
| Termux closes when you switch apps | Battery settings in step 3, close other apps, use split screen. |
| `couldn't bind HTTP server socket` | A server is already running. Run `pkill llama-server`. |
| `llama-server not reachable` | Start the server in another session first. |
| `HTTP 401` | Wrong OpenRouter key. |
| `HTTP 429` | Free model busy or daily limit reached. Wait and retry. |
| `export: '=' not a valid identifier` | Remove the spaces around `=`. |
| Reply marked "cut off" | Raise `MAX_TOKENS` or use `NO_THINK=1`. |
| Phone gets hot, speeds drop | Thermal throttling. Let it cool, then rerun. |

## Repo contents

| File | What it is |
|---|---|
| `README.md` | This page |
| `REPORT.md` | Full write-up: setup, concepts, results, comparison with research, reading list |
| `RESULTS.md` | Community results table |
| `edge_vs_cloud.py` | Phone vs cloud benchmark script |
| `visualize_results.ipynb` | Notebook for turning results into charts |
| `results/` | Raw benchmark output from each contributor |

## What's next

Quality and safety experiments (Q8 vs Q4 answer quality, over-refusal), with a more rigorous protocol: warm-up runs, repeated iterations, and thermal tracking. See section 10 of [REPORT.md](REPORT.md).

## Related research

- Chen et al. (2026), [On-device large language models: a survey of model compression and system optimization](https://link.springer.com/article/10.1007/s10462-026-11538-1), *Artificial Intelligence Review*
- Yan & Ding (2025), [Are We There Yet? A Measurement Study of Efficiency for LLM Applications on Mobile Devices](https://arxiv.org/abs/2504.00002)
- Çöplü et al. (2023), [A Performance Evaluation of a Quantized Large Language Model on Various Smartphones](https://arxiv.org/abs/2312.12472)

## Learn more

- [llama.cpp](https://github.com/ggml-org/llama.cpp), [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md), [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Transformer Inference Arithmetic](https://kipp.ly/blog/transformer-inference-arithmetic/): why generation is usually memory-bound
- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
- [LLM Inference Performance Engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices): TTFT, throughput, and related metrics

## License

MIT

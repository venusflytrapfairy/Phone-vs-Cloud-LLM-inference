# Community Results

Results from phones around the world. Add yours by opening a pull request (see "Contributing" in the README).

All numbers come from `llama-bench` (tokens per second, higher is better) unless marked otherwise. `pp512` = reading a 512-token prompt (prefill). `tg128` = generating 128 tokens (decode).

## Quantization: Q8_0 vs Q4_K_M

| Phone | Chip | RAM | Model | Quant | Threads | pp512 | tg128 | Contributor | Date |
|---|---|---|---|---|---|---|---|---|---|
| Galaxy A14 5G (SM-A146P) | Dimensity 700 | 4 GB | Qwen3-0.6B | Q8_0 | 8 | 35.22 ± 3.89 | 13.68 ± 1.52 | @your-github | 2026-10-14 |
| Galaxy A14 5G (SM-A146P) | Dimensity 700 | 4 GB | Qwen3-0.6B | Q4_K_M | 8 | 43.13 ± 1.87 | 13.63 ± 0.49 | @your-github | 2026-10-14 |

## Threads: 4 vs 8

| Phone | Chip | RAM | Model | Quant | Threads | pp512 | tg128 | Contributor | Date |
|---|---|---|---|---|---|---|---|---|---|
| Galaxy A14 5G (SM-A146P) | Dimensity 700 | 4 GB | Qwen3-0.6B | Q4_K_M | 4 | 32.25 ± 0.15 | 15.68 ± 0.63 | @your-github | 2026-10-14 |
| Galaxy A14 5G (SM-A146P) | Dimensity 700 | 4 GB | Qwen3-0.6B | Q4_K_M | 8 | 42.17 ± 1.33 | 13.49 ± 0.47 | @your-github | 2026-10-14 |

## Edge vs cloud (edge_vs_cloud.py, median of 3 short prompts)

| Phone | Location | Target | TTFT | tok/s | Contributor | Date |
|---|---|---|---|---|---|---|
| Galaxy A14 5G (SM-A146P) | Dubai, UAE | Phone: Qwen3-0.6B Q8_0, 8 threads | 0.56 s | 15.1 | @your-github | 2026-10-14 |
| Galaxy A14 5G (SM-A146P) | Dubai, UAE | Cloud: dots-studio/dots-3-note-preview:free | 1.05 s | 59.3 | @your-github | 2026-10-14 |
| Galaxy A14 5G (SM-A146P) | Dubai, UAE | Cloud: liquid/lfm-2.5-2.6b:free | 0.56 s | 220.6 | @your-github | 2026-10-14 |
| Galaxy A14 5G (SM-A146P) | Dubai, UAE | Cloud: nvidia/nemotron-3.5-lightning:free | 0.55 s | 90.3 | @your-github | 2026-10-14 |

## Findings so far

- **Q4 didn't speed up generation** on a 0.6B model (13.6 vs 13.7 tok/s), despite being 38% smaller. It did speed up prompt reading by 22%. Does this hold on your phone, or with a bigger model?
- **4 threads beat 8 for generation** (+16%), while 8 beat 4 for prompt reading (+31%). Where is the sweet spot on your chip?
- **The phone matched the cloud's time to first token on short prompts**, but generated text 4 to 15 times slower.

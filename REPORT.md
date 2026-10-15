# Phone vs Cloud: Learning Report

Notes from running a large language model (LLM) on my Samsung phone, benchmarking it, and racing it against free cloud models. This file is for learning and reference: what I built, what every command did, what I measured, what it means, and where to read more.

**Date:** 14 October 2026
**Device:** Samsung Galaxy A14 5G (SM-A146P), MediaTek Dimensity 700: 2 fast Cortex-A76 cores at 2.2 GHz + 6 efficiency Cortex-A55 cores at 2.0 GHz, 4 GB RAM (about 3.5 GB visible to Android)
**Tools:** Termux, llama.cpp (Termux package `llama-cpp` 0.6.0), Python 3.14, OpenRouter free models

---

## Contents

1. [Summary of findings](#1-summary-of-findings)
2. [What I built](#2-what-i-built)
3. [Setup, step by step](#3-setup-step-by-step)
4. [How the benchmark script works](#4-how-the-benchmark-script-works)
5. [Experiments and results](#5-experiments-and-results)
6. [Inference engineering concepts](#6-inference-engineering-concepts)
7. [Debugging log: what went wrong and why](#7-debugging-log-what-went-wrong-and-why)
8. [Limitations of these results](#8-limitations-of-these-results)
9. [How this compares with published research](#9-how-this-compares-with-published-research)
10. [Next experiments](#10-next-experiments)
11. [Further reading](#11-further-reading)
12. [Command cheat sheet](#12-command-cheat-sheet)

---

## 1. Summary of findings

| # | Finding | Evidence |
|---|---|---|
| 1 | On short prompts, the phone matched the cloud on time to first token. | Phone 0.56 s vs cloud 0.55 to 1.05 s (median of 3 prompts) |
| 2 | The cloud was 4 to 15 times faster at generating text. | Phone 15.1 tok/s vs cloud 59 to 221 tok/s |
| 3 | The phone reads prompts slowly, so its latency advantage disappears on long prompts. | Prefill 35 to 43 tok/s, so a 500-token prompt would wait roughly 12 to 14 s before the first word |
| 4 | 4-bit quantization made the model 38% smaller but did not speed up generation. | Q8 13.68 ± 1.52 vs Q4_K_M 13.63 ± 0.49 tok/s (llama-bench tg128) |
| 5 | 4-bit was 22% faster at reading prompts. | Q8 35.22 vs Q4_K_M 43.13 tok/s (llama-bench pp512) |
| 6 | Generation was faster on 4 threads; prompt reading was faster on 8. | tg128: 15.68 (4t) vs 13.49 (8t); pp512: 32.25 (4t) vs 42.17 (8t) |
| 7 | Quick tests can mislead. | A 3-prompt test on a hot phone said Q4 was *slower* than Q8; repeated benchmarks showed a tie |
| 8 | The first request after starting or idling is slow (cold start). | First prompt 3.3 to 6.7 s to first token, later prompts about 0.6 s |

---

## 2. What I built

```
 ┌──────────────────────── Samsung phone ────────────────────────┐
 │                                                               │
 │  Termux session 1                 Termux session 2            │
 │  ┌──────────────────────┐         ┌─────────────────────────┐ │
 │  │ llama-server         │  HTTP   │ edge_vs_cloud.py        │ │
 │  │ Qwen3-0.6B (GGUF)    │◀───────▶│ sends the same prompts  │ │
 │  │ 127.0.0.1:8080       │ (local) │ to every target, times  │ │
 │  └──────────────────────┘         │ each streamed token     │ │
 │                                   └───────────┬─────────────┘ │
 └───────────────────────────────────────────────┼───────────────┘
                                                 │ HTTPS (internet)
                                                 ▼
                                   ┌──────────────────────────┐
                                   │ OpenRouter API           │
                                   │ free cloud models (:free)│
                                   └──────────────────────────┘
```

- **The model server** (`llama-server`) loads the model file and exposes an OpenAI-compatible chat API. It listens on `127.0.0.1`, which means only the phone itself can connect. Nothing is exposed to the internet.
- **The benchmark script** sends identical prompts to the phone and to cloud models, streams the replies, and records when each token arrives.
- **`llama-bench`** is llama.cpp's own benchmarking tool. It runs controlled, repeated tests without any network or server involved, which made it the most reliable source of numbers.

---

## 3. Setup, step by step

### 3.1 Install Termux

Termux is a Linux terminal environment for Android. Install it from **F-Droid**, not Google Play; the Termux maintainers say the Play Store build is under development and has missing features.

F-Droid warns that Termux "can't auto-update because it was built for an older Android version." That is expected: Termux deliberately targets an older Android version, because newer versions restrict apps from running programs they downloaded, which is exactly what Termux needs to do. You update Termux packages from inside the app instead (`pkg upgrade`).

### 3.2 Install the software

```bash
pkg update && pkg upgrade -y        # refresh package lists, upgrade installed packages
pkg install -y python llama-cpp wget
pip install requests                # HTTP library used by the benchmark script
termux-setup-storage                # grant access to phone storage (creates ~/storage/...)
```

| Package | Role |
|---|---|
| `llama-cpp` | The inference engine. Provides `llama-server`, `llama-bench`, `llama-cli`. |
| `python` | Runs the benchmark script. |
| `wget` | Downloads model files. |
| `requests` | Lets Python make HTTP requests and read streamed responses. |

### 3.3 Download models

Models come from Hugging Face as **GGUF** files. The download link must contain `/resolve/` (the raw file).

```bash
# 8-bit version, official Qwen repo (~604 MiB)
wget https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf

# 4-bit version, unsloth repo (~373 MiB)
wget https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf

ls -lh    # check sizes: hundreds of MB, not KB
```

**Why Qwen3-0.6B:** it's small enough to run on almost any phone, free and not gated (no Hugging Face login needed to download).

### 3.4 Keep Android from killing Termux

Android, especially on Samsung phones, aggressively closes background apps to save memory and battery. A running model uses a lot of memory, which makes Termux a target.

- Settings → Apps → Termux → Battery → **Unrestricted**
- Settings → Battery → Background usage limits → **Never sleeping apps** → add Termux
- In Termux: `termux-wake-lock` (release later with `termux-wake-unlock`)

### 3.5 Start the model server

```bash
llama-server -m Qwen3-0.6B-Q8_0.gguf -c 2048
```

| Flag | Meaning |
|---|---|
| `-m` | Model file to load |
| `-c 2048` | Context size: the maximum number of tokens (prompt + reply) held in memory at once. Smaller uses less RAM. |
| (default host) | `127.0.0.1`, reachable only from the phone. Never add `--host 0.0.0.0` unless you intend to expose it. |

The warning *"no API key is set and CORS allows all origins"* only matters when the server is reachable from a network. Bound to `127.0.0.1`, it isn't.

Quick check from a second session, without a browser:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hi in 5 words"}],"max_tokens":50}'
```

### 3.6 Run the benchmark script

```bash
export OPENROUTER_API_KEY=sk-or-...     # no spaces around "="
clear                                   # wipe the key from the screen
python edge_vs_cloud.py                 # phone + 3 free cloud models
OPENROUTER_API_KEY= python edge_vs_cloud.py   # phone only (blanks the key for this one command)
```

### 3.7 Run llama-bench

Stop the server first so the two don't compete for CPU (`pkill llama-server`), and let the phone cool down.

```bash
llama-bench -m Qwen3-0.6B-Q8_0.gguf -m Qwen3-0.6B-Q4_K_M.gguf   # compare quantizations
llama-bench -m Qwen3-0.6B-Q4_K_M.gguf -t 4,8                    # compare thread counts
```

---

## 4. How the benchmark script works

`edge_vs_cloud.py` is about 150 lines of Python. The main ideas:

### 4.1 One API for every target

Both `llama-server` and OpenRouter speak the **OpenAI-compatible chat completions API** (`POST /v1/chat/completions`). So one function can benchmark the phone and the cloud; only the URL and the authorization header change. This is why the OpenAI API shape has become a de facto standard: it makes models swappable.

### 4.2 Streaming

The request sets `"stream": true`, so the server sends the reply token by token as **server-sent events (SSE)**: lines starting with `data:` each carrying a small JSON chunk, ending with `data: [DONE]`. Streaming is essential for measuring latency: without it, you only learn when the *whole* reply arrived.

### 4.3 What it measures

```
t0 ───────── t_first ──────────────────────────── t_end
   (request)  (first token)                        (last token)
   |── TTFT ──|──────── decode time ──────────────|
```

- **TTFT** = `t_first − t0`
- **tok/s** = `(tokens − 1) / (t_end − t_first)`. The first token is excluded because its time is already counted in TTFT, so this is pure decode speed.
- **Token count** comes from the `usage.completion_tokens` field (requested with `stream_options.include_usage`), falling back to counting chunks.
- Each target gets 3 prompts; the summary uses the **median**, which ignores one outlier such as a cold start.
- Cloud requests are spaced 3.5 s apart to stay under OpenRouter's free limit of 20 requests per minute.

### 4.4 The reasoning-model bug (fixed)

Qwen3 is a **thinking model**: it writes reasoning before its answer. `llama-server` streams that reasoning in a field called `reasoning_content`, which the first version of the script ignored. Result: when a reply was all thinking (200-token limit reached mid-thought), the script saw "no tokens," and when it did see an answer, it started the TTFT clock only after thinking finished, while counting all tokens, which inflated tok/s (e.g. a fake 47.8 tok/s).

Fix: count `content`, `reasoning`, *and* `reasoning_content` as generated tokens. Lesson: **validate a measurement tool before trusting its numbers.** A result that looks too good (or too bad) is a signal to check the tool.


| Setting | Effect |
|---|---|
| `LABEL=Q4` | Names the phone model in the output |
| `MAX_TOKENS=600` | Reply length limit (default 200, too short for quality tests) |
| `NO_THINK=1` | Turns off Qwen3's thinking on the phone model |
| `MODELS=a:free,b:free` | Chooses specific cloud models |

---

## 5. Experiments and results

### 5.1 Experiment 1: Phone vs cloud (Q8, 8 threads, fixed script)

Prompts were short (about 10 to 15 words, plus chat-template tokens). `max_tokens` = 200.

| Target | Prompt 1 | Prompt 2 | Prompt 3 | **Median TTFT** | **Median tok/s** |
|---|---|---|---|---|---|
| Phone: Qwen3-0.6B Q8_0 | 6.73 s · 14.3 | 0.56 s · 16.2 | 0.55 s · 15.1 | **0.56 s** | **15.1** |
| Cloud: dots-studio/dots-3-note-preview:free | 1.05 s · 58.0 | 1.05 s · 59.3 | 0.78 s · 64.7 | **1.05 s** | **59.3** |
| Cloud: liquid/lfm-2.5-2.6b:free | 0.56 s · 244.8 | 0.44 s · 201.5 | 1.12 s · 220.6 | **0.56 s** | **220.6** |
| Cloud: nvidia/nemotron-3.5-lightning:free | 0.55 s · 93.3 | 0.53 s · 58.8 | 0.97 s · 90.3 | **0.55 s** | **90.3** |

(Cloud measured from  UAE over home internet. Cloud TTFT includes the network round trip.)

**Interpretation**
- The phone's 0.56 s TTFT ties the fastest cloud models. It pays no network round trip and waits in no queue; for a short prompt, its slow prefill barely matters.
- The phone's prompt 1 (6.73 s) was a **cold start** (see 6.7).
- Cloud decode is 4 to 15 times faster: datacenter GPUs have far more memory bandwidth and compute.
- Free cloud models vary a lot (dots took 1 to 4 s to start across runs). Shared free capacity is not a stable baseline.

### 5.2 Experiment 2: Q8 vs Q4_K_M (llama-bench, 8 threads)

| Model | Size | pp512 (tok/s) | tg128 (tok/s) |
|---|---|---|---|
| Qwen3-0.6B Q8_0 | 604.15 MiB | 35.22 ± 3.89 | 13.68 ± 1.52 |
| Qwen3-0.6B Q4_K_M | 372.65 MiB | 43.13 ± 1.87 | 13.63 ± 0.49 |
| **Change Q8 → Q4** | **−38%** | **+22%** | **±0% (within noise)** |

- `pp512` = prompt processing: feed a 512-token prompt, measure tokens/s (prefill).
- `tg128` = text generation: generate 128 tokens, measure tokens/s (decode).
- `±` = standard deviation across repetitions. Overlapping ranges mean "no measurable difference."

**Interpretation:** see 6.4 for the full reasoning. In short, if generation were limited by memory bandwidth, a 38% smaller model should have generated noticeably faster. It didn't, so for this tiny model on this phone, the bottleneck during generation is elsewhere (compute, unpacking quantized weights, thread synchronization).

**An earlier, misleading result:** the 3-prompt script run on a warm phone gave Q4 a median of 12.2 tok/s and 0.90 s TTFT, which looked *slower* than Q8. Repeated, controlled `llama-bench` runs showed that was noise and heat, not a real difference.

### 5.3 Experiment 3: thread count (Q4_K_M, llama-bench)

| Threads | pp512 (tok/s) | tg128 (tok/s) |
|---|---|---|
| 4 | 32.25 ± 0.15 | **15.68 ± 0.63** |
| 8 | **42.17 ± 1.33** | 13.49 ± 0.47 |
| **Winner** | 8 threads (+31%) | 4 threads (+16%) |

**Interpretation:** see 6.6. Prefill is a big parallel batch, so more cores help. Decode is one token at a time, small work per step, where slow efficiency cores and synchronization costs drag the fast cores down.

**Resulting config (not yet benchmarked):**

```bash
llama-server -m Qwen3-0.6B-Q4_K_M.gguf -c 2048 -t 4 -tb 8
```

`-t` sets threads for generation; `-tb` sets threads for batch and prompt processing.

---

## 6. Inference engineering concepts

### 6.1 Inference, tokens, parameters

- **Training** creates a model; **inference** is using it to produce output. Inference engineering is about making that fast, cheap, and reliable.
- A **token** is the unit a model reads and writes, roughly ¾ of an English word. Speeds are measured in tokens per second.
- **Parameters** are the learned numbers in the model. "0.6B" means about 600 million. Each parameter must be stored, and (for a dense model) used for every generated token.

### 6.2 The two phases: prefill and decode

Generating a reply happens in two very different phases:

| | Prefill (prompt processing) | Decode (token generation) |
|---|---|---|
| What happens | The model reads the whole prompt at once | The model produces one token at a time, each depending on the last |
| Parallelism | High: all prompt tokens processed together | Low: strictly sequential |
| Typical bottleneck | Compute (math throughput) | Memory bandwidth (reading the weights), on most hardware |
| Shows up as | Time to first token | Tokens per second after that |
| My phone | 32 to 43 tok/s | 13 to 16 tok/s |

**Sanity check:** my prompts were roughly 20 tokens after the chat template was added. At about 35 tok/s prefill, that's about 0.5 to 0.6 s, matching the measured 0.56 s TTFT. A 500-token prompt would take roughly 500 ÷ 35 to 43 ≈ **12 to 14 s** before the first token.

### 6.3 Latency and throughput metrics

- **TTFT (time to first token):** how long the user waits before anything appears. Dominated by network, queueing, and prefill.
- **TPOT (time per output token)** or its inverse, **tokens/s per user:** how fast text streams after it starts. Dominated by decode.
- **Throughput:** total tokens/s a server produces across *all* users. Servers batch many users together to raise throughput, sometimes at the cost of per-user latency. My phone served one user, so per-user speed and throughput were the same.

### 6.4 Memory-bound vs compute-bound

Every computation is limited by one of three things: how fast the processor can do math (**compute**), how fast it can move data from memory (**memory bandwidth**), or fixed costs like launching work and coordinating threads (**overhead**).

During decode, a dense model has to read essentially all its weights from memory for every token, while doing relatively little math per weight. That's why decode is usually **memory-bandwidth bound**, and why quantization (smaller weights) usually speeds it up.

**My data tests this.** Rough effective bandwidth = model size × tokens per second:

| Model | Size | tg128 | Effective bandwidth |
|---|---|---|---|
| Q8_0 | ~634 MB | 13.68 tok/s | ~8.7 GB/s |
| Q4_K_M | ~391 MB | 13.63 tok/s | ~5.3 GB/s |

Q8 already pulled about 8.7 GB/s, so the phone *can* move data that fast. If bandwidth were the limit, Q4, needing only 61% of the data per token, should have run at roughly 22 tok/s. It stayed at 13.6. So on this phone, for this tiny model, decode is limited by something else: likely the compute to unpack the more complex Q4_K format, plus per-token overhead and thread synchronization. (These numbers are approximate: not every weight is read identically per token. The conclusion holds anyway.)

**Why prefill still improved with Q4:** prefill reuses each weight across many tokens at once, so the arithmetic dominates, and the specific kernels for each format perform differently. The general lesson: **measure which phase and which resource is the bottleneck before optimizing.** Quantization isn't automatically "faster."

### 6.5 Quantization, GGUF, and K-quants

- **Quantization** stores weights with fewer bits. Original models usually use 16 bits per weight. Q8 uses about 8, Q4 about 4 to 5.
- Fewer bits mean smaller files, less RAM, and (usually) faster memory-bound decode, at some cost in accuracy. Larger models typically tolerate quantization better than tiny ones.
- **GGUF** is llama.cpp's single-file format: weights plus metadata (architecture, tokenizer, chat template). A real GGUF file starts with the bytes `GGUF`, which is why llama.cpp rejected my HTML download with *"invalid magic characters: '<!do'"*.
- **Q8_0:** simple 8-bit, blocks of 32 weights sharing one scale. Easy and cheap to unpack.
- **Q4_K_M ("K-quant"):** a more elaborate scheme of super-blocks containing blocks, with extra scales, averaging about 4.5 bits per weight. "M" (medium) means some sensitive layers are kept at higher precision. Better quality per bit, but more work to unpack.

### 6.6 Threads and heterogeneous cores

Phone processors use **big.LITTLE** (heterogeneous) designs: a few fast "big" cores plus several power-efficient "little" cores. My Dimensity 700 has only **2 fast cores (Cortex-A76) and 6 slow ones (Cortex-A55)**, so even 4 threads already includes at least 2 slow cores.

- With 8 threads, work is split across all cores, including slow ones. In decode, each token's work is small and must be finished by *every* thread before the next token can start, so the fast cores wait for the slow ones and threads spend time synchronizing. Result: 4 threads beat 8 on decode.
- Prefill has enough work per step to keep all cores busy, so 8 threads beat 4.
- llama.cpp exposes both settings separately (`-t` for generation, `-tb` for batch/prompt).

### 6.7 Cold starts and memory mapping

llama.cpp opens model files with **mmap** (memory mapping): the file is mapped into memory, but its pages are only read from storage when first touched. So the first request pays for loading the weights from flash storage, which explains the 3.3 to 6.7 s first-prompt TTFT. After an idle period, Android may reclaim those pages, making the next request "cold" again.

In production systems, this is why services keep models **warm**, pre-load them, or send warm-up requests, and why benchmarks discard the first run.

### 6.8 Context size and the KV cache

During generation, the model stores intermediate results (the **keys and values** from attention) for every token so far, so it doesn't recompute them for each new token. This **KV cache** grows with the number of tokens in the conversation. The `-c` (context size) flag caps it; that's why a smaller context saves RAM. RAM needed ≈ model file + KV cache + overhead.

### 6.9 Thermal throttling

Phones have no fans. Under sustained load the chip heats up, and Android's thermal management lowers clock speeds to protect the hardware. Effects in my data: speeds drifting downward across runs (e.g. prompt 3 dropping to 9.0 tok/s on a warm phone). Mitigation for benchmarking: cool down between runs, repeat tests, compare back to back.

### 6.10 Reasoning (thinking) models

Qwen3 writes a hidden-style reasoning section before its answer. Those reasoning tokens cost real generation time. For a user, "time to first *answer* token" can be much longer than "time to first token." Different servers expose reasoning in different fields (`reasoning`, `reasoning_content`), so benchmark tools must handle them. Qwen3's thinking can be turned off with `/no_think` in the prompt or `enable_thinking=False` in the chat template.

### 6.11 Measurement methodology

- **Repeat** each test and report spread (± standard deviation). llama-bench does this; my script uses medians of 3.
- **Control variables:** same prompts, same `max_tokens`, same temperature of the device, nothing else running.
- **Separate phases:** measure prefill and decode separately (pp vs tg) instead of only end-to-end time.
- **Validate tools:** my script's bug produced plausible-looking wrong numbers.
- **Note conditions:** network location, time of day, and free-tier load all affect cloud numbers.

### 6.12 Rate limits and free tiers

OpenRouter free models (IDs ending in `:free`) allow 20 requests per minute, and 50 requests per day for accounts that have bought fewer than 10 credits (1,000 per day after buying at least 10). Free capacity is donated by providers and can be busy (HTTP 429) or removed (HTTP 404). One full script run uses 9 cloud requests.

### 6.13 Security basics

- Bind local servers to `127.0.0.1` unless you mean to expose them.
- API keys are passwords: never paste them in chats, screenshots, or code. If exposed, delete the key and create a new one (I did this once during the session).
- `export` keeps the key in the current terminal session's memory only.

---

## 7. Debugging log: what went wrong and why

| Symptom | Cause | Fix | Lesson |
|---|---|---|---|
| `invalid magic characters: '<!do', expected 'GGUF'` | Downloaded a `/blob/` link, which saved the Hugging Face web page as a `.gguf` file | Use `/resolve/` links; check size with `ls -lh` | Check what you downloaded; file headers ("magic bytes") identify formats |
| Termux restarted when opening the browser | Android killed Termux to free memory | Battery "Unrestricted", "Never sleeping apps", `termux-wake-lock`, split screen | Mobile OSes aggressively kill background processes |
| `cp: cannot stat '/Internal'` | "Internal storage" is a display name; spaces split arguments; no destination given | `cp ~/storage/downloads/file .` | Shell arguments are split on spaces; quote paths with spaces |
| `export: '=' not a valid identifier` | Spaces around `=` | `export NAME=value` | Shell assignment syntax allows no spaces |
| `no tokens returned`, inflated tok/s | Script ignored `reasoning_content` | Patched with `sed -i` | Validate measurement tools |
| `couldn't bind HTTP server socket ... port: 8080` | Old server still running on the same port | `pkill llama-server` | Only one process can listen on a port |
| Script behaved like the old version | Ran an unpatched copy in a different folder | Re-applied the patch, checked with `grep` | Know which copy of a file you're running |
| "Q4 is slower than Q8" | Small sample (3 prompts), warm phone | Repeated, controlled `llama-bench` | Don't trust a single noisy measurement |

---

## 8. Limitations of these results

- **One device, one model family.** Results may differ on other phones (different chips, cores, memory) and bigger models, where Q4 usually *does* speed up decode.
- **Tiny model.** 0.6B is far smaller than the cloud models, so this isn't a quality comparison, only speed.
- **Short prompts** in the script. TTFT comparisons don't hold for long prompts (see 6.2).
- **Cloud results depend on location and load.** Measured from the UAE on free shared capacity at one time of day.
- **Few repetitions** in the script (3 prompts per target). llama-bench numbers are more reliable.
- **Thermal state** was not controlled precisely.
- The `-t 4 -tb 8` combination was derived from separate tests and not benchmarked as a pair.

---

## 9. How this compares with published research

**Chen et al. (2026), on-device LLM survey.** The survey argues that on-device inference is limited by memory more than by compute (the "memory wall"), in two ways:
- *Capacity:* big models simply don't fit, so compression is a prerequisite. This was my hardest constraint: with about 3.5 GB of usable RAM, a 0.6B model was the realistic choice.
- *Bandwidth:* prefill is compute-bound, while decode is limited by how fast weights can be read from memory. My thread results fit the first half: prefill sped up 31% with more cores, as compute-bound work should. The second half didn't show up cleanly: Q4 moved 38% less data per token but decoded no faster (section 6.4). The survey itself notes that real quantization speedups depend on kernels, compiler support and scheduling, and recommends reporting results together with latency, memory and kernel details. A tiny model on a phone CPU, using a quantization format that's costly to unpack, looks like a case where those costs dominate. I'd expect bandwidth limits to show up with bigger models, which is testable.
- *A–L–E–M:* they judge on-device systems on Accuracy, Latency, Energy and Memory. I only measured Latency so far.
- Two of their practical rules match my setup: start with 4-bit post-training quantization when memory is tight, and use edge-first with cloud fallback when connectivity matters, which is the trade-off my phone vs cloud race measured.

**Yan & Ding (2025), "Are We There Yet?"** They found only models under 4B parameters ran on powerful phones, that compression lowers hardware requirements but can hurt quality, and that phones needed over 30 s for meaningful output versus under 10 s in the cloud.
- My budget phone is consistent: 0.6B was the practical size.
- My phone only *tied* the cloud on time to first token because my prompts and answers were short. With my measured speeds, a 500-token prompt plus a 200-token answer would take roughly 25 to 28 s on the phone, close to their figure. Their finding describes realistic workloads better than my short-prompt test.
- I haven't measured the quality cost of compression yet, which is the next step.

**Çöplü et al. (2023), quantized 7B model on five iPhones.** They tracked iOS thermal states across repeated runs and found performance fell sharply in the critical state, with generation hurt more than prompt processing as the phone heated up. My heat-affected Q4 run (prompt 3 dropping to 9.0 tok/s) is a small, informal version of the same effect. Their protocol is more rigorous than mine, and I'll adopt it next.

---

## 10. Next experiments

Next I want to run **quality and safety experiments**, using a more rigorous methodology inspired by Çöplü et al. (2023):
- A warm-up run before measuring, then 3 runs × 3 iterations per configuration, averaged.
- A fixed delay between iterations, context reset each time, and fixed (greedy) sampling.
- Full battery, no background apps, and the phone's thermal condition recorded across iterations.

Ideas from Chen et al. (2026) to add:
- Cover all of A–L–E–M: add energy per token and peak memory alongside latency, and a small standard accuracy benchmark.
- Report runtime details (llama.cpp version, quantization format, threads) with every result.
- Test a larger model to see whether decode becomes bandwidth-bound and Q4 starts to help.
- Measure how memory and speed change as context length grows (KV cache).

---

## 11. Further reading

Documentation, articles and specs for further reading.

**Research papers**
- [Chen et al. (2026), On-device large language models: a survey of model compression and system optimization](https://link.springer.com/article/10.1007/s10462-026-11538-1), *Artificial Intelligence Review* 59, 191
- [Yan & Ding (2025), Are We There Yet? A Measurement Study of Efficiency for LLM Applications on Mobile Devices](https://arxiv.org/abs/2504.00002)
- [Çöplü et al. (2023), A Performance Evaluation of a Quantized Large Language Model on Various Smartphones](https://arxiv.org/abs/2312.12472)

**Tools used**
- [llama.cpp repository](https://github.com/ggml-org/llama.cpp): the inference engine
- [llama.cpp on Android](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md): official notes, including building in Termux
- [llama-server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md): every server flag, including `-t` and `-tb`
- [llama-bench README](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md): how pp and tg tests work
- [Termux app](https://github.com/termux/termux-app): install sources and known Android issues
- [OpenRouter rate limits](https://openrouter.ai/docs/limits) and [free model variants](https://openrouter.ai/docs/guides/routing/model-variants/free)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B): thinking mode and how to disable it

**Concepts**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Jay Alammar): how the model architecture works, visually
- [Transformer Inference Arithmetic](https://kipp.ly/blog/transformer-inference-arithmetic/) (Carol Chen): the math of memory-bound vs compute-bound inference and the KV cache
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) (Horace He): compute vs memory bandwidth vs overhead
- [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) (Databricks): TTFT, time per output token, throughput, batching
- [KV Caching Explained](https://huggingface.co/blog/not-lain/kv-caching) (Hugging Face blog)
- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) (Maarten Grootendorst)
- [GGUF on Hugging Face](https://huggingface.co/docs/hub/gguf): the format and its quantization types, with bits per weight
- [k-quants pull request](https://github.com/ggml-org/llama.cpp/pull/1684): the original design of Q4_K and friends

**Systems background**
- [Arm big.LITTLE](https://www.arm.com/technologies/big-little): heterogeneous phone cores
- [Android thermal mitigation](https://source.android.com/docs/core/power/thermal-mitigation): how Android throttles hot devices
- [mmap(2) man page](https://man7.org/linux/man-pages/man2/mmap.2.html): memory-mapped files and lazy loading
- [Using server-sent events (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events): the streaming format behind token-by-token responses

---

## 12. Command cheat sheet

```bash
# Termux basics
pkg update && pkg upgrade -y
pkg install -y python llama-cpp wget
pip install requests
termux-setup-storage
termux-wake-lock            # keep Termux alive; termux-wake-unlock when done

# Files
ls -lh                      # list files with readable sizes
cd ~                        # go to home folder
cp ~/storage/downloads/edge_vs_cloud.py .
mv results.csv results_q4.csv
rm bad_file.gguf

# Models
wget https://huggingface.co/<repo>/resolve/main/<file>.gguf

# Serve (local only)
llama-server -m <model>.gguf -c 2048 -t 4 -tb 8
pkill llama-server          # stop any running server

# Benchmark
python edge_vs_cloud.py                       # phone + cloud
OPENROUTER_API_KEY= python edge_vs_cloud.py   # phone only
LABEL=Q4 NO_THINK=1 MAX_TOKENS=600 python edge_vs_cloud.py   # quality run, replies saved to responses.md
llama-bench -m <a>.gguf -m <b>.gguf           # compare models
llama-bench -m <model>.gguf -t 4,8            # compare thread counts

# Device info for reports
getprop ro.product.model     # phone model
getprop ro.soc.model         # chip (Android 12+)
grep MemTotal /proc/meminfo  # RAM
nproc                        # CPU cores
```

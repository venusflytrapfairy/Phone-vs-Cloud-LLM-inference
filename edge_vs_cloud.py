#!/usr/bin/env python3
"""Edge vs Cloud: benchmark a model running on your phone against free OpenRouter models.

Measures, per request:
  TTFT  - time to first token (latency the user feels)
  tok/s - decode speed after the first token (throughput)
Run in Termux. Needs: llama-server running locally on port 8080, OPENROUTER_API_KEY set.
"""
import csv, json, os, statistics, sys, time
import requests

LOCAL_URL = "http://127.0.0.1:8080/v1/chat/completions"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
OR_KEY = os.environ.get("OPENROUTER_API_KEY")
N_CLOUD = 3          # how many free cloud models to test
MAX_TOKENS = 200
PAUSE = 3.5          # seconds between cloud calls (free tier = 20 req/min)

PROMPTS = [
    "Explain what a KV cache is in two sentences.",
    "Write a short haiku about a phone running an AI model.",
    "List 5 creative names for a coffee shop run by robots.",
]


def free_models(n):
    """Pick free models from OpenRouter, or use MODELS=a,b,c if you set it."""
    if os.environ.get("MODELS"):
        return [m.strip() for m in os.environ["MODELS"].split(",") if m.strip()]
    r = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
    ids = [m["id"] for m in r.json()["data"] if m["id"].endswith(":free")]
    return ids[:n]


def run(url, model, prompt, headers):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    t_first, chunks, usage_tokens = None, 0, None
    try:
        with requests.post(url, json=body, headers=headers, stream=True, timeout=180) as r:
            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}: {r.text[:100]}"}
            for raw in r.iter_lines():
                line = raw.decode("utf-8", "ignore")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage_tokens = obj["usage"].get("completion_tokens")
                for ch in obj.get("choices", []):
                    d = ch.get("delta", {})
                    if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                        if t_first is None:
                            t_first = time.perf_counter()
                        chunks += 1
    except requests.RequestException as e:
        return {"error": str(e)[:100]}
    t_end = time.perf_counter()
    if t_first is None:
        return {"error": "no tokens returned"}
    toks = usage_tokens or chunks
    gen = t_end - t_first
    return {
        "ttft": t_first - t0,
        "tokens": toks,
        "tps": (toks - 1) / gen if gen > 0 and toks > 1 else 0.0,
        "total": t_end - t0,
    }


def main():
    targets = []
    try:
        requests.get("http://127.0.0.1:8080/health", timeout=3)
        targets.append(("PHONE (on-device)", LOCAL_URL, "local", {}))
    except requests.RequestException:
        print("! llama-server not reachable on 127.0.0.1:8080, skipping on-device run")

    if OR_KEY:
        hdr = {"Authorization": f"Bearer {OR_KEY}"}
        for m in free_models(N_CLOUD):
            targets.append((f"CLOUD {m}", OR_URL, m, hdr))
    else:
        print("! OPENROUTER_API_KEY not set, skipping cloud models")

    if not targets:
        sys.exit("Nothing to benchmark.")

    rows = []
    for label, url, model, hdr in targets:
        print(f"\n== {label}")
        for i, p in enumerate(PROMPTS, 1):
            res = run(url, model, p, hdr)
            if "error" in res:
                print(f"  prompt {i}: ERROR {res['error']}")
            else:
                print(f"  prompt {i}: TTFT {res['ttft']:.2f}s | {res['tps']:.1f} tok/s | {res['tokens']} tokens")
                rows.append({"target": label, "prompt": i, **res})
            if url == OR_URL:
                time.sleep(PAUSE)

    if not rows:
        sys.exit("\nNo successful runs.")

    summary = []
    for label, *_ in targets:
        rs = [r for r in rows if r["target"] == label]
        if rs:
            summary.append((label,
                            statistics.median(r["ttft"] for r in rs),
                            statistics.median(r["tps"] for r in rs)))

    best_tps = max(s[2] for s in summary) or 1
    best_ttft = max(s[1] for s in summary) or 1
    print("\n" + "=" * 56)
    print("  EDGE vs CLOUD  (median of", len(PROMPTS), "prompts)")
    print("=" * 56)
    for label, ttft, tps in summary:
        print(f"\n{label[:50]}")
        print(f"  TTFT  {ttft:6.2f}s   " + "#" * max(1, int(20 * ttft / best_ttft)))
        print(f"  speed {tps:6.1f} t/s  " + "=" * max(1, int(20 * tps / best_tps)))
    print("\n(TTFT: shorter bar = better | speed: longer bar = better)")

    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "prompt", "ttft", "tokens", "tps", "total"])
        w.writeheader()
        w.writerows(rows)
    print("Saved results.csv")


if __name__ == "__main__":
    main()

import warnings

# macOS CLT Python links LibreSSL; urllib3 v2 warns but HTTPS still works for typical use.
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+.*",
)

import requests
import concurrent.futures
import time
import sys
import os
import statistics
import random
import argparse
import threading

# ================= Basic config =================
DEFAULT_GATEWAY_URL = "http://192.168.1.145"


def resolve_gateway_candidates(cli_value):
    """Return ordered base URLs to try. If input has no scheme, try http then https."""
    raw = (cli_value or os.environ.get("MODEL_STRESS_GATEWAY_URL") or DEFAULT_GATEWAY_URL).strip()
    raw = raw.rstrip("/")
    lower = raw.lower()
    if lower.startswith("http://") or lower.startswith("https://"):
        return [raw]
    return [f"http://{raw}", f"https://{raw}"]

def get_args():
    parser = argparse.ArgumentParser(description="vLLM CPU cluster stress test tool")
    parser.add_argument(
        "-r", "--random",
        action="store_true",
        help="Enable random prefix mode (test cache MISS)",
    )
    # -c concurrency
    parser.add_argument("-c", "--concurrency", type=int, default=10, help="Concurrent requests (default: 10)")
    # -t Token length
    parser.add_argument("-t", "--tokens", type=int, default=512, help="Max generation tokens (default: 512)")
    parser.add_argument(
        "-T",
        "--timeout",
        type=float,
        default=600.0,
        metavar="SEC",
        help=(
            "Timeout (seconds): idle window that resets on each successful response, "
            "and per-request HTTP timeout. Default: 600."
        ),
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="After each success, print the model completion text in a fixed delimiter block (with thread-safe ordering).",
    )
    parser.add_argument(
        "-p",
        "--prompt-repeats",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Repeat the one-line prompt N times with spaces (scales input length; "
            "values below 1 are treated as 1). Default: 1."
        ),
    )
    parser.add_argument(
        "-g",
        "--gateway-url",
        default=None,
        metavar="URL",
        help=(
            "OpenAI-compatible gateway base URL (no trailing path). "
            "Host or host:port may omit scheme (try http then https). "
            "Overrides MODEL_STRESS_GATEWAY_URL; default is built-in if unset."
        ),
    )
    return parser.parse_args()


def _assistant_text_from_completion(data):
    """Extract assistant message text from OpenAI-compatible chat completion JSON."""
    try:
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(part.get("text") or "")
                    else:
                        parts.append(str(part))
                else:
                    parts.append(str(part))
            return "\n".join(parts)
        if content is None:
            return ""
        return str(content)
    except (KeyError, IndexError, TypeError):
        return ""

def get_active_model(gateway_candidates):
    last_err = None
    for base in gateway_candidates:
        print(f"[INFO] Trying gateway {base} ...")
        try:
            res = requests.get(f"{base}/v1/models", timeout=10)
            model_id = res.json()["data"][0]["id"]
            print(f"[INFO] Using gateway {base}")
            return model_id, base
        except Exception as e:
            last_err = e
            print(f"[WARN] {base} unavailable: {e}")
    print(f"[FAIL] Could not fetch model info from any URL. Last error: {last_err}")
    sys.exit(1)

def generate_prompt(is_random, prompt_repeats=1):
    """One-sentence user prompt; optional repeats join with spaces for length tests."""
    if is_random:
        line = (
            f"Stress test ping (session {random.randint(100000, 999999)}), "
            "reply with the single word OK and nothing else."
        )
    else:
        line = "Stress test ping, reply with the single word OK and nothing else."

    repeats = max(1, int(prompt_repeats))
    return " ".join([line] * repeats)

# ---------------------------------------------------------
# send_request uses the generator above
# ---------------------------------------------------------

def _format_debug_model_output(req_id, text):
    return (
        f"======== DEBUG MODEL OUTPUT req={req_id:02d} ========\n"
        f"{text}\n"
        f"======== END DEBUG MODEL OUTPUT req={req_id:02d} ========"
    )


def send_request(
    req_id,
    model_name,
    gateway_url,
    is_random,
    max_tokens,
    request_timeout,
    prompt_repeats,
    debug=False,
    debug_lock=None,
):
    # [Opt 1] Random startup delay (jitter) so NGINX can refresh connection counts
    time.sleep(random.uniform(0.1, 0.8))

    prompt = generate_prompt(is_random, prompt_repeats)

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0
    }

    # [Opt 2] Force Connection: close (no keep-alive reuse)
    headers = {
        "Connection": "close"
    }

    start_time = time.time()
    try:
        response = requests.post(
            f"{gateway_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=request_timeout,
        )
        end_time = time.time()

        if response.status_code == 200:
            data = response.json()
            tokens = data["usage"]["completion_tokens"]
            latency = end_time - start_time
            this_tps = tokens / latency if latency > 0 else 0

            ok_line = (
                f"[OK] [req {req_id:02d}] done | tokens: {tokens} | "
                f"latency: {latency:.2f}s | TPS: {this_tps:.2f}"
            )
            if debug:
                body = _assistant_text_from_completion(data)
                block = _format_debug_model_output(req_id, body)
                with debug_lock:
                    print(ok_line, flush=True)
                    print(block, flush=True)
            else:
                print(ok_line, flush=True)
            return {"status": 1, "tokens": tokens, "latency": latency}
        else:
            print(f"[FAIL] [req {req_id:02d}] HTTP {response.status_code}")
            return {"status": 0}
    except Exception as e:
        print(f"[WARN] [req {req_id:02d}] error: {e}")
        return {"status": 0}


def main():
    args = get_args()
    candidates = resolve_gateway_candidates(args.gateway_url)
    model_name, gateway_url = get_active_model(candidates)

    mode_info = "[RANDOM] cache MISS" if args.random else "[STATIC] cache HIT"

    print("\n" + "=" * 60)
    print(
        f"[START] cluster stress | concurrency: {args.concurrency} | "
        f"prompt_repeats: {max(1, args.prompt_repeats)} | model: {model_name}"
    )
    print(f"[MODE] {mode_info}")
    print("=" * 60 + "\n")

    idle_limit = args.timeout
    last_ok_lock = threading.Lock()
    last_ok = {"t": time.monotonic()}

    def touch_success():
        with last_ok_lock:
            last_ok["t"] = time.monotonic()

    debug_lock = threading.Lock() if args.debug else None

    def run_one(req_id):
        r = send_request(
            req_id,
            model_name,
            gateway_url,
            args.random,
            args.tokens,
            args.timeout,
            args.prompt_repeats,
            debug=args.debug,
            debug_lock=debug_lock,
        )
        if r.get("status") == 1:
            touch_success()
        return r

    total_start = time.time()
    results = []
    idle_abort = False
    pending = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        pending = {
            executor.submit(run_one, i): i
            for i in range(args.concurrency)
        }
        while pending:
            done, _ = concurrent.futures.wait(
                pending.keys(),
                timeout=1.0,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                results.append(fut.result())
                del pending[fut]
            now = time.monotonic()
            with last_ok_lock:
                deadline = last_ok["t"]
            if now - deadline > idle_limit:
                print(f"\n[FAIL] Idle timeout: no success within {idle_limit:.0f}s (timer resets on each success).")
                idle_abort = True
                break

    total_end = time.time()

    for fut in pending:
        fut.cancel()

    success = [r for r in results if r["status"] == 1]
    if not success:
        print("\n[FAIL] Stress run failed: no successful requests.")
        return

    total_tokens = sum(r["tokens"] for r in success)
    total_duration = total_end - total_start
    cluster_tps = total_tokens / total_duration

    print("\n" + "-" * 60)
    print("[SUMMARY] Stress test report")
    if idle_abort:
        print(f"[NOTE] Stopped early: {len(results)}/{args.concurrency} requests finished before idle timeout.")
    print(f"[STAT] success: {len(success)}/{args.concurrency}")
    print(f"[TIME] total: {total_duration:.2f} s")
    print(f"[THROUGHPUT] cluster: {cluster_tps:.2f} tokens/s")
    print(f"[LATENCY] mean per request: {statistics.mean(r['latency'] for r in success):.2f} s")
    print("-" * 60 + "\n")

if __name__ == "__main__":
    main()

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

__version__ = "1.0.0"
__updated__ = "2026-03-30"
__author__ = "chris.li@arista.com"
__company__ = "Arista Network"

PROMPT_PRESETS = {
    "short": (
        "Write an original English science fiction short story of about 500 words "
        "about human migration to or settlement on Mars. Output story prose only, with no preamble."
    ),
    "medium": (
        "Write an original English science fiction short story of about 500 words "
        "about human migration to or settlement on Mars (not an outline or review). "
        "Structure the narrative with a clear beginning, development, and resolution. "
        "Ground characters and conflict in the Mars setting. Aim for roughly 450–550 words. "
        "Output prose only: do not use lead-in phrases (for example, 'Sure, here is') "
        "and do not add a separate title line unless it is embedded in the narrative."
    ),
    "long": (
        "Write an original English science fiction short story about human migration to or settlement on Mars.\n\n"
        "Requirements:\n"
        "- Length: about 450–550 words (target ~500 words).\n"
        "- Form: complete short fiction with setup, development, and resolution—not an outline, synopsis, or book review.\n"
        "- Setting: the story must be anchored in Mars—domes, transit from Earth, terraforming, society on Mars, or Earth–Mars tension.\n"
        "- Voice: third person or first person is fine; keep the tone serious or lightly adventurous science fiction, not comedy skit.\n"
        "- Output: story prose only. Do not preface with meta-commentary. Do not include a standalone title line; you may name places in the text.\n"
        "- Do not break the fourth wall or address the reader about the assignment.\n\n"
        "Begin the story immediately with narrative prose."
    ),
}

MARS_ANGLE_FOCUS = [
    "first-generation life inside a pressurized dome",
    "dispute over water-ice mining rights",
    "Earth–Mars communications delay shaping a relationship or decision",
    "a failed or difficult terraforming season",
    "identity of someone born on Mars versus ties to Earth",
    "arrival of a new wave of immigrants at a Martian settlement",
    "political tension between Earth governance and Martian settlers",
    "a generation ship's final approach and integration with surface life",
]


def resolve_gateway_candidates(cli_value):
    """Return ordered base URLs to try. If input has no scheme, try http then https."""
    raw = (cli_value or os.environ.get("MODEL_STRESS_GATEWAY_URL") or DEFAULT_GATEWAY_URL).strip()
    raw = raw.rstrip("/")
    lower = raw.lower()
    if lower.startswith("http://") or lower.startswith("https://"):
        return [raw]
    return [f"http://{raw}", f"https://{raw}"]

def get_args():
    _epilog = (
        f"Author: {__author__}\n"
        f"Company: {__company__}\n"
        f"Version: {__version__}\n"
        f"Last updated: {__updated__}"
    )
    parser = argparse.ArgumentParser(
        description="LLM cluster stress test tool",
        epilog=_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__} ({__updated__})",
    )
    parser.add_argument(
        "-r", "--random",
        action="store_true",
        help="Enable random prefix mode (test cache MISS)",
    )
    # -c concurrency
    parser.add_argument("-c", "--concurrency", type=int, default=10, help="Concurrent requests (default: 10)")
    # -t Token length (~500 English words needs roughly 650–800+ completion tokens; default leaves headroom)
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        default=1024,
        help=(
            "Max generation tokens (default: 1024). For ~500 English words use at least ~1024; "
            "increase for long prompts or longer outputs."
        ),
    )
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
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose per-request lines: tokens, latency, TPS.",
    )
    output_group.add_argument(
        "-b",
        "--brief",
        action="store_true",
        help=(
            "Brief per-request output: success '!', failure '.' (compact, like ping). "
            "Default when neither -v nor -b is given."
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
            "Repeat the full user prompt N times joined with spaces (scales input length; "
            "values below 1 are treated as 1). Default: 1."
        ),
    )
    parser.add_argument(
        "--prompt-size",
        choices=("short", "medium", "long"),
        default="medium",
        help=(
            "Built-in English Mars-migration story prompt length (input token load). "
            "Ignored when -P/--prompt/--user-prompt is set to a non-empty string. Default: medium."
        ),
    )
    parser.add_argument(
        "-P",
        "--prompt",
        "--user-prompt",
        dest="user_prompt",
        default=None,
        metavar="TEXT",
        help=(
            "Custom user message (replaces built-in preset; --prompt-size is then ignored). "
            "If omitted or empty, use the built-in preset for --prompt-size."
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
            f"Overrides MODEL_STRESS_GATEWAY_URL; if unset, default is {DEFAULT_GATEWAY_URL}."
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

def generate_prompt(
    is_random,
    prompt_repeats=1,
    prompt_size="medium",
    user_prompt=None,
):
    """Build user message: built-in Mars SF preset or custom text; optional cache-MISS variation; -p repeats."""
    custom = (user_prompt or "").strip()
    if custom:
        base = custom
    else:
        base = PROMPT_PRESETS[prompt_size]

    if is_random:
        sid = random.randint(100000, 999999)
        angle = random.choice(MARS_ANGLE_FOCUS)
        base = (
            f"{base}\n\n"
            f"Variation for this request: session {sid}. "
            f"Emphasize this angle: {angle}."
        )

    repeats = max(1, int(prompt_repeats))
    return " ".join([base] * repeats)

# ---------------------------------------------------------
# send_request uses the generator above
# ---------------------------------------------------------

def _format_debug_model_output(req_id, text):
    req_s = f"req={req_id:02d}"
    bar = "=" * 8
    line = f"{bar} {req_s} {bar}"
    return f"{line}\n{text}\n{line}\n"


def send_request(
    req_id,
    model_name,
    gateway_url,
    is_random,
    max_tokens,
    request_timeout,
    prompt_repeats,
    prompt_size="medium",
    user_prompt=None,
    brief=False,
    debug=False,
    debug_lock=None,
    debug_print_state=None,
):
    # Random startup delay (jitter) so NGINX can refresh connection counts
    time.sleep(random.uniform(0.1, 0.8))

    prompt = generate_prompt(
        is_random,
        prompt_repeats,
        prompt_size=prompt_size,
        user_prompt=user_prompt,
    )

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0
    }

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
                    if not brief:
                        print(ok_line, flush=True)
                    prefix = ""
                    if debug_print_state and debug_print_state["after_first"]:
                        prefix = "\n"
                    if debug_print_state is not None:
                        debug_print_state["after_first"] = True
                    print(prefix + block, end="", flush=True)
            elif brief:
                print("!", end="", flush=True)
            else:
                print(ok_line, flush=True)
            return {"status": 1, "tokens": tokens, "latency": latency}
        else:
            if brief:
                print(".", end="", flush=True)
            else:
                print(f"[FAIL] [req {req_id:02d}] HTTP {response.status_code}")
            return {"status": 0}
    except Exception as e:
        if brief:
            print(".", end="", flush=True)
        else:
            print(f"[WARN] [req {req_id:02d}] error: {e}")
        return {"status": 0}


def main():
    args = get_args()
    candidates = resolve_gateway_candidates(args.gateway_url)
    model_name, gateway_url = get_active_model(candidates)

    mode_info = "[RANDOM] cache MISS" if args.random else "[STATIC] cache HIT"
    if args.user_prompt and str(args.user_prompt).strip():
        prompt_info = "prompt=CUSTOM"
    else:
        prompt_info = f"prompt_size={args.prompt_size}"

    print("\n" + "=" * 60)
    print(
        f"[START] cluster stress | concurrency: {args.concurrency} | "
        f"prompt_repeats: {max(1, args.prompt_repeats)} | {prompt_info} | model: {model_name}"
    )
    print(f"[MODE] {mode_info}")
    print("=" * 60 + "\n")

    idle_limit = args.timeout
    last_ok_lock = threading.Lock()
    last_ok = {"t": time.monotonic()}

    def touch_success():
        with last_ok_lock:
            last_ok["t"] = time.monotonic()

    brief = args.brief or not args.verbose
    debug_lock = threading.Lock() if args.debug else None
    debug_print_state = {"after_first": False} if args.debug else None

    def run_one(req_id):
        r = send_request(
            req_id,
            model_name,
            gateway_url,
            args.random,
            args.tokens,
            args.timeout,
            args.prompt_repeats,
            prompt_size=args.prompt_size,
            user_prompt=args.user_prompt,
            brief=brief,
            debug=args.debug,
            debug_lock=debug_lock,
            debug_print_state=debug_print_state,
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

    if brief:
        print(flush=True)

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

# model_stress_tool

A small Python CLI that stress-tests an **OpenAI-compatible** HTTP gateway (for example vLLM behind a reverse proxy). It discovers the active model from `GET /v1/models`, then runs **concurrent** `POST /v1/chat/completions` requests and prints per-request metrics plus an aggregate summary.

## Requirements

- Python 3.x
- [`requests`](https://pypi.org/project/requests/)

Install dependencies (from this directory):

```bash
pip install requests
```

## Gateway URL

Resolution order (first wins):

1. **CLI:** `--gateway-url` / `-g`
2. **Environment:** `MODEL_STRESS_GATEWAY_URL`
3. **Built-in default:** `http://192.168.1.145` (see `DEFAULT_GATEWAY_URL` in [`model_stress_tool.py`](model_stress_tool.py))

Pass the **base URL only** (host or `host:port`); **omit scheme** to try **`http://` first, then `https://`** on the same host/port. If you pass `http://...` or `https://...`, only that URL is used. Do not include `/v1/...`. A trailing slash is stripped.

## Usage

```bash
python model_stress_tool.py [options]
```

### Options

| Short | Long | Default | Description |
|-------|------|---------|-------------|
| | `--help` | | Show help. |
| `-r` | `--random` | off | Randomize a session fragment in the prompt to encourage **cache MISS** on the prefix. |
| `-c` | `--concurrency` | `10` | Number of concurrent requests (one batch). |
| `-t` | `--tokens` | `512` | `max_tokens` for each completion. |
| `-T` | `--timeout` | `600` | **Sliding idle limit** (seconds): abort waiting if no request succeeds for this long; the clock **resets on every success**. Also passed as each HTTP request timeout. |
| `-d` | `--debug` | off | After each success, print the assistant reply in a **fixed delimiter block** (thread-safe with the `[OK]` line). |
| `-p` | `--prompt-repeats` | `1` | Repeat the one-line prompt `N` times, space-separated (length scaling). Values `< 1` are treated as `1`. |
| `-g` | `--gateway-url` | env / built-in | OpenAI-compatible gateway base URL. |

### Prompt

The default user message is a **single short English sentence** asking the model to reply with `OK`. With `--random`, a random session number is embedded so parallel runs are less likely to share an identical prefix.

### Timeout behavior

- **`--timeout` / `-T`** applies in two ways:
  1. **Orchestration:** If there is **no successful** HTTP 200 completion for `T` seconds, the tool stops waiting for the rest of the batch (partial results may still be summarized if any request succeeded).
  2. **HTTP:** Each `requests` call uses the same value as its request timeout.

### Random startup jitter

Each worker sleeps a random **0.1–0.8 s** before sending, to reduce synchronized thundering herds (e.g. against NGINX connection accounting).

### Connection header

Requests set `Connection: close` so connections are not aggressively reused (useful when testing connection-level behavior).

### Debug output format

When `--debug` / `-d` is set, each successful response is followed by:

```text
======== DEBUG MODEL OUTPUT req=XX ========
<assistant text>
======== END DEBUG MODEL OUTPUT req=XX ========
```

## Examples

Default run (10 concurrent requests, 512 max tokens, 600 s timeout):

```bash
python model_stress_tool.py
```

Higher concurrency and longer generations:

```bash
python model_stress_tool.py -c 32 -t 1024
```

Prefix-randomized prompts and stricter idle window:

```bash
python model_stress_tool.py -r -c 16 -T 300
```

Longer input (same sentence repeated 50 times) with model text on stdout:

```bash
python model_stress_tool.py -p 50 -d
```

## License

None specified in this repository; treat as internal / project-specific unless you add one.

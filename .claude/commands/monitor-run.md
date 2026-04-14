---
description: Poll any HTTP status endpoint on an interval and pretty-print JSON responses. Generic — not tied to any specific server.
---

# /monitor-run — HTTP Status Poller

Generic polling loop for watching any HTTP status/health endpoint. Useful during long-running training, rollouts, evals, migrations, or any background process that exposes a JSON `/status` (or similar) endpoint.

## Arguments

`$ARGUMENTS` — expected form:

```
--url <endpoint> [--interval <seconds>] [--fields <csv>] [--max-iters <n>] [--header 'K: V']
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--url` | *required* | The endpoint to GET |
| `--interval` | `10` | Seconds between polls |
| `--fields` | *all* | Comma-separated JSON field names to project (else pretty-print whole body) |
| `--max-iters` | *unbounded* | Stop after N polls |
| `--header` | *none* | Extra request header(s); repeatable |

Example invocations:

```
/monitor-run --url http://localhost:8006/status --interval 5
/monitor-run --url http://localhost:8006/status --fields init_q_depth,run_q_depth,eval_q_depth,active_workers
/monitor-run --url http://localhost:9090/-/healthy --interval 30 --max-iters 20
/monitor-run --url https://api.example.com/health --header 'Authorization: Bearer $TOKEN'
```

## Behavior

1. Validate `--url` was provided. Abort with a friendly error otherwise.
2. Loop: GET the URL, parse response. If JSON, pretty-print (or project requested fields); if not JSON, print body verbatim.
3. Between polls, sleep `--interval` seconds. Handle `Ctrl+C` cleanly.
4. Stop on any of: SIGINT, `--max-iters` reached, HTTP 404, or 5 consecutive connection failures.
5. Emit each poll with a timestamp prefix: `[2026-04-13T14:30:05Z] { "init_q_depth": 3, ... }`.
6. Include a one-line summary on exit: total polls, unique status codes observed, wall-clock duration.

## Implementation

Use Bash with `curl` + `jq`:

```bash
# Example shape of the inner loop:
while :; do
  body=$(curl --silent --show-error --max-time 5 "$URL" "${HEADER_ARGS[@]}")
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [ -n "$FIELDS" ]; then
    echo "$body" | jq --arg ts "$ts" --arg fields "$FIELDS" \
      '. as $d | ($fields | split(",")) | map({key: ., value: $d[.]}) | from_entries | . + {ts: $ts}'
  else
    echo "[$ts]"
    echo "$body" | jq . 2>/dev/null || echo "$body"
  fi
  ((iter++))
  [ -n "$MAX_ITERS" ] && [ "$iter" -ge "$MAX_ITERS" ] && break
  sleep "$INTERVAL"
done
```

Implement in plain bash/jq so nothing extra has to be installed.

## Out of scope

- This command does not start services. It only observes them.
- This command is not a load tester — it polls sequentially, not concurrently.
- For complex monitoring (histograms, time-series), integrate Prometheus/Grafana instead.

## Related

- `/harness-audit` — audit Claude Code harness config
- `/context-budget` — audit token usage

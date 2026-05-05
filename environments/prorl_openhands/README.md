# ProRL / OpenHands — EnvironmentProvider

Implements the `EnvironmentProvider` protocol for SWE-Bench tasks using OpenHands agents.

**Install** (on the host running this service):
```bash
cd environments/prorl_openhands && poetry install
```

**Start**:
```bash
bash environments/prorl_openhands/scripts/start.sh
```

Health gate: `GET http://localhost:8006/status` → `{"status":"running"}`

See root `CLAUDE.md` for the full startup sequence and boundary conditions.

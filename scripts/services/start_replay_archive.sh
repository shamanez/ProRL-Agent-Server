#!/usr/bin/env bash
# Start the ReplayArchive service (slot 5.5 — S3, optional).
# If disabled, the rollout worker runs without the tee.
# Start BEFORE start_rollout_worker.sh if enabled.
#
# Env vars:
#   REPLAY_ARCHIVE_ROOT       (default /home/ubuntu/replay_archive)
#   REPLAY_ARCHIVE_PORT       HTTP health port (default 8080)
set -euo pipefail

ARCHIVE_ROOT="${REPLAY_ARCHIVE_ROOT:-/home/ubuntu/replay_archive}"
PORT="${REPLAY_ARCHIVE_PORT:-8080}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${ARCHIVE_ROOT}"

echo "[replay_archive] archive root=${ARCHIVE_ROOT}"

# The archive is co-located with the rollout worker (in-process via
# ReplayArchiveWriter). This script exposes a tiny health HTTP endpoint
# so the orchestrator can probe it independently.
exec /home/ubuntu/.cache/pypoetry/virtualenvs/openhands-ai-342rfuwh-py3.12/bin/python -c "
import http.server, logging, os, sys, threading
logging.basicConfig(level=os.environ.get('LOG_LEVEL','INFO'),
    format='%(asctime)s %(levelname)s replay_archive: %(message)s')
sys.path.insert(0, '${REPO_ROOT}')
from replay_archive.server import ArchiveServer
server = ArchiveServer('${ARCHIVE_ROOT}')
print('[replay_archive] healthy archive_root=${ARCHIVE_ROOT}', flush=True)

class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ok')
    def log_message(self, *a): pass

httpd = http.server.HTTPServer(('', ${PORT}), HealthHandler)
httpd.serve_forever()
"

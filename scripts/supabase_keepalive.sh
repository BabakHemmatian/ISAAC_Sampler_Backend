#!/usr/bin/env bash
# Keep the Supabase free-tier project warm so it is not auto-paused after
# ~7 days of inactivity. Installed 2026-06-03. Runs daily via cron.
# Uses the public anon key only; writes no data. Emails on failure.
set -uo pipefail

PROJECT="https://rfyavjpuqoepfkxhtzie.supabase.co"
KEY="***REMOVED***"
LOG="/home/ubuntu/supabase_keepalive.log"
NOTIFY="/home/ubuntu/keepalive_notify.py"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 1) Auth service health -- hits the API gateway, returns 200.
health=$(curl -sS -m 20 -o /dev/null -w '%{http_code}' \
  -H "apikey: $KEY" "$PROJECT/auth/v1/health" 2>/dev/null || echo ERR)

# 2) Token grant with throwaway credentials -- forces a read against the
#    auth.users table (real DB activity). Always fails with 400; no write.
dbtouch=$(curl -sS -m 20 -o /dev/null -w '%{http_code}' -X POST \
  -H "apikey: $KEY" -H "Content-Type: application/json" \
  -d '{"email":"keepalive@isaac.invalid","password":"not-a-real-account"}' \
  "$PROJECT/auth/v1/token?grant_type=password" 2>/dev/null || echo ERR)

echo "$TS health=$health db_touch=$dbtouch" >> "$LOG"
tail -n 400 "$LOG" > "$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG"

# Failure = auth health not reachable/200 (project paused or VM lost egress).
if [ "$health" != "200" ]; then
  python3 "$NOTIFY" \
    "[ISAAC] Supabase keep-alive FAILED ($health)" \
    "The daily Supabase keep-alive could not reach the project.

Time (UTC): $TS
auth/health: $health
db_touch:    $dbtouch

This usually means the project was paused or the VM lost connectivity.
Resume it at: https://supabase.com/dashboard/project/rfyavjpuqoepfkxhtzie
Recent log:  ssh isaac_backend 'tail /home/ubuntu/supabase_keepalive.log'" \
    >> "$LOG" 2>&1 || echo "$TS notify_error" >> "$LOG"
fi

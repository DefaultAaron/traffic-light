#!/usr/bin/env bash
# Tailscale connectivity diagnosis — run on the offline-flapping host (e.g. `jun`).
# Distinguishes: (a) local daemon issue, (b) network/middlebox interference,
# (c) Tailscale server-side issue, (d) DNS, (e) firewall.
#
# Usage:
#   bash scripts/tailscale_diagnose.sh                # ~3-min snapshot
#   DURATION=600 bash scripts/tailscale_diagnose.sh   # longer soak (10 min)
#
# Output: timestamped report at /tmp/tailscale_diagnose.<host>.<ts>.log

set -u

DURATION=${DURATION:-180}              # seconds to soak-test long-poll
HOST=$(hostname -s)
TS=$(date +%Y%m%d-%H%M%S)
LOG=${LOG:-/tmp/tailscale_diagnose_${HOST}_${TS}.log}

# Capture EVERYTHING (stdout+stderr from this script and any child) to the log
exec > >(tee -a "$LOG") 2>&1

hr() { printf '\n========== %s ==========\n' "$*"; }
ok() { printf '  [ok]   %s\n' "$*"; }
bad() { printf '  [BAD]  %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }

echo "Tailscale diagnose @ $(date -Iseconds) on $HOST"
echo "Log: $LOG"
echo "Soak duration: ${DURATION}s"

hr "1. Local daemon"
if systemctl is-active --quiet tailscaled; then
    ok "tailscaled active"
else
    bad "tailscaled not active"
fi
systemctl status tailscaled --no-pager | head -8

hr "2. tailscale status"
tailscale status 2>&1 | head -20

hr "3. Layer-3 reachability (no DNS)"
for ip in 1.1.1.1 8.8.8.8 9.9.9.9; do
    if ping -c 2 -W 2 "$ip" >/dev/null 2>&1; then
        ok "ping $ip"
    else
        bad "ping $ip FAILED"
    fi
done

hr "4. DNS"
cat /etc/resolv.conf 2>/dev/null | grep -v '^#' | head -10
for host in controlplane.tailscale.com login.tailscale.com derp20.tailscale.com; do
    if getent hosts "$host" >/dev/null 2>&1; then
        ok "resolve $host -> $(getent hosts $host | awk '{print $1}' | head -1)"
    else
        bad "resolve $host FAILED"
    fi
done

hr "5. Short HTTPS to controlplane (does the server respond at all?)"
for i in 1 2 3; do
    rc=$(curl -s -o /dev/null --max-time 10 -w '%{http_code} time=%{time_total}s' \
         https://controlplane.tailscale.com/health)
    echo "  attempt $i: $rc"
done

hr "6. TCP/443 reachability (raw)"
for host in controlplane.tailscale.com login.tailscale.com; do
    if timeout 5 bash -c "</dev/tcp/$host/443" 2>/dev/null; then
        ok "tcp $host:443"
    else
        bad "tcp $host:443 FAILED"
    fi
done

hr "7. DERP relays (Hong Kong nearest for CN)"
for region in derp20 derp21; do
    rc=$(curl -s -o /dev/null --max-time 8 -w '%{http_code}' \
         "https://${region}.tailscale.com/derp/probe" 2>/dev/null || echo "fail")
    echo "  $region: $rc"
done

hr "8. Path discovery (where are the 502s coming from?)"
echo "MTR-style probe to controlplane.tailscale.com (TCP/443):"
if command -v mtr >/dev/null; then
    mtr -T -P 443 -c 5 -r controlplane.tailscale.com 2>/dev/null || \
        traceroute -T -p 443 -m 20 controlplane.tailscale.com 2>/dev/null | head -25
else
    traceroute -T -p 443 -m 20 controlplane.tailscale.com 2>/dev/null | head -25 || \
        warn "no traceroute/mtr installed"
fi

hr "9. MTU sanity (broken MSS clamping = stuck long-poll)"
ip -br link show | grep -E 'tailscale|wlo|eth|en' | head -5
echo "Probing path MTU to controlplane:"
for sz in 1472 1400 1280 1200; do
    if ping -c 1 -W 2 -M do -s "$sz" controlplane.tailscale.com >/dev/null 2>&1; then
        ok "MTU ${sz}+28 ok"
        break
    else
        warn "MTU ${sz}+28 fragments/drops"
    fi
done

hr "10. Long-poll soak test (THIS is what's failing in the journal)"
echo "Holding a long-running HTTPS connection to controlplane for ${DURATION}s..."
echo "If it dies before ${DURATION}s, that's the smoking gun for middlebox interference."
SOAK_START=$(date +%s)
timeout "$DURATION" curl -sN --max-time "$DURATION" \
    -o /dev/null \
    -w 'soak ended: code=%{http_code} time=%{time_total}s downloaded=%{size_download}B\n' \
    https://controlplane.tailscale.com/derpmap/default &
SOAK_PID=$!

hr "11. Live journal tail during soak (look for new 502/EOF)"
SINCE=$(date -Iseconds)
sudo journalctl -u tailscaled -f --since "$SINCE" --no-pager &
JOURNAL_PID=$!
wait $SOAK_PID
SOAK_RC=$?
SOAK_ELAPSED=$(( $(date +%s) - SOAK_START ))
kill $JOURNAL_PID 2>/dev/null
wait $JOURNAL_PID 2>/dev/null
echo "soak rc=$SOAK_RC elapsed=${SOAK_ELAPSED}s (target=${DURATION}s)"
if (( SOAK_ELAPSED < DURATION - 5 )); then
    bad "soak died early — middlebox is likely killing long-lived streams"
else
    ok "soak held full duration"
fi

hr "12. Recent tailscaled errors (last 10 min)"
sudo journalctl -u tailscaled --since "10 min ago" --no-pager | \
    grep -iE '502|EOF|gateway|context canceled|timed out|error' | tail -20

hr "13. Verdict heuristics"
err_count=$(sudo journalctl -u tailscaled --since "10 min ago" --no-pager | \
            grep -cE '502|unexpected EOF|context canceled' || echo 0)
if [[ $err_count -gt 5 ]]; then
    bad "${err_count} long-poll errors in last 10 min — control-plane connection unstable"
    echo "  Most likely: network middlebox killing long-lived HTTPS streams,"
    echo "  or transient Tailscale server-side issue (check status.tailscale.com)."
elif [[ $err_count -gt 0 ]]; then
    warn "${err_count} long-poll errors in last 10 min — intermittent"
else
    ok "no recent long-poll errors"
fi

echo
echo "============================================================"
echo "Done @ $(date -Iseconds)"
echo "Full log saved to: $LOG"
echo
echo "Bring this log back with:"
echo "  scp ${HOST}:${LOG} ./"
echo "  # or copy/paste contents — it's plain text"
echo "============================================================"

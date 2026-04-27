#!/usr/bin/env bash
# Pin Tailscale control-plane to IPv4 to work around a broken IPv6 path
# (symptom: tailscaled long-poll to /machine/map dies every ~60s with
# 502 Bad Gateway / unexpected EOF, while short HTTPS works fine).
#
# What it does:
#   1. Backs up /etc/hosts to /etc/hosts.bak.<ts>
#   2. Resolves IPv4 (A record) for controlplane + login + log endpoints
#   3. Adds /etc/hosts pins inside a clearly-marked managed block (idempotent)
#   4. Restarts tailscaled
#   5. Soaks the journal for ~90s and counts 502/EOF errors
#   6. Reports verdict
#
# Usage:
#   sudo bash scripts/tailscale_fix_ipv4.sh           # apply + verify
#   sudo bash scripts/tailscale_fix_ipv4.sh --revert  # remove the pins
#
# Run with sudo (writes /etc/hosts, restarts tailscaled, reads journal).

set -u

HOSTS_FILE=/etc/hosts
BLOCK_BEGIN="# >>> tailscale-ipv4-pin (managed by tailscale_fix_ipv4.sh) >>>"
BLOCK_END="# <<< tailscale-ipv4-pin <<<"
DOMAINS=(
    controlplane.tailscale.com
    login.tailscale.com
    log.tailscale.io
)
SOAK_SECS=${SOAK_SECS:-90}
TS=$(date +%Y%m%d-%H%M%S)
LOG=${LOG:-/tmp/tailscale_fix_ipv4_${TS}.log}

exec > >(tee -a "$LOG") 2>&1

hr()   { printf '\n========== %s ==========\n' "$*"; }
ok()   { printf '  [ok]   %s\n' "$*"; }
bad()  { printf '  [BAD]  %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
die()  { bad "$*"; exit 1; }

require_root() {
    [[ $EUID -eq 0 ]] || die "must run as root (sudo)"
}

remove_block() {
    if grep -qF "$BLOCK_BEGIN" "$HOSTS_FILE"; then
        sed -i.tmp "/$(printf '%s' "$BLOCK_BEGIN" | sed 's/[][\.*^$/]/\\&/g')/,/$(printf '%s' "$BLOCK_END" | sed 's/[][\.*^$/]/\\&/g')/d" "$HOSTS_FILE"
        rm -f "${HOSTS_FILE}.tmp"
        ok "removed managed block from $HOSTS_FILE"
    else
        ok "no managed block present"
    fi
}

resolve_ipv4() {
    local host=$1
    # Prefer dig; fall back to getent ahostsv4; then python.
    if command -v dig >/dev/null 2>&1; then
        dig +short +time=3 +tries=2 A "$host" | grep -E '^[0-9.]+$' | head -1
    elif command -v getent >/dev/null 2>&1; then
        getent ahostsv4 "$host" | awk '{print $1; exit}'
    else
        python3 -c "import socket; print(socket.getaddrinfo('$host', None, socket.AF_INET)[0][4][0])" 2>/dev/null
    fi
}

count_errors_since() {
    # Count tailscaled long-poll errors since the given timestamp.
    local since=$1
    journalctl -u tailscaled --since "$since" --no-pager 2>/dev/null | \
        grep -cE '502 Bad gateway|unexpected EOF|Post .*machine/map.*context canceled|long-poll timed out' || true
}

# ---------- main ----------

require_root

echo "Tailscale IPv4 pin @ $(date -Iseconds) on $(hostname -s)"
echo "Log: $LOG"

case "${1:-}" in
    --revert)
        hr "Reverting"
        remove_block
        echo
        echo "Restarting tailscaled..."
        systemctl restart tailscaled
        sleep 5
        ok "done. /etc/hosts now uses normal DNS again."
        exit 0
        ;;
esac

hr "1. Backup /etc/hosts"
BACKUP="${HOSTS_FILE}.bak.${TS}"
cp -a "$HOSTS_FILE" "$BACKUP"
ok "backed up to $BACKUP"

hr "2. Resolve IPv4 addresses"
declare -A IPV4
all_ok=1
for d in "${DOMAINS[@]}"; do
    ip=$(resolve_ipv4 "$d" || true)
    if [[ -n "$ip" && "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        ok "$d -> $ip"
        IPV4[$d]=$ip
    else
        bad "$d -> resolution FAILED (got: '$ip')"
        all_ok=0
    fi
done
if (( all_ok == 0 )); then
    warn "some hosts failed to resolve. Will pin only the ones that did."
fi
if (( ${#IPV4[@]} == 0 )); then
    die "no hosts resolved — cannot proceed. Check DNS first."
fi

hr "3. Apply /etc/hosts pin (idempotent)"
remove_block
{
    echo "$BLOCK_BEGIN"
    echo "# Added $(date -Iseconds) — forces IPv4 for Tailscale control plane."
    echo "# Reason: this network has a broken IPv6 path to *.tailscale.com,"
    echo "# causing /machine/map long-poll to die with 502/EOF every ~60s."
    echo "# Revert with: sudo bash $(realpath "$0") --revert"
    for d in "${DOMAINS[@]}"; do
        if [[ -n "${IPV4[$d]:-}" ]]; then
            printf '%-16s %s\n' "${IPV4[$d]}" "$d"
        fi
    done
    echo "$BLOCK_END"
} >> "$HOSTS_FILE"
ok "wrote pin block to $HOSTS_FILE"

hr "4. Verify resolution now returns IPv4"
for d in "${!IPV4[@]}"; do
    got=$(getent hosts "$d" | awk '{print $1; exit}')
    if [[ "$got" == "${IPV4[$d]}" ]]; then
        ok "$d resolves to ${IPV4[$d]} (IPv4)"
    else
        warn "$d resolved to '$got' (expected ${IPV4[$d]})"
    fi
done

hr "5. Restart tailscaled"
SINCE=$(date -Iseconds)
systemctl restart tailscaled
sleep 8
if systemctl is-active --quiet tailscaled; then
    ok "tailscaled restarted"
else
    die "tailscaled failed to start — check 'journalctl -u tailscaled'"
fi

hr "6. Soak for ${SOAK_SECS}s and watch journal"
echo "Sampling tailscaled journal — should NOT see 502/EOF if the fix worked."
echo
for ((i=10; i<=SOAK_SECS; i+=10)); do
    sleep 10
    n=$(count_errors_since "$SINCE")
    printf '  t=%3ds  long-poll errors so far: %d\n' "$i" "$n"
done

hr "7. Final journal sample (last ${SOAK_SECS}s)"
journalctl -u tailscaled --since "$SINCE" --no-pager | tail -40

hr "8. tailscale status"
tailscale status 2>&1 | head -10 || true

hr "9. Verdict"
final_errors=$(count_errors_since "$SINCE")
if (( final_errors == 0 )); then
    ok "ZERO long-poll errors during ${SOAK_SECS}s soak — fix appears to be working."
    echo "  Watch 'tailscale status' from peers; jun should come online within 1-2 min."
elif (( final_errors <= 2 )); then
    warn "${final_errors} long-poll errors during soak — better, but not clean."
    echo "  Wait another minute and re-check 'journalctl -u tailscaled --since \"2 min ago\"'."
else
    bad "${final_errors} long-poll errors during soak — fix did NOT help."
    echo "  Possible causes:"
    echo "    - The IPv4 path is also broken (test: curl -4 -v https://controlplane.tailscale.com/health)"
    echo "    - Tailscale server-side issue (check status.tailscale.com)"
    echo "    - Middlebox doing deep inspection on *.tailscale.com regardless of IP"
    echo "  Revert with: sudo bash $0 --revert"
fi

echo
echo "============================================================"
echo "Done @ $(date -Iseconds)"
echo "Log: $LOG"
echo "Backup of /etc/hosts: $BACKUP"
echo "Revert with: sudo bash $0 --revert"
echo "============================================================"

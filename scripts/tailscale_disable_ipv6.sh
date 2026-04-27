#!/usr/bin/env bash
# Disable IPv6 system-wide on this host, then restart tailscaled and verify
# that the Tailscale control-plane long-poll stops failing with 502 / EOF.
#
# Use this when the lighter `/etc/hosts` IPv4 pin didn't take effect because
# systemd-resolved / Go resolver kept finding AAAA records and using a broken
# IPv6 path to *.tailscale.com.
#
# Usage:
#   sudo bash scripts/tailscale_disable_ipv6.sh           # apply + verify
#   sudo bash scripts/tailscale_disable_ipv6.sh --revert  # re-enable IPv6
#
# Run with sudo.

set -u

SYSCTL_FILE=/etc/sysctl.d/99-disable-ipv6.conf
SOAK_SECS=${SOAK_SECS:-180}
TS=$(date +%Y%m%d-%H%M%S)
LOG=${LOG:-/tmp/tailscale_disable_ipv6_${TS}.log}

exec > >(tee -a "$LOG") 2>&1

hr()   { printf '\n========== %s ==========\n' "$*"; }
ok()   { printf '  [ok]   %s\n' "$*"; }
bad()  { printf '  [BAD]  %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
die()  { bad "$*"; exit 1; }

require_root() {
    [[ $EUID -eq 0 ]] || die "must run as root (sudo)"
}

# Catches both "502 Bad gateway" and "502: Bad gateway", plus the other
# long-poll failure modes seen in tailscaled's journal.
ERR_REGEX='502:? Bad gateway|unexpected EOF|Post .*machine/map.*context canceled|long-poll timed out|initial fetch failed'

count_errors_since() {
    local since=$1
    journalctl -u tailscaled --since "$since" --no-pager 2>/dev/null | \
        grep -cE "$ERR_REGEX" || true
}

require_root

echo "Tailscale IPv6-disable @ $(date -Iseconds) on $(hostname -s)"
echo "Log: $LOG"

case "${1:-}" in
    --revert)
        hr "Reverting — re-enabling IPv6"
        if [[ -f "$SYSCTL_FILE" ]]; then
            rm -f "$SYSCTL_FILE"
            ok "removed $SYSCTL_FILE"
        else
            ok "no $SYSCTL_FILE present"
        fi
        sysctl -w net.ipv6.conf.all.disable_ipv6=0     >/dev/null
        sysctl -w net.ipv6.conf.default.disable_ipv6=0 >/dev/null
        sysctl -w net.ipv6.conf.lo.disable_ipv6=0      >/dev/null
        sysctl --system >/dev/null
        ok "ipv6 re-enabled at runtime"

        echo "Restarting NetworkManager (so IPv6 addresses come back)..."
        if systemctl is-active --quiet NetworkManager; then
            systemctl restart NetworkManager
            sleep 5
        fi
        echo "Restarting tailscaled..."
        systemctl restart tailscaled
        sleep 5
        ok "done. Verify with: ip -6 addr show"
        exit 0
        ;;
esac

hr "1. Pre-flight: current IPv6 state"
echo "IPv6 addresses currently bound:"
ip -6 addr show 2>/dev/null | grep -E 'inet6' | grep -v 'scope host' || ok "no global IPv6 addresses"
echo
echo "Current resolution of controlplane.tailscale.com:"
getent ahosts controlplane.tailscale.com | head -5

hr "2. Pre-flight: count baseline long-poll errors (last 5 min)"
baseline=$(count_errors_since "5 min ago")
echo "  baseline errors in last 5 min: $baseline"

hr "3. Write sysctl drop-in to disable IPv6"
cat > "$SYSCTL_FILE" <<'EOF'
# Added by tailscale_disable_ipv6.sh.
# Reason: this network has a broken IPv6 path to *.tailscale.com that causes
# tailscaled's /machine/map long-poll to fail with 502/EOF every ~60s.
# Revert with: sudo bash scripts/tailscale_disable_ipv6.sh --revert
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1
EOF
ok "wrote $SYSCTL_FILE"
sysctl --system >/dev/null
ok "applied via sysctl --system"

hr "4. Verify IPv6 is now disabled"
v6_count=$(ip -6 addr show 2>/dev/null | grep -E 'inet6' | grep -vc 'scope host' || true)
if [[ $v6_count -eq 0 ]]; then
    ok "no global IPv6 addresses bound"
else
    warn "$v6_count IPv6 addresses still bound — may need NetworkManager restart"
    if systemctl is-active --quiet NetworkManager; then
        echo "  restarting NetworkManager..."
        systemctl restart NetworkManager
        sleep 5
        v6_count=$(ip -6 addr show 2>/dev/null | grep -E 'inet6' | grep -vc 'scope host' || true)
        echo "  after NM restart: $v6_count IPv6 addresses"
    fi
fi

hr "5. Verify resolver now returns IPv4"
echo "getent hosts controlplane.tailscale.com:"
getent hosts controlplane.tailscale.com | head -5
ipv4=$(getent hosts controlplane.tailscale.com | awk '{print $1; exit}')
if [[ "$ipv4" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    ok "resolves to IPv4: $ipv4"
else
    bad "still resolving to '$ipv4' (expected IPv4) — flushing systemd-resolved cache"
    if command -v resolvectl >/dev/null 2>&1; then
        resolvectl flush-caches
        sleep 1
        ipv4=$(getent hosts controlplane.tailscale.com | awk '{print $1; exit}')
        echo "  after flush: $ipv4"
    fi
fi

hr "6. Restart tailscaled"
SINCE=$(date -Iseconds)
systemctl restart tailscaled
sleep 8
if systemctl is-active --quiet tailscaled; then
    ok "tailscaled active"
else
    die "tailscaled failed to start"
fi

hr "7. Soak ${SOAK_SECS}s — count long-poll errors"
echo "Errors during soak (sampled every 15s):"
for ((i=15; i<=SOAK_SECS; i+=15)); do
    sleep 15
    n=$(count_errors_since "$SINCE")
    printf '  t=%4ds  errors so far: %d\n' "$i" "$n"
done

hr "8. Final journal sample (full window)"
journalctl -u tailscaled --since "$SINCE" --no-pager | tail -50

hr "9. tailscale status"
tailscale status 2>&1 | head -10 || true

hr "10. Connectivity probes"
echo "Direct curl to controlplane (should be quick, IPv4):"
for i in 1 2 3; do
    rc=$(curl -4 -s -o /dev/null --max-time 10 -w '%{http_code} time=%{time_total}s remote=%{remote_ip}' \
         https://controlplane.tailscale.com/health)
    echo "  attempt $i: $rc"
done

echo
echo "Sustain test (30s hold against /machine/map endpoint via curl-4):"
sustain_start=$(date +%s)
curl -4 -sN --max-time 30 -o /dev/null \
     -w 'sustain ended: code=%{http_code} time=%{time_total}s downloaded=%{size_download}B\n' \
     https://controlplane.tailscale.com/derpmap/default || true
sustain_elapsed=$(( $(date +%s) - sustain_start ))
echo "  sustain elapsed: ${sustain_elapsed}s"

hr "11. Verdict"
final=$(count_errors_since "$SINCE")
echo "Long-poll errors during ${SOAK_SECS}s soak: $final  (baseline before fix: $baseline)"
if (( final == 0 )); then
    ok "ZERO long-poll errors. Fix is working."
    echo "  Next: from your Mac, run:"
    echo "    sudo tailscale down && sudo tailscale up"
    echo "    tailscale ping jun"
    echo "    ssh user@jun"
elif (( final < baseline / 2 )); then
    warn "$final errors during soak — improved vs baseline ($baseline) but not clean."
    echo "  May still be partially usable. Try SSH from Mac and see."
else
    bad "$final errors during soak — IPv6 disable did NOT help."
    echo "  Means the IPv4 path is also broken (network-level interference, not IPv6)."
    echo "  Next: try mobile hotspot to isolate, or use autossh reverse tunnel."
    echo "  Revert with: sudo bash $0 --revert"
fi

echo
echo "============================================================"
echo "Done @ $(date -Iseconds)"
echo "Log: $LOG"
echo "Revert with: sudo bash $0 --revert"
echo "============================================================"

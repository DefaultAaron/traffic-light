#!/usr/bin/env bash
# Set up an autossh reverse tunnel from this host (e.g. `jun`) to a public VPS
# so you can SSH to it even when Tailscale's control plane is being throttled
# on this network.
#
# Topology after setup:
#   [your mac]  --ssh-->  [VPS public:2222]  --reverse tunnel-->  [jun:22]
#   i.e. on your mac:  ssh -p 2222 jun-user@vps.example.com
#
# Prerequisites:
#   1. A VPS with public IP, your public SSH key in its ~/.ssh/authorized_keys
#      for $VPS_USER, and sshd allowing GatewayPorts (see end of this script).
#   2. autossh installed on this host (script will apt-install if missing).
#
# Usage:
#   sudo VPS_HOST=vps.example.com VPS_USER=ubuntu VPS_PORT=22 \
#        REMOTE_PORT=2222 LOCAL_SSH_PORT=22 \
#        bash scripts/setup_reverse_tunnel.sh
#
#   sudo bash scripts/setup_reverse_tunnel.sh --status
#   sudo bash scripts/setup_reverse_tunnel.sh --disable
#
# Environment variables (all overridable):
#   VPS_HOST        public hostname / IP of the VPS                (REQUIRED)
#   VPS_USER        ssh user on VPS                                (REQUIRED)
#   VPS_PORT        ssh port on VPS                                (default: 22)
#   REMOTE_PORT     port on VPS that will forward to local sshd    (default: 2222)
#   LOCAL_SSH_PORT  this host's sshd port                          (default: 22)
#   TUNNEL_USER     local user that owns the tunnel + ssh key      (default: $SUDO_USER or current)
#   TUNNEL_NAME     systemd unit name suffix                       (default: vps)

set -u

VPS_HOST=${VPS_HOST:-}
VPS_USER=${VPS_USER:-}
VPS_PORT=${VPS_PORT:-22}
REMOTE_PORT=${REMOTE_PORT:-2222}
LOCAL_SSH_PORT=${LOCAL_SSH_PORT:-22}
TUNNEL_USER=${TUNNEL_USER:-${SUDO_USER:-$(id -un)}}
TUNNEL_NAME=${TUNNEL_NAME:-vps}

UNIT="autossh-tunnel-${TUNNEL_NAME}.service"
UNIT_PATH="/etc/systemd/system/$UNIT"
KEY_DIR="/home/${TUNNEL_USER}/.ssh"
KEY_PATH="${KEY_DIR}/id_ed25519_tunnel"

hr()   { printf '\n========== %s ==========\n' "$*"; }
ok()   { printf '  [ok]   %s\n' "$*"; }
bad()  { printf '  [BAD]  %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
die()  { bad "$*"; exit 1; }

require_root() {
    [[ $EUID -eq 0 ]] || die "must run as root (sudo)"
}

case "${1:-}" in
    --status)
        require_root
        systemctl status "$UNIT" --no-pager 2>&1 | head -20
        echo
        echo "Recent journal:"
        journalctl -u "$UNIT" -n 20 --no-pager
        exit 0
        ;;
    --disable)
        require_root
        systemctl stop    "$UNIT" 2>/dev/null || true
        systemctl disable "$UNIT" 2>/dev/null || true
        rm -f "$UNIT_PATH"
        systemctl daemon-reload
        ok "tunnel service stopped, disabled, and removed."
        exit 0
        ;;
esac

require_root

echo "Reverse-tunnel setup @ $(date -Iseconds) on $(hostname -s)"
echo "Tunnel user: $TUNNEL_USER"
echo "VPS:         ${VPS_USER}@${VPS_HOST}:${VPS_PORT}"
echo "Topology:    mac --ssh--> ${VPS_HOST}:${REMOTE_PORT} --tunnel--> $(hostname -s):${LOCAL_SSH_PORT}"
echo "Unit:        $UNIT"

[[ -z "$VPS_HOST" || -z "$VPS_USER" ]] && \
    die "VPS_HOST and VPS_USER are required. Re-run with:
  sudo VPS_HOST=... VPS_USER=... bash $0"

id "$TUNNEL_USER" >/dev/null 2>&1 || die "tunnel user '$TUNNEL_USER' does not exist"

hr "1. Install autossh"
if ! command -v autossh >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y autossh
fi
autossh -V 2>&1 | head -1
ok "autossh present"

hr "2. Generate dedicated SSH key for the tunnel (if missing)"
mkdir -p "$KEY_DIR"
chmod 700 "$KEY_DIR"
chown "$TUNNEL_USER":"$TUNNEL_USER" "$KEY_DIR"
if [[ -f "$KEY_PATH" ]]; then
    ok "key already exists at $KEY_PATH"
else
    sudo -u "$TUNNEL_USER" ssh-keygen -t ed25519 -N '' -f "$KEY_PATH" -C "autossh-tunnel-$(hostname -s)"
    ok "generated $KEY_PATH"
fi

hr "3. Public key — add this to ${VPS_USER}@${VPS_HOST}:~/.ssh/authorized_keys"
echo "------- copy from here -------"
cat "${KEY_PATH}.pub"
echo "------- copy to here ---------"
echo
echo "Run on the VPS (one-liner):"
echo "  mkdir -p ~/.ssh && chmod 700 ~/.ssh && \\"
echo "  echo '$(cat ${KEY_PATH}.pub)' >> ~/.ssh/authorized_keys && \\"
echo "  chmod 600 ~/.ssh/authorized_keys"
echo
read -r -p "Press Enter once the public key is on the VPS, or Ctrl-C to abort... " _

hr "4. Test SSH from this host to VPS (must be passwordless)"
if sudo -u "$TUNNEL_USER" ssh \
        -i "$KEY_PATH" -p "$VPS_PORT" \
        -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        -o ConnectTimeout=10 \
        "${VPS_USER}@${VPS_HOST}" 'echo VPS reachable: $(hostname)'; then
    ok "passwordless SSH to VPS works"
else
    die "cannot SSH to VPS — fix authorized_keys / network before continuing"
fi

hr "5. Write systemd unit"
cat > "$UNIT_PATH" <<EOF
[Unit]
Description=AutoSSH reverse tunnel to ${VPS_HOST} (exposes local sshd on :${REMOTE_PORT})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${TUNNEL_USER}
Environment=AUTOSSH_GATETIME=0
Environment=AUTOSSH_PORT=0
ExecStart=/usr/bin/autossh -M 0 -N \\
    -o ServerAliveInterval=30 \\
    -o ServerAliveCountMax=3 \\
    -o ExitOnForwardFailure=yes \\
    -o StrictHostKeyChecking=accept-new \\
    -o UserKnownHostsFile=${KEY_DIR}/known_hosts \\
    -i ${KEY_PATH} \\
    -p ${VPS_PORT} \\
    -R ${REMOTE_PORT}:localhost:${LOCAL_SSH_PORT} \\
    ${VPS_USER}@${VPS_HOST}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
ok "wrote $UNIT_PATH"

hr "6. Enable + start"
systemctl daemon-reload
systemctl enable "$UNIT"
systemctl restart "$UNIT"
sleep 5
if systemctl is-active --quiet "$UNIT"; then
    ok "$UNIT is active"
else
    bad "$UNIT failed to start"
    journalctl -u "$UNIT" -n 30 --no-pager
    exit 1
fi

hr "7. Verify tunnel from VPS side"
echo "Asking VPS whether port ${REMOTE_PORT} is now listening..."
remote_check=$(sudo -u "$TUNNEL_USER" ssh \
    -i "$KEY_PATH" -p "$VPS_PORT" \
    -o BatchMode=yes -o StrictHostKeyChecking=no \
    -o ConnectTimeout=10 \
    "${VPS_USER}@${VPS_HOST}" \
    "ss -tlnp 2>/dev/null | grep ':${REMOTE_PORT} ' || netstat -tlnp 2>/dev/null | grep ':${REMOTE_PORT} '" || true)
if [[ -n "$remote_check" ]]; then
    ok "VPS sees port ${REMOTE_PORT} listening:"
    echo "    $remote_check"
else
    warn "couldn't confirm via ss/netstat (may need sudo on VPS). Test manually:"
    echo "    ssh ${VPS_USER}@${VPS_HOST} 'ss -tln | grep ${REMOTE_PORT}'"
fi

hr "8. Mac-side instructions"
cat <<EOF

On your Mac, add this to ~/.ssh/config (replace 'jun-user' with the user you SSH in as):

    Host jun-via-vps
        HostName ${VPS_HOST}
        Port ${REMOTE_PORT}
        User <your-user-on-jun>
        ServerAliveInterval 30

Then connect with:
    ssh jun-via-vps

Important — VPS sshd config (one-time, on the VPS):
  The VPS sshd must allow remote port-forwarding to bind to public addresses
  if you want to skip the SSH-jump-through-VPS step. Easiest is to keep the
  forward bound to VPS localhost (default) and just SSH through the VPS:

    Host jun-via-vps
        HostName 127.0.0.1
        Port ${REMOTE_PORT}
        User <your-user-on-jun>
        ProxyJump ${VPS_USER}@${VPS_HOST}:${VPS_PORT}

  This is more secure (no public exposure of jun's SSH) and works without
  GatewayPorts on the VPS.

EOF

hr "9. Useful commands"
cat <<EOF
  Status:   sudo systemctl status $UNIT
  Logs:     sudo journalctl -u $UNIT -f
  Restart:  sudo systemctl restart $UNIT
  Disable:  sudo bash $0 --disable
EOF

echo
echo "Done @ $(date -Iseconds)"

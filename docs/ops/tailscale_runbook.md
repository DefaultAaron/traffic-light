# Tailscale on campus network — quick runbook

**Status:** open / workaround in production
**Affected host:** GPU server `jun` on Tsinghua / CERNET LAN
**Symptom:** SSH from Mac to `jun` over Tailscale times out during business hours; `tailscale status` flaps between `offline` and DERP-relayed `active`.

---

## 30-second triage

Run on `jun`:

```bash
sudo journalctl -u tailscaled --since "10 min ago" --no-pager \
    | grep -cE '502:? Bad gateway|unexpected EOF|long-poll timed out'
```

- **Non-zero count** → campus DPI / connection-table eviction is killing the long-poll right now. Use the autossh tunnel (below). Tailscale will recover overnight / early morning when load drops.
- **Zero** → Tailscale should be working. If SSH still fails, problem is elsewhere; run the full diagnostic.

## Workaround: autossh reverse tunnel through a public VPS

```
[Mac] --ssh--> [VPS public IP:2222] --reverse-tunnel--> [jun:22]
```

```bash
# On jun (one-time):
sudo VPS_HOST=<vps-host> VPS_USER=<vps-user> bash scripts/setup_reverse_tunnel.sh
sudo bash scripts/setup_reverse_tunnel.sh --status
sudo bash scripts/setup_reverse_tunnel.sh --disable
```

Mac `~/.ssh/config` (keeps `jun`'s sshd off the public internet, jumps through VPS):

```ssh-config
Host jun
    HostName 127.0.0.1
    Port 2222
    User <jun-user>
    ProxyJump <vps-user>@<vps-host>
```

VPS requirement: any cheap server with public IP and outbound port 22 reachable from campus (Aliyun / Tencent Cloud lightservers ~¥30/月 work well from CERNET). VPS sshd needs `GatewayPorts yes`.

## Scripts in `scripts/`

| Script | Purpose |
|---|---|
| `setup_reverse_tunnel.sh` | Install systemd-managed autossh reverse tunnel — the actual workaround. Writes a timestamped log to `/tmp/<name>_<ts>.log`. |

Earlier diagnostic / fix-IPv4 / disable-IPv6 scripts have been removed:
the diagnostic was a one-off snapshot, and the IPv4 pin / IPv6 disable
both verifiably failed against the campus-DPI case (root cause is
network-level interference with long-lived HTTPS streams, not IPv6 path
quality). The `setup_reverse_tunnel.sh` workaround is sufficient.

## Root cause (high confidence, won't fix from inside Tailscale)

Network-level interference with long-lived HTTPS streams to `*.tailscale.com`. Most likely **time-of-day campus DPI / connection-table eviction**. Short HTTPS to `controlplane.tailscale.com` succeeds; the 60-second long-poll on `/machine/map` dies every cycle with 502 / EOF / timeout. Affects both IPv4 and IPv6 paths (verified by disabling IPv6). Restarting tailscaled, re-authenticating, or pinning hosts does not change the cadence — the middlebox kills the connection regardless of the client side.

## Future hardening (not yet tried)

- **Self-hosted Headscale** on a personal VPS — same wire protocol, but DPI signatures keyed on `*.tailscale.com` won't match. Highest effort, most durable.
- **`tailscale up --netfilter-mode=off`** + DERP-only — quieter on the wire (no UDP STUN probes); occasionally helps with DPI.
- **MTU tuning** on `tailscale0` — unlikely to help here since short HTTPS works fine.

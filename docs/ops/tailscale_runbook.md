# Tailscale instability on Tsinghua campus network (CERNET)

**Status:** open / workaround available
**First observed:** 2026-04-27 (worked in the morning, broke around midday)
**Affected host:** GPU server `jun` (Lenovo Legion, Ubuntu) on campus LAN
**Symptom:** SSH from Mac (`mbp`) to `jun` over Tailscale times out; `tailscale status` shows `jun` flapping between `offline` and `active; relay "sin"/"hkg"`.

---

## Symptom signature

`tailscaled` journal on `jun` shows a repeating pattern every ~60–120 s:

```
control: lite map update error after 1m0.2s: initial fetch failed 502: Bad gateway
control: map response long-poll timed out!
control: lite map update error after 2m0.001s: Post "https://controlplane.tailscale.com/machine/map": context canceled
Received error: PollNetMap: unexpected EOF
```

Critically:
- **Short HTTPS to `controlplane.tailscale.com` succeeds** in <1 s (curl returns valid responses)
- **The 60-second long-poll on `/machine/map` dies** every cycle with 502 / EOF / timeout
- Affects both **IPv4 and IPv6** paths (verified by disabling IPv6)

## Root cause (high confidence)

Network-level interference with long-lived HTTPS streams to `*.tailscale.com`. Most likely **time-of-day campus DPI / connection-table eviction** on the Tsinghua / CERNET network. Evidence:

1. Public IPv4 `166.111.159.130` falls in `166.111.0.0/16` — **Tsinghua University** allocation.
2. IPv6 traceroute showed terrible CERNET2 transit: HK → Marseille → Paris → Frankfurt to reach Tailscale's IPv6 anycast (`lb.fra.tailscale.com`), with several 100% loss hops and 3000–7000 ms jitter spikes.
3. Same problem persists after disabling IPv6 — proves it's not IPv6-specific.
4. Short HTTPS works, long HTTPS dies — classic stateful-firewall idle-eviction or DPI long-flow termination signature.
5. Worked in the morning, broken by midday — consistent with time-of-day policy escalation.

This is **not** fixable from inside Tailscale: not by reinstalling, not by re-authenticating, not by tuning client flags. The middlebox in the path between `jun` and Tailscale's control plane is killing the connection.

## What was tried (all failed to fix; documented to avoid re-trying)

| Attempt | Result |
|---|---|
| `sudo tailscale up --reset` | No effect on 502 cadence |
| `sudo systemctl restart tailscaled` | Reconnects briefly, then 502s resume |
| `sudo tailscale logout && tailscale up` | Same |
| Pin `controlplane.tailscale.com` → IPv4 in `/etc/hosts` | systemd-resolved kept returning AAAA from cache; resolver pin alone insufficient |
| `resolvectl flush-caches` after pin | Pin took effect but 502s still occurred on IPv4 |
| Disable IPv6 system-wide via `/etc/sysctl.d/99-disable-ipv6.conf` | Forces IPv4 (verified: `192.200.0.105`); 502s still occurred on IPv4 path |
| Reinstall (not attempted) | Diagnosis ruled this out — not a local-state issue |

## Workaround: autossh reverse tunnel through a public VPS

When campus DPI is active, route SSH around Tailscale entirely.

```
[Mac] --ssh--> [VPS public IP:2222] --reverse-tunnel--> [jun:22]
```

Setup script (idempotent, reversible):
- `scripts/setup_reverse_tunnel.sh`
- Usage: `sudo VPS_HOST=... VPS_USER=... bash scripts/setup_reverse_tunnel.sh`
- Disable: `sudo bash scripts/setup_reverse_tunnel.sh --disable`
- Status: `sudo bash scripts/setup_reverse_tunnel.sh --status`

Mac `~/.ssh/config` (recommended — keeps `jun`'s sshd off the public internet, jumps through VPS):

```ssh-config
Host jun
    HostName 127.0.0.1
    Port 2222
    User <jun-user>
    ProxyJump <vps-user>@<vps-host>
```

VPS requirement: any cheap server with public IP and outbound port 22 reachable from campus. Aliyun / Tencent Cloud lightservers (~¥30/月) are convenient because they're well-connected from CERNET.

## Diagnostic / fix scripts (in `scripts/`)

All write a timestamped log to `/tmp/<name>_<ts>.log`. Logs already collected for the 2026-04-27 incident sit in `logs/`.

| Script | Purpose |
|---|---|
| `tailscale_diagnose.sh` | Full diagnostic snapshot: daemon, DNS, L3/L4 reachability, MTU probe, traceroute, long-poll soak. Run first when symptom recurs. |
| `tailscale_fix_ipv4.sh` | Pin Tailscale endpoints to IPv4 in `/etc/hosts`. **Did not solve this incident** but kept for cases where IPv6 is the only problem. `--revert` to undo. |
| `tailscale_disable_ipv6.sh` | Disable IPv6 system-wide via sysctl. **Did not solve this incident.** `--revert` to undo. |
| `setup_reverse_tunnel.sh` | Install systemd-managed autossh reverse tunnel to a VPS — the actual workaround. |

## Reproduction / monitoring

Quick recheck (run on `jun`):

```bash
sudo journalctl -u tailscaled --since "10 min ago" --no-pager \
    | grep -cE '502:? Bad gateway|unexpected EOF|long-poll timed out'
```

A non-zero count over 10 minutes means the throttle is active *now*. Zero means it's a good window — Tailscale will work normally.

Empirically: Tailscale tends to recover overnight / early morning when campus traffic and DPI load drop, then break again during business hours. If you need predictable access, treat the reverse tunnel as primary and Tailscale as a bonus when it happens to work.

## Future hardening options (not yet tried)

- **Self-hosted Headscale** on a personal VPS. Same Tailscale wire protocol, but the control plane lives at *your* domain — DPI signatures keyed on `*.tailscale.com` won't match. Highest effort, most durable fix.
- **`tailscale up --netfilter-mode=off`** + DERP-only — quieter on the wire (no UDP STUN probes), occasionally helps with DPI. Untested in this network.
- **MTU tuning** on `tailscale0` (`ip link set tailscale0 mtu 1280`) — unlikely to help here since short HTTPS works fine, but cheap.

## Reference: timeline of the 2026-04-27 incident

- ~morning: Tailscale working normally; SSH from Mac to `jun` fine.
- ~11:30: SSH starts timing out; `tailscale status` on `jun` shows `offline`.
- 11:34–12:00: Multiple restart / `--reset` / `logout && up` cycles. Each connects briefly, then 502s resume after ~60 s.
- 12:03: `tailscale_diagnose.sh` run — confirmed short HTTPS OK, long-poll dies, 12 long-poll errors in 10 min.
- 12:13: `/etc/hosts` IPv4 pin applied — pin not honored by systemd-resolved without cache flush; 502s continue.
- 12:24: System-wide IPv6 disable applied — IPv6 confirmed off, all traffic IPv4; 502s **still** occur on IPv4. Definitive proof the issue is not IPv6-specific.
- ~12:30+: Status briefly shows `active; relay "sin"` (Singapore DERP relay) — data-plane path established despite control-plane flapping. SSH still timed out at the time of writing.
- Plan: leave Tailscale installed, set up autossh reverse tunnel as durable backup, retry Tailscale during off-hours.

"""Best-effort LAN address discovery for the Pilot 2 Open Platforms setup
page. These are candidates to present to the user for manual confirmation,
not an authoritative answer — a Docker container's view of its own network
interfaces is not reliable for "what address should the RC point at."
"""

from __future__ import annotations

import socket


def candidate_lan_addresses() -> list[str]:
    candidates: set[str] = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            addr = info[4][0]
            if not addr.startswith("127."):
                candidates.add(addr)
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            candidates.add(probe.getsockname()[0])
    except OSError:
        pass
    return sorted(candidates)

from __future__ import annotations

"""Print available spawn points for the current CARLA map."""

import argparse
import carla


def main() -> int:
    """Connect to CARLA and list spawn points."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    m = world.get_map()

    sps = m.get_spawn_points()
    print(f"Map: {m.name}")
    print(f"Spawn points: {len(sps)}")
    for i, tr in enumerate(sps):
        l = tr.location
        r = tr.rotation
        print(f"[{i:03d}] loc=({l.x:.2f},{l.y:.2f},{l.z:.2f}) rot=(pitch={r.pitch:.1f},yaw={r.yaw:.1f},roll={r.roll:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

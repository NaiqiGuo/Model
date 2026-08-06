"""
Create the committed single-event bridge fixture used by the bridge field
recompute test, so that test can run in CI without the full (gitignored,
~73 MB) events.pkl.

Run this once in an environment that has quakeio + events.pkl:

    python tests/make_bridge_fixture.py

Produces tests/fixtures/bridge_event_1.pkl.gz -- event index 0, which is
event_id "1" (events are sorted by peak_accel). The pickle holds a quakeio
event object, so unpickling it (here and in CI) requires quakeio installed.
Regenerate only if the bridge golden data legitimately changes.
"""
import gzip
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVENTS_PKL = ROOT / "events.pkl"
OUT = ROOT / "tests" / "fixtures" / "bridge_event_1.pkl.gz"


def main():
    with open(EVENTS_PKL, "rb") as f:
        events = pickle.load(f)
    event = events[0]  # sorted by peak_accel -> event_id "1"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wb") as f:
        pickle.dump(event, f)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

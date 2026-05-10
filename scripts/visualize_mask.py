"""Pull one frame from Redis and overlay the motion-gating mask region(s)
on it for visual inspection.

Usage:
    uv run scripts/visualize_mask.py [room] [x0,y0,x1,y1 ...]
    # examples:
    uv run scripts/visualize_mask.py bedroom 0.75,0,1.0,0.10
    uv run scripts/visualize_mask.py bedroom                # uses default mask
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from ai_baby_monitor.stream import RedisStreamHandler

load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

DEFAULT_MASKS = [(0.75, 0.0, 1.0, 0.10)]


def parse_region(spec: str):
    parts = [float(p) for p in spec.split(",")]
    if len(parts) != 4:
        raise SystemExit(f"Expected 4 floats, got: {spec}")
    return tuple(parts)


def main():
    room = "bedroom"
    regions = []
    for arg in sys.argv[1:]:
        if "," in arg:
            regions.append(parse_region(arg))
        else:
            room = arg
    if not regions:
        regions = DEFAULT_MASKS

    h = RedisStreamHandler(redis_host=REDIS_HOST, redis_port=REDIS_PORT)
    frames = h.get_latest_frames(f"{room}:subsampled", 1)
    if not frames:
        raise SystemExit(f"No frames in {room}:subsampled")

    buf = frames[0].frame_data
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.ndim == 1 else buf
    H, W = img.shape[:2]
    print(f"Frame size: {W} x {H}")

    overlay = img.copy()

    for i, (x0, y0, x1, y1) in enumerate(regions):
        xa, xb = int(round(x0 * W)), int(round(x1 * W))
        ya, yb = int(round(y0 * H)), int(round(y1 * H))
        # Translucent red fill.
        cv2.rectangle(overlay, (xa, ya), (xb, yb), (0, 0, 255), thickness=-1)
        # Solid border + label.
        cv2.rectangle(img, (xa, ya), (xb, yb), (0, 0, 255), 2)
        label = f"#{i+1} ({x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f})"
        cv2.putText(img, label, (xa + 4, max(ya - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        print(f"Region #{i+1}: pixels x[{xa}..{xb}] y[{ya}..{yb}] "
              f"size {xb-xa}x{yb-ya}")

    blended = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    out_dir = Path("/tmp/motion_inspect")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "mask_overlay.png"
    cv2.imwrite(str(out), blended)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

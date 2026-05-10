"""One-shot diagnostic: pull two frames from Redis ~1s apart, decode them,
diff them, and write three PNGs plus a bounding box around the changing area.

Run from project root:
    REDIS_HOST=localhost uv run scripts/inspect_motion.py bedroom
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from ai_baby_monitor.stream import RedisStreamHandler

load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


def decode(buf: np.ndarray) -> np.ndarray:
    if buf.ndim == 1:
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return buf


def main():
    room = sys.argv[1] if len(sys.argv) > 1 else "bedroom"
    out_dir = Path("/tmp/motion_inspect")
    out_dir.mkdir(parents=True, exist_ok=True)

    h = RedisStreamHandler(redis_host=REDIS_HOST, redis_port=REDIS_PORT)
    key = f"{room}:subsampled"

    f1 = h.get_latest_frames(key, 1)
    if not f1:
        print(f"No frames in {key}", file=sys.stderr)
        sys.exit(1)
    img1 = decode(f1[0].frame_data)
    print(f"frame1 shape={img1.shape} dtype={img1.dtype}")

    # Wait a bit longer than 1s to make sure timestamp digits flip.
    time.sleep(1.5)

    f2 = h.get_latest_frames(key, 1)
    img2 = decode(f2[0].frame_data)
    print(f"frame2 shape={img2.shape} dtype={img2.dtype}")

    # Convert to grayscale, diff, threshold to find changing pixels.
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Locate changed regions.
    ys, xs = np.where(mask > 0)
    h_img, w_img = mask.shape
    if xs.size == 0:
        print("No pixel diff above threshold (try lowering 30 -> 15).")
    else:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        print(f"Image size : {w_img} x {h_img}")
        print(f"Diff bbox  : x[{x0}..{x1}] y[{y0}..{y1}]")
        print(f"Diff bbox normalized: "
              f"({x0/w_img:.3f}, {y0/h_img:.3f}, {x1/w_img:.3f}, {y1/h_img:.3f})")
        print(f"Suggested --motion-ignore "
              f"{max(0, x0/w_img - 0.01):.2f},"
              f"{max(0, y0/h_img - 0.01):.2f},"
              f"{min(1, x1/w_img + 0.01):.2f},"
              f"{min(1, y1/h_img + 0.01):.2f}")

        # Draw rectangle on a copy of frame2 for visualization.
        annotated = img2.copy()
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / "annotated.png"), annotated)
        print(f"Annotated frame -> {out_dir / 'annotated.png'}")

    cv2.imwrite(str(out_dir / "frame1.png"), img1)
    cv2.imwrite(str(out_dir / "frame2.png"), img2)
    cv2.imwrite(str(out_dir / "diff.png"), diff)
    cv2.imwrite(str(out_dir / "mask.png"), mask)
    print(f"Wrote frame1.png, frame2.png, diff.png, mask.png to {out_dir}")


if __name__ == "__main__":
    main()

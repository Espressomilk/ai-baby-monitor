import argparse
import contextlib
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import structlog
from dotenv import load_dotenv
from playsound3 import playsound


@contextlib.contextmanager
def _suppress_stderr():
    """Silence native (C-level) stderr -- needed to mute ALSA's own prints."""
    fd = sys.stderr.fileno()
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


def _is_wsl() -> bool:
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _wsl_to_windows_path(linux_path: str) -> str | None:
    """Convert a WSL POSIX path to a Windows UNC path Windows tools can read."""
    p = Path(linux_path).resolve()
    # Try wslpath if available (handles /mnt/c/... too).
    if shutil.which("wslpath"):
        try:
            out = subprocess.check_output(
                ["wslpath", "-w", str(p)], text=True, timeout=2
            ).strip()
            return out or None
        except Exception:
            pass
    # Manual fallback for files inside the WSL filesystem.
    distro = os.environ.get("WSL_DISTRO_NAME", "")
    if distro:
        return rf"\\wsl.localhost\{distro}{str(p).replace('/', chr(92))}"
    return None


def play_alert(sound_path: str) -> None:
    """Play an alert sound. On WSL we go through Windows PowerShell because
    the WSL Linux side has no real audio device by default. Falls back to
    playsound3 elsewhere.
    """
    if _is_wsl():
        win_path = _wsl_to_windows_path(sound_path)
        if not win_path:
            raise RuntimeError(f"Could not resolve Windows path for {sound_path}")
        ps = shutil.which("powershell.exe") or "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        cmd = [
            ps, "-NoProfile", "-Command",
            f'$p = New-Object Media.SoundPlayer "{win_path}"; $p.PlaySync() | Out-Null',
        ]
        subprocess.run(
            cmd, check=True, timeout=10,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    # Non-WSL Linux / macOS / Windows-native: use playsound3.
    with _suppress_stderr():
        playsound(sound_path)

from ai_baby_monitor.config import load_room_config_file
from ai_baby_monitor.stream import RedisStreamHandler
from ai_baby_monitor.watcher import Watcher

logger = structlog.get_logger()

load_dotenv()
REDIS_HOST = "localhost"
REDIS_PORT = os.getenv("REDIS_PORT")
VLLM_HOST = "localhost"
VLLM_PORT = os.getenv("VLLM_PORT")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")


class MotionGate:
    """Cheap CPU-side motion detector to skip LLM calls when nothing is moving.

    Compares mean absolute pixel diff between consecutive downsampled grayscale
    frames. If any recent frame is above the threshold, we run the LLM. After
    motion stops we keep running for `cooldown_s` seconds so we don't miss the
    tail of an event (someone walking out, finishing covering a face, etc.).

    `ignore_regions` is a list of normalized (x0, y0, x1, y1) rectangles in
    [0, 1] coordinates that get zeroed out before the diff. Use this to mask
    out a camera-overlay timestamp or any other always-changing UI chrome.
    """

    def __init__(
        self,
        threshold: float = 2.0,
        cooldown_s: float = 5.0,
        ignore_regions: list[tuple[float, float, float, float]] | None = None,
    ):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.ignore_regions = ignore_regions or []
        self._prev_gray: np.ndarray | None = None
        self._last_motion_ts: float = 0.0
        self._mask: np.ndarray | None = None  # built lazily once we know the size

    def _apply_mask(self, gray: np.ndarray) -> np.ndarray:
        if not self.ignore_regions:
            return gray
        if self._mask is None or self._mask.shape != gray.shape:
            h, w = gray.shape
            mask = np.ones((h, w), dtype=np.uint8)
            for x0, y0, x1, y1 in self.ignore_regions:
                xa, xb = int(round(x0 * w)), int(round(x1 * w))
                ya, yb = int(round(y0 * h)), int(round(y1 * h))
                xa, xb = max(0, min(xa, w)), max(0, min(xb, w))
                ya, yb = max(0, min(ya, h)), max(0, min(yb, h))
                mask[ya:yb, xa:xb] = 0
            self._mask = mask
        return gray * self._mask

    @staticmethod
    def _downsample(frame: np.ndarray) -> np.ndarray:
        # Frames may arrive as a JPEG-encoded byte buffer (1D uint8) or as a
        # raw pixel array (2D grayscale, 3D BGR, 4D BGRA).
        try:
            if frame.ndim == 1:
                # JPEG bytes -- decode straight to grayscale to skip a colour pass.
                gray = cv2.imdecode(frame, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise ValueError("cv2.imdecode returned None on 1D buffer")
            elif frame.ndim == 2:
                gray = frame
            elif frame.ndim == 3 and frame.shape[2] == 4:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            elif frame.ndim == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif frame.ndim == 3:
                gray = frame[..., 0]
            else:
                raise ValueError(
                    f"Unexpected frame shape {frame.shape}, dtype {frame.dtype}"
                )
            return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        except Exception:
            logger.error(
                "MotionGate downsample failed",
                shape=getattr(frame, "shape", None),
                dtype=str(getattr(frame, "dtype", None)),
                ndim=getattr(frame, "ndim", None),
                type=type(frame).__name__,
            )
            raise

    def update(self, frames) -> tuple[bool, float]:
        """Returns (should_run_llm, max_diff_seen_in_batch)."""
        max_diff = 0.0
        for f in frames:
            g = self._apply_mask(self._downsample(f.frame_data))
            if self._prev_gray is not None:
                diff = float(cv2.absdiff(g, self._prev_gray).mean())
                if diff > max_diff:
                    max_diff = diff
            self._prev_gray = g

        if self._prev_gray is None:
            # First frame ever -- run once to establish baseline state.
            self._last_motion_ts = time.time()
            return True, max_diff

        if max_diff > self.threshold:
            self._last_motion_ts = time.time()
            return True, max_diff

        # No motion this batch, but stay active during cooldown after last motion.
        if (time.time() - self._last_motion_ts) <= self.cooldown_s:
            return True, max_diff

        return False, max_diff


class AudioGate:
    """Audio loudness monitor that consumes per-window RMS values produced
    by the streamer (stream_to_redis.py with --audio) from a Redis stream.

    Sharing one ffmpeg/RTSP connection through the streamer (rather than
    opening a second one from the watcher) avoids starving the camera's
    client slots, which on the Synology shared stream caused both audio
    and video to break.

    `threshold_rms` is the int16-RMS value above which a window counts as
    loud. Tune by watching the values in the watcher logs at rest vs
    during a real cry.
    """

    def __init__(
        self,
        redis_handler,
        room_key: str,
        threshold_rms: float = 800.0,
        cooldown_s: float = 8.0,
        sustained_windows: int = 2,
        max_age_s: float = 5.0,
    ):
        self.redis_handler = redis_handler
        self.audio_key = f"{room_key}:audio_rms"
        self.threshold_rms = threshold_rms
        self.cooldown_s = cooldown_s
        # Require N consecutive loud windows to trigger -- avoids one-off
        # claps/door-slams from looking like a sustained cry.
        self.sustained_windows = max(1, sustained_windows)
        # If the most recent audio sample in Redis is older than this,
        # treat audio as unavailable (streamer might be stuck reconnecting).
        self.max_age_s = max_age_s

        self._last_event_ts: float = 0.0
        self._last_rms: float = 0.0
        self._consecutive_loud: int = 0
        self._last_seen_id: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._fresh_loud_event = False  # latched, consumed by main loop

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        last_periodic_log = time.time()
        peak_since_log = 0.0
        windows_seen = 0
        first_logged = False
        no_data_warned_at: float | None = None
        while not self._stop.is_set():
            try:
                # Pull any new entries since last poll. xrevrange is fine;
                # we sort + iterate in chronological order via [::-1].
                entries = self.redis_handler.get_latest_entries(
                    self.audio_key,
                    count=20,
                    last_id=self._last_seen_id,
                )
            except Exception as e:
                logger.warning("AudioGate redis read error", error=str(e))
                self._stop.wait(1.0)
                continue

            if not entries:
                # Nothing new -- check if the stream has gone stale.
                if no_data_warned_at is None or (time.time() - no_data_warned_at) > 30:
                    if windows_seen == 0:
                        logger.info(
                            "AudioGate waiting for data",
                            audio_key=self.audio_key,
                        )
                    no_data_warned_at = time.time()
                self._stop.wait(0.25)
                continue

            no_data_warned_at = None
            for entry_id, fields in entries:
                eid = (
                    entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                )
                # Skip the entry that matches our last-seen id (xrevrange's
                # min bound is inclusive, so we'd see it again every poll).
                if eid == self._last_seen_id:
                    continue
                self._last_seen_id = eid
                rms_raw = fields.get(b"rms") or fields.get("rms")
                ts_raw = fields.get(b"timestamp") or fields.get("timestamp")
                if rms_raw is None:
                    continue
                try:
                    rms = float(rms_raw)
                except Exception:
                    continue
                # Drop ancient entries -- if streamer was paused, don't
                # let stale loudness from minutes ago re-fire an alert.
                if ts_raw is not None:
                    try:
                        ts = float(ts_raw)
                        if (time.time() - ts) > self.max_age_s:
                            continue
                    except Exception:
                        pass
                self._last_rms = rms
                peak_since_log = max(peak_since_log, rms)
                windows_seen += 1
                if not first_logged:
                    logger.info("AudioGate first window", rms=round(rms, 1))
                    first_logged = True
                if rms >= self.threshold_rms:
                    self._consecutive_loud += 1
                    if self._consecutive_loud >= self.sustained_windows:
                        self._last_event_ts = time.time()
                        self._fresh_loud_event = True
                    elif self._consecutive_loud == 1:
                        # Note transient spikes so users can see them go by
                        # without flipping the alert.
                        logger.info(
                            "🔊 loud spike (not yet sustained)",
                            rms=round(rms, 1),
                            need=self.sustained_windows,
                        )
                else:
                    self._consecutive_loud = 0

            now = time.time()
            if now - last_periodic_log >= 5.0:
                logger.info(
                    "AudioGate heartbeat",
                    windows=windows_seen,
                    peak_rms_5s=round(peak_since_log, 1),
                    current_rms=round(self._last_rms, 1),
                )
                last_periodic_log = now
                peak_since_log = 0.0

            self._stop.wait(0.1)

    def is_active(self) -> tuple[bool, float]:
        """True if currently within cooldown after a sustained loud event."""
        active = (time.time() - self._last_event_ts) <= self.cooldown_s
        return active, self._last_rms

    def consume_event(self) -> bool:
        """Returns True exactly once per fresh sustained-loud event."""
        if self._fresh_loud_event:
            self._fresh_loud_event = False
            return True
        return False


def run_watcher(
    redis_stream_key: str,
    redis_host: str,
    redis_port: int,
    instructions: list[str],
    vllm_host: str,
    vllm_port: int,
    model_name: str,
    num_frames_to_process: int,
    motion_threshold: float,
    motion_cooldown_s: float,
    idle_sleep_s: float,
    motion_ignore_regions: list[tuple[float, float, float, float]] | None = None,
    enable_audio: bool = False,
    audio_threshold_rms: float = 800.0,
    audio_cooldown_s: float = 8.0,
    audio_sustained_windows: int = 2,
):
    """
    Run the Watcher continuously to monitor frames from Redis stream.

    Args:
        redis_stream_key: Base Redis stream key (e.g., room name from config). Will use {key}:subsampled for video frames and {key}:logs for logs.
        redis_host: Redis server host.
        redis_port: Redis server port.
        instructions: List of monitoring instructions to check (from room config).
        vllm_host: vLLM server host.
        vllm_port: vLLM server port.
        model_name: Model name to use for inference (from room config).
        num_frames_to_process: Number of frames to analyze in each batch.
    """
    # Initialize Redis stream handler
    redis_handler = RedisStreamHandler(
        redis_host=redis_host,
        redis_port=redis_port,
    )

    # Initialize Watcher
    nanny_watcher = Watcher(
        instructions=instructions,
        vllm_host=vllm_host,
        vllm_port=vllm_port,
        model_name=model_name,
    )

    # Subsampled stream key
    subsampled_key = f"{redis_stream_key}:subsampled"
    logs_key = f"{redis_stream_key}:logs"
    logger.info(
        "Starting Watcher monitoring Redis",
        video_queue_key=subsampled_key,
        logs_queue_key=logs_key,
    )
    logger.info(
        "Using model", model_name=model_name, vllm_host=vllm_host, vllm_port=vllm_port
    )
    logger.info("Monitoring instructions", instructions=instructions)

    motion_gate = MotionGate(
        threshold=motion_threshold,
        cooldown_s=motion_cooldown_s,
        ignore_regions=motion_ignore_regions,
    )
    logger.info(
        "Motion gating enabled",
        threshold=motion_threshold,
        cooldown_s=motion_cooldown_s,
        idle_sleep_s=idle_sleep_s,
        ignore_regions=motion_ignore_regions,
    )

    audio_gate: AudioGate | None = None
    if enable_audio:
        audio_gate = AudioGate(
            redis_handler=redis_handler,
            room_key=redis_stream_key,
            threshold_rms=audio_threshold_rms,
            cooldown_s=audio_cooldown_s,
            sustained_windows=audio_sustained_windows,
        )
        audio_gate.start()
        logger.info(
            "Audio gating enabled (consuming from streamer's audio_rms stream)",
            audio_key=f"{redis_stream_key}:audio_rms",
            threshold_rms=audio_threshold_rms,
            cooldown_s=audio_cooldown_s,
            sustained_windows=audio_sustained_windows,
        )

    # Warm up vLLM with a single real request so the first motion event
    # doesn't pay the cold-start kernel-compile tax (often 30-180s).
    logger.info("Warming up vLLM (one inference call)...")
    warmup_started = time.time()
    warmup_frames = []
    waited = 0
    while not warmup_frames and waited < 30:
        warmup_frames = redis_handler.get_latest_frames(
            subsampled_key, num_frames_to_process
        )
        if not warmup_frames:
            time.sleep(0.5)
            waited += 0.5
    if warmup_frames:
        try:
            nanny_watcher.process_frames(warmup_frames)
            logger.info(
                "vLLM warmup complete",
                seconds=round(time.time() - warmup_started, 1),
            )
            # Reset motion baseline so warmup frames don't count as a "first event".
            motion_gate._prev_gray = None
            motion_gate._last_motion_ts = 0.0
        except Exception as e:
            logger.warning("vLLM warmup failed (non-fatal)", error=str(e))
    else:
        logger.warning("Could not warm up vLLM: no frames in stream after 30s")

    try:
        while True:
            # Get latest frames from Redis
            frames = redis_handler.get_latest_frames(
                subsampled_key, num_frames_to_process
            )

            if not frames:
                logger.warning("· no frames in stream", queue=subsampled_key)
                time.sleep(0.3)
                continue

            motion_run, motion_score = motion_gate.update(frames)
            audio_run, audio_rms = (False, 0.0)
            audio_event = False
            if audio_gate is not None:
                audio_run, audio_rms = audio_gate.is_active()
                audio_event = audio_gate.consume_event()
                # A *fresh* sustained loud event also fires the alert sound
                # immediately, regardless of what VLM concludes.
                if audio_event:
                    logger.info(
                        "🔔 audio alert: sustained loud noise",
                        rms=round(audio_rms, 1),
                    )
                    redis_handler.add_logs(
                        logs_key,
                        {
                            "timestamp": time.time(),
                            "should_alert": 1,
                            "awareness_level": "HIGH",
                            "reasoning": (
                                f"Audio: sustained loud noise "
                                f"(RMS={round(audio_rms, 1)} >= {audio_threshold_rms})."
                            ),
                        },
                    )
                    try:
                        play_alert("assets/alert.wav")
                    except Exception as e:
                        logger.warning(
                            "Could not play alert sound", error=str(e)
                        )

            should_run = motion_run or audio_run
            if not should_run:
                gate_log = {"motion": round(motion_score, 2)}
                if audio_gate is not None:
                    gate_log["audio"] = round(audio_rms, 1)
                logger.info("· idle", **gate_log)
                time.sleep(idle_sleep_s)
                continue

            trigger = (
                "audio" if (audio_run and not motion_run)
                else "audio+motion" if (audio_run and motion_run)
                else "motion"
            )
            gate_log = {
                "trigger": trigger,
                "motion": round(motion_score, 2),
                "frames": len(frames),
            }
            if audio_gate is not None:
                gate_log["audio"] = round(audio_rms, 1)
            logger.info("→ analyzing frames", **gate_log)

            # Process frames with Watcher
            result = nanny_watcher.process_frames(frames)

            if result["success"]:
                awareness = result["recommended_awareness_level"]
                if result["should_alert"]:
                    logger.info(
                        "🚨 ALERT",
                        level=awareness,
                        reason=result["reasoning"],
                    )
                else:
                    logger.info(
                        "✓ ok",
                        level=awareness,
                        reason=result["reasoning"],
                    )

                # Stream logs back to Redis
                log_data = {
                    "timestamp": time.time(),
                    "should_alert": int(result["should_alert"]),
                    "awareness_level": awareness,
                    "reasoning": result["reasoning"],
                }
                redis_handler.add_logs(logs_key, log_data)

                if result["should_alert"]:
                    try:
                        play_alert("assets/alert.wav")
                    except Exception as e:
                        logger.warning("Could not play alert sound", error=str(e))
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error("Error processing frames", error=error_msg)

                # Stream error to Redis
                error_data = {
                    "timestamp": time.time(),
                    "error": error_msg,
                }
                redis_handler.add_logs(logs_key, error_data)

                time.sleep(0.3)

    except KeyboardInterrupt:
        logger.info("Watcher stopped by user")
    except Exception as e:
        logger.error("Error in Watcher", error=e)
    finally:
        if audio_gate is not None:
            audio_gate.stop()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Watcher to monitor Redis stream frames based on room configuration."
    )

    parser.add_argument(
        "--config-file", required=True, help="Path to room configuration YAML file"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Only show alerts and errors (suppress INFO/WARNING)",
    )
    parser.add_argument(
        "--motion-threshold", type=float, default=2.0,
        help=(
            "Motion-gating threshold (mean abs pixel diff on a 160x90 grayscale "
            "downsample). Lower = more sensitive. Default: 2.0"
        ),
    )
    parser.add_argument(
        "--motion-cooldown", type=float, default=5.0,
        help="Keep analyzing this many seconds after last detected motion. Default: 5.0",
    )
    parser.add_argument(
        "--idle-sleep", type=float, default=1.0,
        help="Seconds to sleep between motion checks when idle. Default: 1.0",
    )
    parser.add_argument(
        "--no-motion-gate", action="store_true",
        help="Disable motion gating and run LLM on every batch (legacy behavior).",
    )
    parser.add_argument(
        "--motion-ignore", action="append", default=[],
        metavar="x0,y0,x1,y1",
        help=(
            "Mask out a region from motion detection (normalized [0,1] "
            "coords). Repeat for multiple regions. Example for top-right "
            "timestamp: --motion-ignore 0.7,0,1,0.1"
        ),
    )
    parser.add_argument(
        "--audio", action="store_true",
        help=(
            "Enable audio gating: pull audio from the camera RTSP stream "
            "via ffmpeg, trigger VLM analysis (and a direct alert) on "
            "sustained loud noise."
        ),
    )
    parser.add_argument(
        "--audio-threshold", type=float, default=800.0,
        help="Int16-RMS loudness threshold per 0.25s window. Default: 800",
    )
    parser.add_argument(
        "--audio-cooldown", type=float, default=8.0,
        help="Keep VLM analysis active this many seconds after the last loud "
             "audio event. Default: 8.0",
    )
    parser.add_argument(
        "--audio-sustained-windows", type=int, default=2,
        help="How many consecutive loud 0.25s windows count as a real event "
             "(filters out one-off claps/door-slams). Default: 2",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Pretty console output: short timestamp, colored level, padded event
    # message, key=value tail. Falls back to plain output if not a TTY.
    is_tty = sys.stderr.isatty()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(
                colors=is_tty,
                pad_event=32,
                sort_keys=True,
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.ERROR if args.quiet else logging.INFO
        ),
    )

    try:
        # Load room configuration from file
        room_config = load_room_config_file(args.config_file)
        config_file_path = Path(args.config_file)
        logger.info(
            f"Loaded configuration for room: {room_config.name}",
            config_file=str(config_file_path.resolve()),
        )

        # Extract parameters from config
        redis_stream_key = room_config.name
        instructions = room_config.instructions
        num_frames_to_process = room_config.num_frames_to_process

        # Ensure instructions are provided, as RoomConfig defaults to an empty list if not in YAML.
        if not instructions:
            logger.error(
                "The 'instructions' list is empty or missing in the configuration file. At least one instruction is required for the watcher.",
                config_file=str(config_file_path.resolve()),
            )
            exit(1)

        # If --no-motion-gate, set threshold to -inf so update() always returns True.
        effective_threshold = (
            float("-inf") if args.no_motion_gate else args.motion_threshold
        )

        ignore_regions: list[tuple[float, float, float, float]] = []
        for spec in args.motion_ignore:
            try:
                parts = [float(p) for p in spec.split(",")]
                if len(parts) != 4:
                    raise ValueError("expected exactly 4 comma-separated floats")
                ignore_regions.append(tuple(parts))  # type: ignore[arg-type]
            except Exception as e:
                logger.error("Invalid --motion-ignore value", spec=spec, error=str(e))
                exit(1)

        run_watcher(
            redis_stream_key=redis_stream_key,
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            instructions=instructions,
            vllm_host=VLLM_HOST,
            vllm_port=VLLM_PORT,
            model_name=LLM_MODEL_NAME,
            num_frames_to_process=num_frames_to_process,
            motion_threshold=effective_threshold,
            motion_cooldown_s=args.motion_cooldown,
            idle_sleep_s=args.idle_sleep,
            motion_ignore_regions=ignore_regions or None,
            enable_audio=args.audio,
            audio_threshold_rms=args.audio_threshold,
            audio_cooldown_s=args.audio_cooldown,
            audio_sustained_windows=args.audio_sustained_windows,
        )
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config_file}")
        exit(1)
    except Exception as e:
        logger.error("Failed to start watcher", error=str(e))
        exit(1)

import argparse
import contextlib
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import structlog
from dotenv import load_dotenv

from ai_baby_monitor.config import load_room_config_file
from ai_baby_monitor.stream import CameraStream, RedisStreamHandler

logger = structlog.get_logger()

load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")


class AudioStreamer:
    """Pulls audio from a RTSP URL via ffmpeg, computes RMS per 0.25s
    window, and pushes the values to Redis stream `<room>:audio_rms`.

    Runs in a background thread inside stream_to_redis so we share one
    container (and ideally one set of RTSP "client slots" from the
    camera/Synology's POV) with the video streamer rather than having
    a second host-side process compete for slots.
    """

    WINDOW_SECONDS = 0.25
    SAMPLE_RATE = 16000
    # Quiet windows below this RMS get throttled (only every Nth published)
    # to reduce Redis traffic. Loud spikes always publish so the watcher
    # never misses a real event.
    PUBLISH_FLOOR_RMS = 200.0
    QUIET_PUBLISH_EVERY_N = 4  # publish ~1Hz when quiet

    def __init__(
        self,
        rtsp_url: str,
        redis_handler: RedisStreamHandler,
        redis_stream_key: str,
        maxlen: int = 240,  # ~60s of 0.25s windows
    ):
        self.rtsp_url = rtsp_url
        self.redis_handler = redis_handler
        self.audio_key = f"{redis_stream_key}:audio_rms"
        self.maxlen = maxlen
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._stderr_path: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            with contextlib.suppress(Exception):
                self._proc.kill()

    def _spawn_ffmpeg(self) -> subprocess.Popen:
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-vn",
            "-map", "0:a",
            "-af", "aresample=async=1",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", str(self.SAMPLE_RATE),
            "-f", "s16le",
            "-flush_packets", "1",
            "-fflags", "+discardcorrupt",
            "-nostats",
            "-loglevel", "error",
            "pipe:1",
        ]
        # Real file for stderr, not a pipe (avoid kernel buffer deadlock).
        f = tempfile.NamedTemporaryFile(
            prefix="audio_streamer_stderr_", suffix=".log", delete=False
        )
        self._stderr_path = f.name
        return subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=f, bufsize=0,
        )

    def _run(self) -> None:
        bytes_per_window = int(self.SAMPLE_RATE * self.WINDOW_SECONDS) * 2
        backoff = 1.0
        while not self._stop.is_set():
            window_idx = 0
            try:
                self._proc = self._spawn_ffmpeg()
                logger.info("AudioStreamer ffmpeg started", pid=self._proc.pid)
                while not self._stop.is_set():
                    chunk = b""
                    eof = False
                    while len(chunk) < bytes_per_window:
                        more = self._proc.stdout.read(
                            bytes_per_window - len(chunk)
                        )
                        if not more:
                            eof = True
                            break
                        chunk += more
                    if eof:
                        break
                    samples = np.frombuffer(chunk, dtype=np.int16).astype(
                        np.float32
                    )
                    rms = float(np.sqrt(np.mean(samples * samples)))
                    # Always publish "interesting" windows (above the lower
                    # publish threshold) so the watcher never misses a
                    # spike, but throttle quiet windows to once a second
                    # to keep Redis traffic low.
                    is_interesting = rms >= self.PUBLISH_FLOOR_RMS
                    should_publish = (
                        is_interesting
                        or window_idx % self.QUIET_PUBLISH_EVERY_N == 0
                    )
                    if should_publish:
                        self.redis_handler.add_logs(
                            self.audio_key,
                            {"timestamp": time.time(), "rms": rms},
                            maxlen=self.maxlen,
                        )
                    window_idx += 1
            except Exception as e:
                logger.warning("AudioStreamer error", error=str(e))
            finally:
                stderr_tail = ""
                returncode = None
                if self._proc:
                    if self._proc.poll() is None:
                        with contextlib.suppress(Exception):
                            self._proc.kill()
                    with contextlib.suppress(Exception):
                        self._proc.wait(timeout=2)
                    returncode = self._proc.returncode
                if self._stderr_path:
                    with contextlib.suppress(Exception):
                        with open(self._stderr_path, "rb") as f:
                            stderr_tail = f.read().decode(
                                "utf-8", errors="replace"
                            ).strip()
                    with contextlib.suppress(Exception):
                        os.unlink(self._stderr_path)
                    self._stderr_path = None
                self._proc = None
            if self._stop.is_set():
                break
            if returncode == 0 and window_idx >= 20:
                backoff = 1.0
            logger.warning(
                "AudioStreamer ffmpeg exited; retrying",
                backoff_s=backoff,
                returncode=returncode,
                windows_processed=window_idx,
                stderr_tail=stderr_tail[-1500:] if stderr_tail else "",
            )
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 30.0)


def stream_to_redis(
    camera_uri: str,
    redis_stream_key: str,
    redis_host: str,
    redis_port: int,
    subsampled_stream_maxlen: int,
    save_stream_path: str | None,
    frame_width: int | None,
    frame_height: int | None,
    subsample_rate: int = 4,
    enable_audio: bool = False,
):
    """
    Stream camera frames to Redis short realtime and long subsampled queues.

    This function is used by the main entry point and expects configuration
    parameters to be loaded from the room's YAML config file.
    """
    # Convert camera_uri to int if it's a digit (for webcam index)
    if camera_uri.isdigit():
        camera_uri = int(camera_uri)

    # Set frame shape if both dimensions are provided
    frame_shape = None
    if frame_width is not None and frame_height is not None:
        frame_shape = (frame_width, frame_height)

    logger.info("Initializing camera stream", camera_uri=camera_uri)
    logger.info(
        "Streaming to Redis",
        redis_stream_key=redis_stream_key,
        redis_host=redis_host,
        redis_port=redis_port,
    )
    logger.info(
        f"Using subsample rate of 1 out of {subsample_rate} for subsampled queue"
    )

    try:
        # Initialize camera stream
        camera = CameraStream(
            uri=camera_uri,
            save_stream_path=save_stream_path,
            frame_shape=frame_shape,
        )

        # Initialize Redis stream handler
        redis_handler = RedisStreamHandler(
            redis_host=redis_host,
            redis_port=redis_port,
        )

        # Optionally start the audio streamer in the background. We do
        # it in this same process/container so both RTSP connections
        # come from the same host, which seems to coexist better with
        # the Synology share than two separate processes.
        audio_streamer = None
        if enable_audio and isinstance(camera_uri, str) and camera_uri.startswith("rtsp://"):
            audio_streamer = AudioStreamer(
                rtsp_url=camera_uri,
                redis_handler=redis_handler,
                redis_stream_key=redis_stream_key,
            )
            audio_streamer.start()
            logger.info(
                "AudioStreamer started",
                audio_key=f"{redis_stream_key}:audio_rms",
            )

        logger.info("Starting to stream to redis.")

        # Main streaming loop
        while True:
            # Only capture when a new frame is available
            frame = camera.capture_new_frame()
            if frame:
                # Always add frame to realtime queue
                redis_handler.add_frame(
                    frame, f"{redis_stream_key}:realtime", 3, approximate=False
                )

                # Add to subsampled queue every nth frame
                if frame.frame_idx % subsample_rate == 0:
                    redis_handler.add_frame(
                        frame,
                        f"{redis_stream_key}:subsampled",
                        subsampled_stream_maxlen,
                    )
                    logger.info(
                        "Added frame to subsampled queue",
                        frame_idx=frame.frame_idx,
                        timestamp=frame.timestamp,
                    )
            else:
                logger.warning("Failed to capture frame, retrying...")
                time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Stream interrupted by user")
    except Exception as e:
        logger.error("Error in streaming", error=e)
    finally:
        if "audio_streamer" in locals() and audio_streamer is not None:
            audio_streamer.stop()
        if "camera" in locals():
            camera.close()
        logger.info("Stream closed")


def parse_args():
    """Parse command line arguments for room-based configuration."""
    parser = argparse.ArgumentParser(
        description="Stream camera frames to Redis based on room configuration."
    )

    parser.add_argument(
        "--config-file", required=True, help="Path to room configuration YAML file"
    )
    parser.add_argument(
        "--save-stream-path",
        default=None,
        help="Path to save the video stream (optional)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Override camera URI with demo footage if available",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help=(
            "Also pull audio from the RTSP URL via ffmpeg and push per-window "
            "RMS values to Redis stream <room>:audio_rms. The watcher's "
            "AudioGate consumes this stream instead of opening its own RTSP "
            "session, which avoids starving the camera's client slots."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load room configuration from file
    try:
        room_config = load_room_config_file(args.config_file)
        config_file_path = Path(args.config_file)
        logger.info(
            f"Loaded configuration for room: {room_config.name}",
            config_file=str(config_file_path.resolve()),
        )

        # Extract camera settings from config
        camera_uri = room_config.camera_uri

        # Override with demo if requested
        if args.demo:
            camera_uri = "assets/demo/demo.mp4"
            logger.info("Using demo video source", camera_uri=camera_uri)

        # Use the room name as the redis stream key
        redis_stream_key = room_config.name

        stream_to_redis(
            camera_uri=camera_uri,
            redis_stream_key=redis_stream_key,
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            subsampled_stream_maxlen=room_config.subsampled_stream_maxlen,
            save_stream_path=args.save_stream_path,
            frame_width=room_config.frame_width,
            frame_height=room_config.frame_height,
            subsample_rate=room_config.subsample_rate,
            enable_audio=args.audio,
        )
    except Exception as e:
        logger.error("Failed to start streaming", error=e)
        exit(1)

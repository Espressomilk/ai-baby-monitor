import argparse
import io
import os

import streamlit as st
import structlog
from PIL import Image
from dotenv import load_dotenv

from ai_baby_monitor.config import load_multiple_room_configs
from ai_baby_monitor.ui import (
    display_sidebar,
    fetch_logs,
    get_cached_redis_handler,
    get_last_image_with_timestamp,
    render_logs,
)

load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")

logger = structlog.get_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Baby Monitor Stream Viewer")
    parser.add_argument(
        "--config-files",
        nargs="+",  # Accept one or more config files
        required=True,
        help="Paths to room configuration YAML files",
    )

    return parser.parse_known_args()[0]


st.set_page_config(
    page_title="Baby Monitor Stream Viewer",
    page_icon="👶",
)

# Storing room configs in state to avoid reparsing
if "room_configs" not in st.session_state:
    args = parse_args()
    room_configs = load_multiple_room_configs(args.config_files)
    if not room_configs:
        st.error("No room config file was provided.")
    st.session_state["room_configs"] = room_configs

redis_handler = get_cached_redis_handler(REDIS_HOST, REDIS_PORT)

# Sidebar has a side-effect of setting selected config and real-time stream vs historic logs
display_sidebar(
    room_configs_key="room_configs",
    config_key="selected_config",
    mode_key="selected_mode",
)
selected_config = st.session_state["selected_config"]
selected_mode = st.session_state["selected_mode"]

if selected_mode == "Real-time stream":
    st.title("Baby Monitor Stream Viewer")

    # Placeholders for the stream and logs
    stream_placeholder = st.empty()
    log_placeholder = st.empty()

    frame_key = f"{selected_config.name}:realtime"
    last_frame_id = "$"  # only wait for *future* frames

    # Render whatever is on the stream right now so the page isn't blank
    # while we wait for the first new frame.
    image, timestamp = get_last_image_with_timestamp(
        redis_handler, selected_config.name
    )
    if image:
        with stream_placeholder.container():
            st.image(image, use_container_width=True)
            with st.expander("Frame Info"):
                st.caption(f"Timestamp: {timestamp}")

    while True:
        # Block on the Redis stream until a new frame is published, or
        # 1s passes (so we still refresh logs even when the camera is
        # silent). This keeps CPU near zero between frames instead of
        # spinning at 116%.
        result = redis_handler.wait_for_new_frame(
            frame_key, last_id=last_frame_id, block_ms=1000
        )
        if result is not None:
            last_frame_id, frame = result
            if frame is not None:
                jpeg_bytes = bytes(frame.frame_data)
                image = Image.open(io.BytesIO(jpeg_bytes))
                with stream_placeholder.container():
                    st.image(image, use_container_width=True)
                    with st.expander("Frame Info"):
                        st.caption(f"Timestamp: {frame.timestamp}")

        # Refresh logs on every loop iteration (cheap; just an xrevrange).
        with log_placeholder.container(height=350):
            with st.expander("LLM Logs", expanded=True, icon="🤖"):
                logs = fetch_logs(
                    redis_handler,
                    selected_config.name,
                    num_logs=1,
                )
                render_logs(logs)

else:
    st.title("Baby Monitor Logs Viewer")

    num_logs = st.number_input(
        "Number of latest logs to fetch",
        min_value=1,
        value=100,
        step=1,
    )
    if st.button("Fetch logs", type="primary"):
        logs = fetch_logs(redis_handler, selected_config.name, num_logs=num_logs)[::-1]
        with st.container(height=800):
            render_logs(logs)

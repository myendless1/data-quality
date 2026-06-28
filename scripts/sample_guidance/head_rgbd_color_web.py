#!/usr/bin/env python3
"""Serve the Astribot head_rgbd color_image stream as a small MJPEG website."""

from __future__ import annotations

import argparse
import asyncio
import signal
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from aiohttp import web

import rospy
from camera_subscriber_py.sub.image_subscriber import ImageSubscriber


DEFAULT_TOPIC = "/astribot_camera/head_rgbd/color_image"


@dataclass
class StreamState:
    frame_bgr: np.ndarray | None = None
    timestamp: float = 0.0
    seq: int = 0
    shape: tuple[int, ...] | None = None
    encoded_frames: int = 0


class HeadRgbdColorStream:
    def __init__(self, topic: str, jpeg_quality: int, stale_seconds: float):
        self.topic = topic
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self.stale_seconds = stale_seconds
        self._lock = threading.Lock()
        self._state = StreamState()
        self._subscriber: ImageSubscriber | None = None

    def start(self) -> None:
        rospy.init_node("head_rgbd_color_web", anonymous=True, disable_signals=True)
        self._subscriber = ImageSubscriber(
            self.topic,
            self._image_callback,
            low_latency_require=False,
            batch_images_only=False,
        )
        spin_thread = threading.Thread(target=rospy.spin, daemon=True)
        spin_thread.start()

    def _image_callback(self, msg, array: np.ndarray | None) -> None:
        if array is None or getattr(msg, "image_type", None) != 0:
            return
        try:
            frame_bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        except cv2.error:
            return
        with self._lock:
            self._state.frame_bgr = frame_bgr
            self._state.timestamp = time.time()
            self._state.seq += 1
            self._state.shape = tuple(frame_bgr.shape)

    def snapshot(self) -> tuple[bytes, dict[str, object]]:
        with self._lock:
            frame = None if self._state.frame_bgr is None else self._state.frame_bgr.copy()
            timestamp = self._state.timestamp
            seq = self._state.seq
            shape = self._state.shape

        age = time.time() - timestamp if timestamp else None
        stale = age is None or age > self.stale_seconds
        if frame is None:
            frame = self._placeholder("waiting for head_rgbd/color_image")
        elif stale:
            frame = frame.copy()
            self._draw_banner(frame, f"stream stale: {age:.1f}s")

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            encoded = cv2.imencode(".jpg", self._placeholder("encode failed"))[1]

        with self._lock:
            self._state.encoded_frames += 1

        meta = {
            "topic": self.topic,
            "seq": seq,
            "age_seconds": age,
            "stale": stale,
            "shape": shape,
        }
        return encoded.tobytes(), meta

    def status(self) -> dict[str, object]:
        with self._lock:
            timestamp = self._state.timestamp
            seq = self._state.seq
            shape = self._state.shape
            encoded_frames = self._state.encoded_frames
        age = time.time() - timestamp if timestamp else None
        return {
            "topic": self.topic,
            "frames_received": seq,
            "frames_served": encoded_frames,
            "last_frame_age_seconds": age,
            "stale": age is None or age > self.stale_seconds,
            "shape": shape,
            "ros_shutdown": rospy.is_shutdown(),
        }

    @staticmethod
    def _placeholder(text: str) -> np.ndarray:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[:] = (24, 24, 24)
        cv2.putText(
            image,
            text,
            (54, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        return image

    @staticmethod
    def _draw_banner(frame: np.ndarray, text: str) -> None:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 54), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def html_page(topic: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Astribot head_rgbd</title>
  <style>
    body {{
      margin: 0;
      background: #111;
      color: #eee;
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      height: 44px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 14px;
      background: #1d1d1d;
      border-bottom: 1px solid #333;
      box-sizing: border-box;
    }}
    main {{
      height: calc(100vh - 44px);
      display: grid;
      place-items: center;
      overflow: hidden;
    }}
    img {{
      max-width: 100vw;
      max-height: calc(100vh - 44px);
      width: auto;
      height: auto;
      object-fit: contain;
    }}
    code {{ color: #a6d5ff; }}
    #status {{ color: #cfcfcf; }}
  </style>
</head>
<body>
  <header>
    <div><code>{topic}</code></div>
    <div id="status">connecting</div>
  </header>
  <main>
    <img src="/stream.mjpg" alt="head_rgbd color stream">
  </main>
  <script>
    async function updateStatus() {{
      try {{
        const response = await fetch('/status', {{cache: 'no-store'}});
        const data = await response.json();
        const age = data.last_frame_age_seconds == null ? 'none' : data.last_frame_age_seconds.toFixed(2) + 's';
        document.getElementById('status').textContent =
          `frames=${{data.frames_received}} age=${{age}} stale=${{data.stale}}`;
      }} catch (error) {{
        document.getElementById('status').textContent = 'status unavailable';
      }}
    }}
    updateStatus();
    setInterval(updateStatus, 1000);
  </script>
</body>
</html>
"""


async def index(request: web.Request) -> web.Response:
    stream: HeadRgbdColorStream = request.app["stream"]
    return web.Response(text=html_page(stream.topic), content_type="text/html")


async def status(request: web.Request) -> web.Response:
    stream: HeadRgbdColorStream = request.app["stream"]
    return web.json_response(stream.status())


async def snapshot(request: web.Request) -> web.Response:
    stream: HeadRgbdColorStream = request.app["stream"]
    jpeg, _ = stream.snapshot()
    return web.Response(body=jpeg, content_type="image/jpeg")


async def mjpeg_stream(request: web.Request) -> web.StreamResponse:
    stream: HeadRgbdColorStream = request.app["stream"]
    fps = request.app["fps"]
    delay = 1.0 / fps
    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        },
    )
    await response.prepare(request)

    try:
        while not rospy.is_shutdown():
            jpeg, _ = stream.snapshot()
            await response.write(b"--frame\r\n")
            await response.write(b"Content-Type: image/jpeg\r\n")
            await response.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
            await response.write(jpeg)
            await response.write(b"\r\n")
            await asyncio.sleep(delay)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--stale-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stream = HeadRgbdColorStream(args.topic, args.jpeg_quality, args.stale_seconds)
    stream.start()

    app = web.Application()
    app["stream"] = stream
    app["fps"] = max(1.0, args.fps)
    app.router.add_get("/", index)
    app.router.add_get("/status", status)
    app.router.add_get("/snapshot.jpg", snapshot)
    app.router.add_get("/stream.mjpg", mjpeg_stream)

    def shutdown(*_: object) -> None:
        rospy.signal_shutdown("head_rgbd_color_web stopped")

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"Serving {args.topic} at http://{args.host}:{args.port}/")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()

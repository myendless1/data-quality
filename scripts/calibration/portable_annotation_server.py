#!/usr/bin/env python3
"""Standalone bbox annotation server for a portable annotation package."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


def read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing manifest.json, annotations.json, samples/, and annotation_ui.html.",
    )
    return parser


class AnnotationHandler(BaseHTTPRequestHandler):
    data_dir: Path

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, status: int, value: object) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in {"/", "/index.html"}:
            self.serve_file(self.data_dir / "annotation_ui.html")
            return
        if path == "/api/manifest":
            self.send_json(200, read_json(self.data_dir / "manifest.json", {}))
            return
        if path == "/api/annotations":
            self.send_json(
                200,
                read_json(
                    self.data_dir / "annotations.json",
                    {"samples": [], "width": 1280, "height": 720},
                ),
            )
            return
        if path.startswith("/data/"):
            rel = Path(path.removeprefix("/data/"))
            target = (self.data_dir / rel).resolve()
            if not str(target).startswith(str(self.data_dir.resolve())):
                self.send_error(403)
                return
            self.serve_file(target)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/annotations":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length)
        try:
            data = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self.send_json(400, {"ok": False, "error": str(exc)})
            return
        write_json(self.data_dir / "annotations.json", data)
        self.send_json(200, {"ok": True})

    def serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        payload = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    args = build_argparser().parse_args()
    data_dir = args.data_dir.resolve()
    if not (data_dir / "manifest.json").exists():
        raise FileNotFoundError(f"Missing manifest.json in {data_dir}")
    if not (data_dir / "annotation_ui.html").exists():
        raise FileNotFoundError(f"Missing annotation_ui.html in {data_dir}")
    AnnotationHandler.data_dir = data_dir
    server = ThreadingHTTPServer((args.host, args.port), AnnotationHandler)
    print(f"Open http://{args.host}:{args.port}")
    print(f"Saving annotations to {data_dir / 'annotations.json'}")
    server.serve_forever()


if __name__ == "__main__":
    main()

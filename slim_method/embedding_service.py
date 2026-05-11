import argparse
import json
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_PATH = "models/embedding_model"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 6000


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


class HFEmbeddingRuntime:
    def __init__(self, model_path: str, max_length: int = 8192) -> None:
        self.model_path = model_path
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.inference_mode()
    def embed(self, input_texts: List[str]) -> List[List[float]]:
        batch = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        outputs = self.model(**batch)
        embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().tolist()


def build_handler(runtime: HFEmbeddingRuntime):
    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self._write_json(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": runtime.model_path, "object": "model"}],
                    },
                )
                return
            self._write_json(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            if self.path != "/v1/embeddings":
                self._write_json(404, {"error": {"message": "not found"}})
                return
            try:
                content_len = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
                req = json.loads(raw)
            except Exception as exc:
                self._write_json(400, {"error": {"message": f"invalid json: {exc}"}})
                return

            input_data = req.get("input", "")
            model_name = req.get("model", runtime.model_path)
            if isinstance(input_data, str):
                texts = [input_data]
            elif isinstance(input_data, list):
                texts = [str(item) for item in input_data]
            else:
                self._write_json(400, {"error": {"message": "input must be str or list[str]"}})
                return

            try:
                vectors = runtime.embed(texts)
            except Exception as exc:
                self._write_json(500, {"error": {"message": f"embedding failed: {exc}"}})
                return

            self._write_json(
                200,
                {
                    "object": "list",
                    "model": model_name,
                    "data": [
                        {"object": "embedding", "index": idx, "embedding": vec}
                        for idx, vec in enumerate(vectors)
                    ],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )

        def log_message(self, fmt: str, *args) -> None:
            return

    return Handler


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class EmbeddingServiceProcess:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        start_cmd: str = "",
        wait_timeout: float = 360.0,
        workdir: Path | None = None,
    ) -> None:
        self.model_path = model_path
        self.host = host
        self.port = int(port)
        self.start_cmd = start_cmd.strip()
        self.wait_timeout = float(wait_timeout)
        self.workdir = workdir
        self.proc: subprocess.Popen | None = None
        self._external_existing = False
        self._logs: deque[str] = deque(maxlen=200)
        self._log_thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _healthcheck(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _resolve_python_executable(self) -> str:
        candidates: list[Path] = []
        if self.workdir is not None:
            candidates.append((self.workdir / ".venv" / "bin" / "python").resolve())
        candidates.append((Path(sys.prefix) / "bin" / "python").resolve())
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return str(candidate)
        return sys.executable

    def _build_start_cmd(self) -> list[str]:
        if self.start_cmd:
            return shlex.split(self.start_cmd)
        return [
            self._resolve_python_executable(),
            str(Path(__file__).resolve()),
            "serve",
            "--model-path",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

    def _stream_child_output(self) -> None:
        if self.proc is None or self.proc.stdout is None:
            return
        for line in self.proc.stdout:
            text = line.rstrip("\n")
            self._logs.append(text)
            print(f"[SLIM-EMBED-CHILD] {text}", flush=True)

    def start(self) -> None:
        if self._healthcheck():
            self._external_existing = True
            print(f"[SLIM-EMBED] already running at {self.base_url}", flush=True)
            return

        cmd = self._build_start_cmd()
        print(f"[SLIM-EMBED] starting: {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.workdir) if self.workdir else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._log_thread = threading.Thread(target=self._stream_child_output, daemon=True)
        self._log_thread.start()

        start = time.monotonic()
        while True:
            if self._healthcheck():
                print(f"[SLIM-EMBED] ready at {self.base_url}", flush=True)
                return
            if self.proc.poll() is not None:
                output = "\n".join(self._logs)
                raise RuntimeError(f"embedding service exited early code={self.proc.returncode}\n{output}")
            if time.monotonic() - start > self.wait_timeout:
                tail = "\n".join(self._logs)
                raise TimeoutError(f"embedding service not ready within {self.wait_timeout}s\nrecent logs:\n{tail}")
            time.sleep(1)

    def stop(self) -> None:
        if self._external_existing:
            print("[SLIM-EMBED] external process detected; skip stop.", flush=True)
            return
        if self.proc is None or self.proc.poll() is not None:
            return
        print("[SLIM-EMBED] stopping embedding service...", flush=True)
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
        print("[SLIM-EMBED] stopped.", flush=True)


def serve(args: argparse.Namespace) -> int:
    runtime = HFEmbeddingRuntime(model_path=args.model_path, max_length=args.max_length)
    server = ReusableThreadingHTTPServer((args.host, args.port), build_handler(runtime))
    print(f"[SLIM-EMBED] ready http://{args.host}:{args.port}/v1/embeddings", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def start_and_wait(args: argparse.Namespace) -> int:
    svc = EmbeddingServiceProcess(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        start_cmd=args.start_cmd,
        wait_timeout=args.wait_timeout,
        workdir=Path(__file__).resolve().parent,
    )
    if args.print_config:
        print(
            json.dumps(
                {
                    "model_path": args.model_path,
                    "host": args.host,
                    "port": args.port,
                    "start_cmd": args.start_cmd or "python embedding_service.py serve ...",
                    "base_url": svc.base_url,
                },
                ensure_ascii=True,
                indent=2,
            )
        )
        return 0

    try:
        svc.start()
        if args.daemon:
            return 0
        print("[SLIM-EMBED] running. Press Ctrl+C to stop.", flush=True)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.daemon:
            svc.stop()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SLIM OpenAI-compatible embedding service.")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Run the embedding HTTP server.")
    serve_parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    serve_parser.add_argument("--host", default=DEFAULT_HOST)
    serve_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve_parser.add_argument("--max-length", type=int, default=8192)

    start_parser = subparsers.add_parser("start", help="Start the server and wait until ready.")
    start_parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    start_parser.add_argument("--host", default=DEFAULT_HOST)
    start_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    start_parser.add_argument("--start-cmd", default="")
    start_parser.add_argument("--wait-timeout", type=float, default=360.0)
    start_parser.add_argument("--daemon", action="store_true")
    start_parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "serve":
        return serve(args)
    if args.command == "start":
        return start_and_wait(args)
    raise SystemExit("usage: embedding_service.py {serve,start} ...")


if __name__ == "__main__":
    raise SystemExit(main())

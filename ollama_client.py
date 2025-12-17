import json
import queue
import threading
import time
import urllib.request


def read_text_file(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def read_prompt_file(path: str) -> str:
    raw = read_text_file(path)
    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def render_prompt_template(template: str, context: dict) -> str:
    rendered = template
    for k, v in context.items():
        rendered = rendered.replace("{{" + k + "}}", str(v))
    return rendered.strip()


def normalize_ollama_url(url_or_base: str) -> str:
    u = (url_or_base or "").strip().rstrip("/")
    if not u:
        return "http://localhost:11434/api/generate"
    if "/api/" in u:
        return u
    return u + "/api/generate"


def normalize_ollama_model(model: str) -> str:
    m = (model or "").strip()
    if m.startswith("ollama/"):
        m = m[len("ollama/") :]
    return m or "llama3.1"


def ollama_generate(url: str, model: str, prompt: str, timeout_s: float) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(
        "utf-8"
    )
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("response") or "").strip()


class OllamaWorker:
    def __init__(
        self, url: str, model: str, prompt_template: str, timeout_s: float = 30.0
    ):
        self.url = url
        self.model = model
        self.prompt_template = prompt_template
        self.timeout_s = timeout_s

        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="ollama-worker", daemon=True
        )

        self.last_text = None
        self.last_error = None
        self.last_ts = 0.0

        self._thread.start()

    def submit(self, context: dict) -> bool:
        try:
            self._q.put_nowait(context)
            return True
        except queue.Full:
            return False

    def close(self):
        self._stop.set()
        try:
            self._q.put_nowait({})
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)

    def _run(self):
        while not self._stop.is_set():
            ctx = self._q.get()
            if self._stop.is_set():
                return
            if not ctx:
                continue
            try:
                prompt = render_prompt_template(self.prompt_template, ctx)
                self.last_text = ollama_generate(
                    self.url, self.model, prompt, self.timeout_s
                )
                self.last_error = None
                self.last_ts = time.time()
                if self.last_text:
                    print(f"[ollama] {self.last_text}")
                else:
                    print("[ollama] (empty response)")
            except Exception as e:
                self.last_error = str(e)
                self.last_ts = time.time()
                print(f"[ollama] error: {self.last_error}")

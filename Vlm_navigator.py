"""
Part 2: LLM Navigation Controller — pluggable local GGUF backends

Supported backends — set MODEL_BACKEND below or via env var:

  "moondream_text"      moondream2-text-model-f16_ct-vicuna.gguf
                        Text-only, Vicuna format. No image. No mmproj.
                        ← use this if you only have the text GGUF

  "moondream_vision"    moondream2-050824-f16.gguf  (Aug 2024 vision release)
                        Full vision model. Needs matching mmproj GGUF.
                        Uses MoondreamChatHandler.

  "smolvlm2_2b"         SmolVLM2-2.2B-Instruct-Q4_K_M.gguf
                        2B vision model, Q4_K_M quantised (1.1 GB).
                        Needs mmproj GGUF. Uses Llava15ChatHandler.

  "smolvlm2_500m"       SmolVLM2-500M-Video-Instruct-f16.gguf
                        500M video-capable model, full f16 precision.
                        Needs mmproj GGUF. Uses Llava15ChatHandler.
                        Lighter than 2B but larger in RAM than a quantised build.

Switch by changing MODEL_BACKEND here, or:
    MODEL_BACKEND=smolvlm2_2b python main.py

State  → Python-style key=value string  (no JSON)
Output → command=<cmd> magnitude=<float> reason=<phrase>

Install llama-cpp-python with vision support:
    CMAKE_ARGS="-DLLAVA_BUILD=on" pip install llama-cpp-python
    # add -DLLAMA_CUBLAS=on for CUDA GPU offload
"""

import os
import re
import time
import math
import threading
import textwrap
import cv2
import numpy as np
from PIL import Image
import io
import base64

from llama_cpp import Llama


# ═══════════════════════════════════════════════════════════════════════════
# ① SWITCH MODELS HERE
# ═══════════════════════════════════════════════════════════════════════════

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "moondream_text")
# Options:
#   "moondream_text"   "moondream_vision"
#   "smolvlm2_2b"      "smolvlm2_500m"


# ── File paths ──────────────────────────────────────────────────────────────
# Override any of these via env vars, or just edit the default strings.

# ── moondream_text ──
MOONDREAM_TEXT_PATH = os.getenv(
    "MOONDREAM_TEXT_PATH",
    "moondream2-text-model-f16_ct-vicuna.gguf",
)

# ── moondream_vision (Aug-2024 release) ──
# You must also supply the matching mmproj file.
# Download from: https://huggingface.co/vikhyatk/moondream2 (GGUF branch)
MOONDREAM_VIS_PATH = os.getenv(
    "MOONDREAM_VIS_PATH",
    "moondream2-050824-f16.gguf",           # ← your file
)
MOONDREAM_MMPROJ = os.getenv(
    "MOONDREAM_MMPROJ",
    "moondream2-mmproj-f16.gguf",           # companion mmproj
)

# ── smolvlm2_2b ──
# Download from: https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF
SMOLVLM2_2B_PATH = os.getenv(
    "SMOLVLM2_2B_PATH",
    "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",  # ← your file
)
SMOLVLM2_2B_MMPROJ = os.getenv(
    "SMOLVLM2_2B_MMPROJ",
    "mmproj-SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
)

# ── smolvlm2_500m ──
# Download from: https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF
SMOLVLM2_500M_PATH = os.getenv(
    "SMOLVLM2_500M_PATH",
    "SmolVLM2-500M-Video-Instruct-f16.gguf",  # ← your file
)
SMOLVLM2_500M_MMPROJ = os.getenv(
    "SMOLVLM2_500M_MMPROJ",
    "mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf",
)


# ── Shared inference settings ───────────────────────────────────────────────

N_CTX              = 2048   # context window (increase if model needs more)
N_THREADS          = 4      # CPU threads
N_GPU_LAYERS       = 0      # >0 = offload layers to GPU (needs CUDA build)
MAX_TOKENS         = 80     # one short line is enough
TEMPERATURE        = 0.1    # low = deterministic
CALL_EVERY_N_STEPS = 10     # how often to fire inference (control steps)
JPEG_QUALITY       = 60     # image compression for vision backends


# ═══════════════════════════════════════════════════════════════════════════
# Valid command set — must match part3_command_mapper.py
# ═══════════════════════════════════════════════════════════════════════════

VALID_COMMANDS = {
    "throttle_up", "throttle_down",
    "move_forward", "move_backward",
    "move_left",    "move_right",
    "move_up",      "move_down",
    "rotate_left",  "rotate_right",
    "hover",
}


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_state_str(state: dict) -> str:
    """
    Drone + obstacle-env state dict → compact Python-style key=value string.
    All backends embed this in their prompt.
    """
    pos  = state.get("position",        [0.0, 0.0, 0.0])
    vel  = state.get("velocity",        [0.0, 0.0, 0.0])
    rpy  = state.get("orientation_rpy", [0.0, 0.0, 0.0])
    goal = state.get("goal_position",   [5.0, 5.0, 1.0])

    dist     = state.get("distance_to_goal",      "?")
    bearing  = state.get("bearing_to_goal_rad",   "?")
    near_obs = state.get("nearest_obstacle_dist", "?")
    reached  = state.get("goal_reached",          False)

    compass = _compass(math.degrees(bearing) % 360) \
              if isinstance(bearing, (int, float)) else "?"

    return (
        f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  "
        f"vel=({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})  "
        f"roll={rpy[0]:.2f} pitch={rpy[1]:.2f} yaw={rpy[2]:.2f}  "
        f"goal=({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})  "
        f"dist_to_goal={dist}m  bearing={compass}  "
        f"nearest_obstacle={near_obs}m  goal_reached={reached}"
    )


def _compass(deg: float) -> str:
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][round(deg / 45) % 8]


def _to_b64uri(frame: np.ndarray) -> str:
    """BGR ndarray → base64 JPEG data-URI (vision backends only)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ═══════════════════════════════════════════════════════════════════════════
# Prompt templates
# ═══════════════════════════════════════════════════════════════════════════

_RULE_BLOCK = (
    "Rules:\n"
    "- alt < 0.3m  →  throttle_up magnitude=0.8  (emergency)\n"
    "- Prefer magnitude 0.3-0.7 for smooth flight\n"
    "- Navigate from [0,0] to goal [5,5], avoid obstacles\n"
    "- One small step at a time\n"
)

_CMD_LIST = (
    "Commands: throttle_up throttle_down move_forward move_backward "
    "move_left move_right rotate_left rotate_right hover"
)

# Text-only template (no image token)
_TMPL_TEXT = textwrap.dedent("""\
    You are a drone autopilot. Reply with ONLY one line:
    command=<cmd> magnitude=<0.0-1.0> reason=<short phrase>

    {cmd_list}

    {rules}
    State: {state_str}

    Best command?
""")

# Vision template for Moondream (image embedded via llama-cpp, not inline token)
_TMPL_MOONDREAM_VIS = textwrap.dedent("""\
    You are a drone autopilot. Reply with ONLY one line:
    command=<cmd> magnitude=<0.0-1.0> reason=<short phrase>

    {cmd_list}

    {rules}
    State: {state_str}

    What is the best command?
""")

# Vision template for SmolVLM2 — chat template handles image tokens internally,
# so no <image> placeholder is needed in the text string
_TMPL_SMOLVLM = textwrap.dedent("""\
    You are a drone autopilot. Reply with ONLY one line:
    command=<cmd> magnitude=<0.0-1.0> reason=<short phrase>

    {cmd_list}

    {rules}
    State: {state_str}

    Best command?
""")


def _fill(tmpl: str, state_str: str) -> str:
    return tmpl.format(cmd_list=_CMD_LIST, rules=_RULE_BLOCK, state_str=state_str)


# ═══════════════════════════════════════════════════════════════════════════
# Backend base
# ═══════════════════════════════════════════════════════════════════════════

class _BackendBase:
    USES_VISION: bool = False

    def load(self): raise NotImplementedError
    def query(self, frame: np.ndarray, state: dict) -> str: raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# Backend A — Moondream2 text-only  (moondream2-text-model-f16_ct-vicuna.gguf)
# ═══════════════════════════════════════════════════════════════════════════

class MoondreamTextBackend(_BackendBase):
    """
    Text-only Moondream2 in Vicuna chat format.
    Image is ignored — all navigation context comes from the state string.
    No mmproj file required.
    """
    USES_VISION = False

    def load(self):
        print(f"[LLM] Backend : moondream_text  (text-only, Vicuna)")
        print(f"      model   : {MOONDREAM_TEXT_PATH}")
        self._llm = Llama(
            model_path   = MOONDREAM_TEXT_PATH,
            n_ctx        = N_CTX,
            n_threads    = N_THREADS,
            n_gpu_layers = N_GPU_LAYERS,
            verbose      = False,
            # Vicuna format is auto-detected from GGUF metadata.
            # No chat_handler or chat_format needed.
        )

    def query(self, frame: np.ndarray, state: dict) -> str:
        prompt = _fill(_TMPL_TEXT, build_state_str(state))
        # Vicuna wire format: USER: … \nASSISTANT:
        resp = self._llm(
            f"USER: {prompt}\nASSISTANT:",
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
            stop        = ["\n", "USER:"],
        )
        return resp["choices"][0]["text"].strip()


# ═══════════════════════════════════════════════════════════════════════════
# Backend B — Moondream2 vision  (moondream2-050824-f16.gguf)
# ═══════════════════════════════════════════════════════════════════════════

class MoondreamVisionBackend(_BackendBase):
    """
    Full Moondream2 vision model (Aug-2024 release).
    Requires: moondream2-050824-f16.gguf  +  moondream2-mmproj-f16.gguf

    Note: the correct chat_format string for llama-cpp-python is "moondream"
    (not "moondream2"). MoondreamChatHandler handles this automatically.
    """
    USES_VISION = True

    def load(self):
        from llama_cpp.llama_chat_format import MoondreamChatHandler
        print(f"[LLM] Backend : moondream_vision  (Aug-2024 vision release)")
        print(f"      model   : {MOONDREAM_VIS_PATH}")
        print(f"      mmproj  : {MOONDREAM_MMPROJ}")
        self._llm = Llama(
            model_path   = MOONDREAM_VIS_PATH,
            chat_handler = MoondreamChatHandler(clip_model_path=MOONDREAM_MMPROJ),
            n_ctx        = N_CTX,
            n_threads    = N_THREADS,
            n_gpu_layers = N_GPU_LAYERS,
            verbose      = False,
        )

    def query(self, frame: np.ndarray, state: dict) -> str:
        question = _fill(_TMPL_MOONDREAM_VIS, build_state_str(state))
        # Moondream requires image BEFORE text in the content list
        resp = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _to_b64uri(frame)}},
                {"type": "text",      "text": question},
            ]}],
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
        )
        return resp["choices"][0]["message"]["content"].strip()


# ═══════════════════════════════════════════════════════════════════════════
# Backend C — SmolVLM2-2.2B  (SmolVLM2-2.2B-Instruct-Q4_K_M.gguf)
# ═══════════════════════════════════════════════════════════════════════════

class SmolVLM2_2BBackend(_BackendBase):
    """
    SmolVLM2 2.2B Instruct, Q4_K_M quantised (≈1.1 GB).
    Requires: SmolVLM2-2.2B-Instruct-Q4_K_M.gguf  +  mmproj GGUF.

    Uses Llava15ChatHandler — SmolVLM2 follows the LLaVA-1.5 wire protocol.
    The GGUF metadata contains the correct chat template so no chat_format
    string is needed.
    """
    USES_VISION = True

    def load(self):
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        print(f"[LLM] Backend : smolvlm2_2b  (2.2B Q4_K_M)")
        print(f"      model   : {SMOLVLM2_2B_PATH}")
        print(f"      mmproj  : {SMOLVLM2_2B_MMPROJ}")
        self._llm = Llama(
            model_path   = SMOLVLM2_2B_PATH,
            chat_handler = Llava15ChatHandler(clip_model_path=SMOLVLM2_2B_MMPROJ),
            n_ctx        = N_CTX,
            n_threads    = N_THREADS,
            n_gpu_layers = N_GPU_LAYERS,
            verbose      = False,
        )

    def query(self, frame: np.ndarray, state: dict) -> str:
        question = _fill(_TMPL_SMOLVLM, build_state_str(state))
        resp = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _to_b64uri(frame)}},
                {"type": "text",      "text": question},
            ]}],
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
        )
        return resp["choices"][0]["message"]["content"].strip()


# ═══════════════════════════════════════════════════════════════════════════
# Backend D — SmolVLM2-500M  (SmolVLM2-500M-Video-Instruct-f16.gguf)
# ═══════════════════════════════════════════════════════════════════════════

class SmolVLM2_500MBackend(_BackendBase):
    """
    SmolVLM2 500M Video-Instruct, full f16 precision.
    Requires: SmolVLM2-500M-Video-Instruct-f16.gguf  +  mmproj GGUF.

    Same handler as the 2B (Llava15ChatHandler) — lighter model, but f16
    means it's actually larger in RAM than a quantised 500M would be.
    Fastest inference of the vision models but lowest quality.
    """
    USES_VISION = True

    def load(self):
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        print(f"[LLM] Backend : smolvlm2_500m  (500M f16, video-instruct)")
        print(f"      model   : {SMOLVLM2_500M_PATH}")
        print(f"      mmproj  : {SMOLVLM2_500M_MMPROJ}")
        self._llm = Llama(
            model_path   = SMOLVLM2_500M_PATH,
            chat_handler = Llava15ChatHandler(clip_model_path=SMOLVLM2_500M_MMPROJ),
            n_ctx        = N_CTX,
            n_threads    = N_THREADS,
            n_gpu_layers = N_GPU_LAYERS,
            verbose      = False,
        )

    def query(self, frame: np.ndarray, state: dict) -> str:
        question = _fill(_TMPL_SMOLVLM, build_state_str(state))
        resp = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _to_b64uri(frame)}},
                {"type": "text",      "text": question},
            ]}],
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
        )
        return resp["choices"][0]["message"]["content"].strip()


# ═══════════════════════════════════════════════════════════════════════════
# Backend registry
# ═══════════════════════════════════════════════════════════════════════════

_BACKENDS: dict = {
    "moondream_text":   MoondreamTextBackend,
    "moondream_vision": MoondreamVisionBackend,
    "smolvlm2_2b":      SmolVLM2_2BBackend,
    "smolvlm2_500m":    SmolVLM2_500MBackend,
}


def _make_backend(name: str) -> _BackendBase:
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown MODEL_BACKEND '{name}'.\n"
            f"Valid options: {list(_BACKENDS.keys())}"
        )
    b = _BACKENDS[name]()
    b.load()
    return b


# ═══════════════════════════════════════════════════════════════════════════
# LLMNavigator — non-blocking threaded wrapper  (interface unchanged)
# ═══════════════════════════════════════════════════════════════════════════

class LLMNavigator:
    """
    Runs local GGUF inference in a daemon background thread.
    get_command() always returns instantly with the latest command.
    Drop-in replacement for the original Groq navigator.

    To switch models:
        MODEL_BACKEND=smolvlm2_2b python main.py
    or edit MODEL_BACKEND at the top of this file.
    """

    def __init__(self):
        self._step_count    = 0
        self._backoff_until = 0.0
        self._last_cmd = {
            "command":   "hover",
            "magnitude": 0.0,
            "reasoning": "Waiting for first model response.",
        }
        self._cmd_lock     = threading.Lock()
        self._pending      = False
        self._pending_lock = threading.Lock()

        self._backend = _make_backend(MODEL_BACKEND)
        print(f"[LLM] Ready — uses_vision={self._backend.USES_VISION}  "
              f"fires every {CALL_EVERY_N_STEPS} steps")

    # ── Public API ────────────────────────────────────────────────────────

    def get_command(self, frame: np.ndarray, state: dict) -> dict:
        """Returns latest command dict instantly; fires background inference periodically."""
        self._step_count += 1
        if (self._step_count % CALL_EVERY_N_STEPS == 0
                and time.time() >= self._backoff_until
                and not self._is_pending()):
            threading.Thread(
                target=self._bg_query,
                args=(frame.copy(), dict(state)),
                daemon=True,
            ).start()
        with self._cmd_lock:
            return dict(self._last_cmd)

    # ── Background thread ─────────────────────────────────────────────────

    def _bg_query(self, frame: np.ndarray, state: dict):
        with self._pending_lock:
            self._pending = True
        try:
            raw = self._backend.query(frame, state)
            print(f"[LLM] <- {raw}")
            cmd = self._parse_response(raw)
            with self._cmd_lock:
                self._last_cmd = cmd
        except Exception as exc:
            print(f"[LLM] Inference error: {str(exc)[:200]}")
            self._backoff_until = time.time() + 5.0
        finally:
            with self._pending_lock:
                self._pending = False

    def _is_pending(self) -> bool:
        with self._pending_lock:
            return self._pending

    # ── Parser — shared by all backends ──────────────────────────────────

    def _parse_response(self, raw: str) -> dict:
        """
        Parse:  command=move_forward magnitude=0.5 reason=heading to goal

        Lenient: case-insensitive, falls back to scanning for any valid
        command word, defaults to hover on complete failure.
        """
        text = raw.strip().lower()

        # command=
        m   = re.search(r"command\s*=\s*([a-z_]+)", text)
        cmd = m.group(1) if m else None
        if cmd not in VALID_COMMANDS:
            cmd   = next((v for v in VALID_COMMANDS if v in text), "hover")
            label = "rescued" if cmd != "hover" else "defaulted"
            print(f"[LLM] command {label} → {cmd}")

        # magnitude=
        m         = re.search(r"magnitude\s*=\s*([0-9]*\.?[0-9]+)", text)
        magnitude = float(np.clip(float(m.group(1)) if m else 0.5, 0.0, 1.0))

        # reason= (preserve original casing)
        m         = re.search(r"reason\s*=\s*(.+)", raw.strip(), re.IGNORECASE)
        reasoning = m.group(1).strip() if m else ""

        return {"command": cmd, "magnitude": magnitude, "reasoning": reasoning}


# ═══════════════════════════════════════════════════════════════════════════
# Standalone smoke-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    nav = LLMNavigator()

    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "TEST", (100, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 0), 3)

    dummy_state = {
        "position":              [0.5,  0.5,  1.0],
        "velocity":              [0.1,  0.0,  0.0],
        "orientation_rpy":       [0.0,  0.01, 0.5],
        "angular_velocity":      [0.0,  0.0,  0.0],
        "goal_position":         [5.0,  5.0,  1.0],
        "distance_to_goal":      6.36,
        "bearing_to_goal_rad":   0.785,
        "nearest_obstacle_dist": 0.8,
        "goal_reached":          False,
    }

    print("\n[Test] State string:")
    print(" ", build_state_str(dummy_state))

    nav._step_count = CALL_EVERY_N_STEPS - 1
    nav.get_command(dummy_frame, dummy_state)
    wait = 5 if MODEL_BACKEND == "moondream_text" else 15
    print(f"\n[Test] Inference running — waiting {wait}s ...")
    time.sleep(wait)
    print("\n[Test] Command:", nav.get_command(dummy_frame, dummy_state))

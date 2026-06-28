"""
data_loader.py  – Clean access to recorded Tello sessions
----------------------------------------------------------
Reads the session folder produced by tello_controller.py:

  data_sessions/session_YYYYMMDD_HHMMSS/
  ├── sensor_data.txt        ← tab-separated, one line per frame
  ├── rgb/000000.jpg  ...
  └── depth/000000.png  ...   (only for frames where depth ran)

Usage examples at the bottom of this file.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

SESSION_DIR = Path("data_sessions")


# ─────────────────────────────────────────────
#  PARSE  sensor_data.txt
# ─────────────────────────────────────────────
def _parse_line(line: str) -> dict | None:
    """
    Parse one tab-separated sensor line, e.g.:
      image_frame: 5  vx_cms: 2  vy_cms: 0  vz_cms: -1  yaw_dps: -2
      roll_deg: 0  pitch_deg: -5  ax: -29.0  ay: -18.0  az: -1006.0  time:5
    Returns a dict with int/float values, or None for blank/bad lines.
    """
    line = line.strip()
    if not line:
        return None
    out = {}
    for token in line.split("\t"):
        token = token.strip()
        if not token:
            continue
        # handle both "key: value" and "key:value"
        if ": " in token:
            k, v = token.split(": ", 1)
        elif ":" in token:
            k, v = token.split(":", 1)
        else:
            continue
        k = k.strip()
        v = v.strip()
        try:
            out[k] = int(v) if "." not in v else float(v)
        except ValueError:
            out[k] = v
    return out if out else None


# ─────────────────────────────────────────────
#  FRAME  dataclass
# ─────────────────────────────────────────────
@dataclass
class Frame:
    """One aligned sample: RGB image + depth map + sensor dict."""
    idx:       int
    elapsed_s: int                    # seconds since session start
    rgb:       np.ndarray             # (H, W, 3) uint8  RGB
    depth:     np.ndarray | None      # (H, W, 3) uint8 colourised, or None
    sensor:    dict

    # ── convenience properties ──────────────────────────────────────
    @property
    def bgr(self) -> np.ndarray:
        return cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)

    @property
    def vx(self)    -> int:   return self.sensor.get("vx_cms", 0)
    @property
    def vy(self)    -> int:   return self.sensor.get("vy_cms", 0)
    @property
    def vz(self)    -> int:   return self.sensor.get("vz_cms", 0)
    @property
    def yaw(self)   -> float: return self.sensor.get("yaw_dps", 0.0)
    @property
    def roll(self)  -> float: return self.sensor.get("roll_deg", 0.0)
    @property
    def pitch(self) -> float: return self.sensor.get("pitch_deg", 0.0)
    @property
    def accel(self) -> np.ndarray:
        return np.array([self.sensor.get("ax", 0),
                         self.sensor.get("ay", 0),
                         self.sensor.get("az", 0)], dtype=np.float32)
    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz], dtype=np.float32)
    @property
    def has_depth(self) -> bool:
        return self.depth is not None


# ─────────────────────────────────────────────
#  SESSION  loader
# ─────────────────────────────────────────────
class TelloSession:
    """
    Loads and iterates a recorded session.

        sess = TelloSession("session_20250526_143012")
        sess.summary()
        for frame in sess:
            use(frame.rgb, frame.depth, frame.vz)
    """

    def __init__(self, session_id: str, base_dir: Path = SESSION_DIR):
        self.root       = base_dir / session_id
        self.session_id = session_id

        if not self.root.exists():
            raise FileNotFoundError(f"Session not found: {self.root}")

        # Parse sensor_data.txt → list of sensor dicts, one per frame
        txt_path = self.root / "sensor_data.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"sensor_data.txt not found in {self.root}")

        self._rows: list[dict] = []
        with open(txt_path) as f:
            for line in f:
                row = _parse_line(line)
                if row is not None:
                    self._rows.append(row)

        self._cache: dict[int, Frame] = {}

    # ── metadata ────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._rows)

    @property
    def duration_s(self) -> int:
        if not self._rows:
            return 0
        return self._rows[-1].get("time", 0)

    @property
    def depth_coverage(self) -> float:
        """Fraction of frames that have a saved depth image."""
        depth_dir = self.root / "depth"
        n_depth   = len(list(depth_dir.glob("*.png")))
        return n_depth / len(self) if self._rows else 0.0

    def summary(self):
        print(f"Session    : {self.session_id}")
        print(f"Frames     : {len(self)}")
        print(f"Duration   : {self.duration_s}s")
        print(f"Depth maps : {self.depth_coverage*100:.0f}% of frames")

    # ── frame access ────────────────────────────────────────────────
    def __getitem__(self, idx: int) -> Frame:
        if idx in self._cache:
            return self._cache[idx]

        row        = self._rows[idx]
        frame_num  = row.get("image_frame", idx)
        frame_name = f"{frame_num:06d}"

        # RGB
        rgb_path = self.root / "rgb" / f"{frame_name}.jpg"
        bgr      = cv2.imread(str(rgb_path))
        if bgr is None:
            raise FileNotFoundError(f"RGB frame missing: {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Depth (optional – colourised PNG)
        depth      = None
        depth_path = self.root / "depth" / f"{frame_name}.png"
        if depth_path.exists():
            depth = cv2.imread(str(depth_path))   # BGR uint8 colourised

        frame = Frame(
            idx       = frame_num,
            elapsed_s = row.get("time", 0),
            rgb       = rgb,
            depth     = depth,
            sensor    = dict(row),
        )
        self._cache[idx] = frame
        return frame

    def __iter__(self) -> Iterator[Frame]:
        for i in range(len(self)):
            yield self[i]

    def depth_frames(self) -> Iterator[Frame]:
        """Yields only frames that have a depth image."""
        for frame in self:
            if frame.has_depth:
                yield frame

    # ── numpy export for ML ─────────────────────────────────────────
    def to_arrays(self, resize=(320, 240)) -> dict:
        """
        Returns dict of numpy arrays ready for ML training:
          rgb     (N, H, W, 3) uint8
          depth   (N, H, W, 3) uint8  colourised  (zeros where missing)
          imu     (N, 7)       float32 [vx,vy,vz, roll,pitch,yaw_dps, az]
          time    (N,)         int32   elapsed seconds
        """
        N    = len(self)
        H, W = resize[1], resize[0]
        rgb_arr   = np.zeros((N, H, W, 3), dtype=np.uint8)
        depth_arr = np.zeros((N, H, W, 3), dtype=np.uint8)
        imu_arr   = np.zeros((N, 7),        dtype=np.float32)
        time_arr  = np.zeros((N,),           dtype=np.int32)

        for i, frame in enumerate(self):
            rgb_arr[i]   = cv2.resize(frame.rgb, resize)
            if frame.depth is not None:
                depth_arr[i] = cv2.resize(frame.depth, resize)
            imu_arr[i]   = [frame.vx, frame.vy, frame.vz,
                             frame.roll, frame.pitch, frame.yaw,
                             frame.sensor.get("az", 0)]
            time_arr[i]  = frame.elapsed_s

        return {"rgb": rgb_arr, "depth": depth_arr, "imu": imu_arr, "time": time_arr}

    # ── export review video ─────────────────────────────────────────
    def export_video(self, out_path: str = "review.mp4",
                     show_depth: bool = True):
        fps    = max(1, self.duration_s and len(self) // self.duration_s or 15)
        frame0 = self[0]
        h, w   = frame0.rgb.shape[:2]
        out_w  = w * 2 if show_depth else w

        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (out_w, h),
        )
        for frame in self:
            bgr = frame.bgr
            if show_depth and frame.depth is not None:
                d = cv2.resize(frame.depth, (w, h))
                combined = np.hstack([bgr, d])
            elif show_depth:
                combined = np.hstack([bgr, np.zeros_like(bgr)])
            else:
                combined = bgr
            writer.write(combined)
        writer.release()
        print(f"[Export] Video saved → {out_path}")


# ─────────────────────────────────────────────
#  LIST ALL SESSIONS
# ─────────────────────────────────────────────
def list_sessions(base_dir: Path = SESSION_DIR) -> list[str]:
    if not base_dir.exists():
        return []
    return sorted([
        p.name for p in base_dir.iterdir()
        if p.is_dir() and (p / "sensor_data.txt").exists()
    ])


# ─────────────────────────────────────────────
#  USAGE  EXAMPLES
# ─────────────────────────────────────────────
if __name__ == "__main__":
    sessions = list_sessions()
    print("Sessions:", sessions)
    if not sessions:
        print("No sessions recorded yet."); exit()

    sess = TelloSession(sessions[-1])
    sess.summary()

    # Iterate every frame
    for frame in sess:
        print(f"  [{frame.idx:06d}]  "
              f"vx={frame.vx:+3d}  vy={frame.vy:+3d}  vz={frame.vz:+3d}  "
              f"roll={frame.roll:+5.1f}°  pitch={frame.pitch:+5.1f}°  "
              f"yaw_dps={frame.yaw:+5.1f}  "
              f"depth={'✓' if frame.has_depth else '✗'}  "
              f"t={frame.elapsed_s}s")

    # ML arrays
    arrays = sess.to_arrays(resize=(224, 224))
    print("rgb  :", arrays["rgb"].shape)
    print("depth:", arrays["depth"].shape)
    print("imu  :", arrays["imu"].shape)

    # Side-by-side RGB+depth review video
    # sess.export_video("review.mp4")

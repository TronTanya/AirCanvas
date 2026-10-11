"""One-time download of the MediaPipe Hand Landmarker model.

Official source (Google AI Edge, MediaPipe Hand Landmarker documentation):
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
Model file:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

After this script has run once, AirCanvas works fully offline.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request

from hand_tracker import MODEL_PATH, MODEL_URL

MIN_EXPECTED_BYTES = 1_000_000


def download(force: bool = False) -> int:
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size >= MIN_EXPECTED_BYTES and not force:
        print(f"Модель уже загружена: {MODEL_PATH}")
        return 0

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_PATH.with_suffix(".part")
    print(f"Загрузка модели:\n  {MODEL_URL}")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, tmp_path.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(64 * 1024):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / total:6.1%} ({done // 1024} КБ)", end="", flush=True)
        print()
    except (urllib.error.URLError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        print(f"\nОшибка загрузки: {exc}\nПроверьте подключение к интернету и повторите.", file=sys.stderr)
        return 1

    size = tmp_path.stat().st_size
    if size < MIN_EXPECTED_BYTES:
        tmp_path.unlink(missing_ok=True)
        print(f"Загруженный файл слишком мал ({size} байт), загрузка прервана.", file=sys.stderr)
        return 1

    tmp_path.replace(MODEL_PATH)
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    print(f"Готово: {MODEL_PATH} ({size / 1e6:.1f} МБ)\nSHA-256: {digest}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Загрузка модели MediaPipe Hand Landmarker")
    parser.add_argument("--force", action="store_true", help="загрузить заново, даже если файл есть")
    args = parser.parse_args()
    return download(force=args.force)


if __name__ == "__main__":
    sys.exit(main())

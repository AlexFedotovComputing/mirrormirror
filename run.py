"""Запуск MirrorLab кнопкой Run в IDE или командой python run.py."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Проверить готовность без открытия окна")
    args = parser.parse_args()

    missing = [name for name in ("numpy", "scipy") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"В выбранном Python отсутствуют: {', '.join(missing)}.\n"
            f'Установите зависимости: "{sys.executable}" -m pip install -r "{ROOT / "requirements.txt"}"'
        )
    package = ROOT / "node_modules" / "electron"
    path_file = package / "path.txt"
    if not path_file.is_file():
        raise RuntimeError("Electron не установлен. Выполните npm install в папке проекта.")
    electron = package / "dist" / path_file.read_text(encoding="utf-8").strip()
    if not electron.is_file():
        raise RuntimeError("Не найден исполняемый файл Electron. Выполните npm run setup:electron.")
    if not (ROOT / "node_modules/plotly.js-dist-min/plotly.min.js").is_file():
        raise RuntimeError("Plotly не установлен. Выполните npm install в папке проекта.")

    print(f"Python: {sys.executable}", flush=True)
    if args.check:
        print("MirrorLab готов к запуску.")
        return 0
    env = os.environ.copy()
    env.pop("ELECTRON_RUN_AS_NODE", None)
    env.pop("MIRRORLAB_HEADLESS", None)
    env["MIRRORLAB_PYTHON"] = sys.executable
    # Запускаем установленный Electron напрямую: Node.js в PATH не требуется.
    return subprocess.call([str(electron), str(ROOT)], cwd=ROOT, env=env)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as error:
        print(f"Не удалось запустить MirrorLab: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)

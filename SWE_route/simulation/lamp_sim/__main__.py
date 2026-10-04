"""Run with: uv run python -m lamp_sim."""

import argparse

from .gestures import load_gestures
from .motion import load_limits


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the character lamp playground")
    parser.add_argument("--port", type=int, default=8080, help="Local HTTP port (default: 8080)")
    parser.add_argument("--check-gestures", action="store_true", help="Validate gesture JSON files and exit")
    parser.add_argument("--backend", choices=("kinematic", "mujoco-kinematic", "mujoco", "mujoco-source"), default="mujoco")
    parser.add_argument("--native-viewer", action="store_true", help="Also open MuJoCo's native contact/force inspector")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.native_viewer and args.backend == "kinematic":
        parser.error("--native-viewer requires a MuJoCo backend")
    if args.check_gestures:
        try:
            gestures = load_gestures(load_limits())
        except ValueError as error:
            parser.error(str(error))
        print(f"Validated {len(gestures)} gestures: {', '.join(gestures)}")
        return
    from .viewer import run

    run(port=args.port, backend=args.backend, native_viewer=args.native_viewer)


if __name__ == "__main__":
    main()

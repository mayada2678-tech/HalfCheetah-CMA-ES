from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_gui_class():
    path = Path(__file__).with_name("halfcheetah_gui.py")
    spec = importlib.util.spec_from_file_location("halfcheetah_gui", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("GUI-Modul konnte nicht geladen werden.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.HalfCheetahUI


def main() -> None:
    try:
        _load_gui_class()().mainloop()
    except Exception as exc:
        message = (
            "HalfCheetah konnte nicht gestartet werden. "
            "Ursache oft: headless/Remote-Umgebung oder kein funktionierender OpenGL-/MuJoCo-Render-Kontext.\n"
            f"Details: {exc}"
        )
        print(message)
        raise


if __name__ == "__main__":
    main()

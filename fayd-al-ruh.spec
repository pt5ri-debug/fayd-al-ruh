# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the portable Fayd Al-Ruh build.

Build with::

    pyinstaller fayd-al-ruh.spec

The result is ``dist/fayd-al-ruh/`` (onedir): a folder that runs on a machine
without Python. The trained language-model checkpoint ships inside the bundle,
while writable state goes next to the executable (see ``core.paths``).
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

PROJECT_ROOT = Path(SPECPATH).resolve()

# Data the program reads at runtime and cannot regenerate on a fresh machine.
datas = [
    (str(PROJECT_ROOT / "ml" / "llm" / "checkpoints"), "ml/llm/checkpoints"),
]

# These modules are imported dynamically (importlib in cli/menu.py) or are
# __main__ modules, so the static analyzer would otherwise miss them. The `gui`
# package is imported lazily by the `gui` command, so it is listed explicitly to
# keep the graphical front end inside the portable bundle too.
hiddenimports = (
    [
        "main",
        "ml.__main__",
        "ml.llm.__main__",
        "evaluation.__main__",
        "gui.__main__",
        "cli.knowledge",
        "cli.ingest",
        "cli.deps",
        "cli.image",
    ]
    + collect_submodules("ml")
    + collect_submodules("evaluation")
    + collect_submodules("ai")
    + collect_submodules("hakeem")
    + collect_submodules("storage")
    + collect_submodules("gui")
    + collect_submodules("ingest")
)

a = Analysis(
    [str(PROJECT_ROOT / "scripts" / "portable_entry.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # ``torch``/``diffusers`` back the optional learned text-to-image extra. They
    # are far too large for the portable bundle and are imported lazily, so the
    # bundle omits them: the procedural renderer stays, and the diffusion backend
    # honestly reports itself unavailable inside a frozen build.
    excludes=["pytest", "mypy", "ruff", "PyInstaller", "torch", "diffusers", "transformers", "accelerate"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="fayd-al-ruh",
    debug=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="fayd-al-ruh",
)

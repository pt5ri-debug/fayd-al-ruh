# Portable Mode

The goal: run Fayd Al-Ruh on a machine **without Python and without any
intermediate software** — no CUDA, no FFmpeg, no external service — like
DeepFaceLab. At the same time, in-repo development stays exactly as it is, with
no changes.

## Why this is possible at all

The project is pure Python. The runtime dependencies are only `numpy` and
`requests`, and the language model is written from scratch in `numpy` with a
checkpoint committed in the repository. There is no heavy learning framework
that needs system libraries.

## Build

```bash
pip install -r requirements.txt
pip install pyinstaller          # build tool only
python scripts/build.py --clean  # output: dist/fayd-al-ruh/
```

The result is a single folder `dist/fayd-al-ruh/` containing:

```
fayd-al-ruh      (or fayd-al-ruh.exe on Windows)
run.sh / run.bat
_internal/       (Python + numpy + the bundled model)
```

Copy the folder to any machine and run it:

```bash
./run.sh            # or double-click run.bat on Windows
./fayd-al-ruh paths
```

> On Windows: double-clicking `run.bat` opens the menu, and when it finishes the
> window stays open so you can read any error message (`Press any key to close`).
> The launcher tries `py -3` first, then `python`, and if it finds no Python it
> explains that instead of closing immediately.

## Where assets are read and where data is written

`core/paths.py` separates two kinds of paths:

- **Assets (read-only):** in the portable build, inside the package
  (`sys._MEIPASS`), including the language-model checkpoint
  `ml/llm/checkpoints/fayd-llm.npz`.
- **Data (write):** beside the executable in a `data/` folder. This is what lets
  a copy to a USB stick keep its state (classifiers, policies, the retrained
  model).

The order in `default_checkpoint()`: a checkpoint in the data folder is preferred
(trained locally), otherwise it falls back to the one bundled in the package.

To point it manually:

```bash
FAYD_DATA_DIR=/path/to/data ./fayd-al-ruh paths
```

| Context | Assets | Data |
|---|---|---|
| From source | repository root | repository root |
| Portable | `_internal/` | beside the executable (`data/`) |

## The unified interface

`python -m cli` and `run.sh` call the same logic:

| Command | What it does |
|---|---|
| `python -m cli` | interactive numbered menu |
| `python -m cli demo` | local cognitive-cycle demo |
| `python -m cli ml --episodes 60` | machine-learning pipeline |
| `python -m cli llm --steps 4000` | train the in-project language model |
| `python -m cli eval` | the scientific evaluation suite |
| `python -m cli paths` | where assets are read and data is written |
| `python -m cli selfcheck` | verify the bundled model loads |

## Distribution via CI

`.github/workflows/ci.yml` has a `portable` job that builds the package on
Ubuntu/Windows/macOS on every change and runs a smoke check (`paths`, then
`demo`, then `selfcheck`). When a `v*` tag is pushed, the archives are uploaded
automatically to the GitHub release page.

## Notes

- When running **from source**, install the dependencies once:
  `python -m pip install -r requirements.txt`. If you forget, the launcher shows
  `Missing dependency: 'numpy'` with the install command instead of a long
  traceback.
- The portable package does not need that: the dependencies are bundled inside.
- PyInstaller is a **build tool** and is not added to `requirements.txt`, so it
  does not burden CI or the end user.
- A package built on one system runs only on that system. To build a `.exe`, use
  Windows or a CI job on `windows-latest`.
- `data/` is listed in `.gitignore`, so runtime data never enters the repository.

# Repository Guidelines

## Project Structure & Module Organization
This repository is a traditional stereo vision and 3D reconstruction project. Root-level Python scripts are the main entry points: `read_stereo.py` builds rectification maps, `SGBM.py` runs traditional stereo matching, `save_IGEV.py` post-processes one disparity output, and `batch_process_igev.py` handles batch point cloud and pointmap generation. C++ stereo utilities live in `stereo_matching.cpp`, `stereo_matching.h`, and `main.cc`; CMake support is provided by `CMakeLists.txt`. Camera calibration files are stored in `config/`. Physical datasets now live in the JieTai workspace-level `datasets/` directory; local `datasets/`, `rec_img_set/`, `LLM_FDJYP-3_out`, and `igev_output` paths are compatibility symlinks. Versioned generated results and reports live below the workspace-level `experiments/` directory. Metrics and conversion helpers are in `metric/` and `tools/`. Pytest-style validation scripts are named `test_*.py` at the repository root.

## Build, Test, and Development Commands
Install Python dependencies before running scripts:

```bash
pip install -r requirements.txt
```

Run common processing flows from the repository root:

```bash
python read_stereo.py          # generate or verify rectification maps
python SGBM.py                 # run SGBM disparity and point cloud output
python save_IGEV.py            # process one IGEV disparity result
python batch_process_igev.py   # batch process IGEV outputs
pytest                         # run Python tests
```

For C++ work, configure and build out of tree:

```bash
cmake -S . -B build
cmake --build build
```

## Coding Style & Naming Conventions
Use 4-space indentation for Python and keep script-level constants near the top of each file. Prefer clear snake_case for functions, variables, and file names. Keep calibration-specific paths and thresholds explicit and documented near the code that uses them. For C++, follow the existing C++11 style in the root files and keep OpenCV-dependent code isolated behind small functions where practical.

## Testing Guidelines
Tests use pytest naming conventions: `test_*.py` files and `test_*` functions. Add focused tests for metric calculations, pointmap parsing, disparity filtering, and regression-prone geometry logic. Run `pytest` before submitting changes; use smaller targeted runs such as `pytest test_pointmap.py` while iterating.

## Commit & Pull Request Guidelines
Git history is not available in this checkout, so use concise imperative commit messages such as `fix pointmap header parsing` or `add stereo metric regression test`. Pull requests should describe the affected pipeline stage, list commands run, note input datasets or calibration files used, and include screenshots or metric tables when visual or numeric output changes.

## Security & Configuration Tips
Do not commit large generated outputs, private datasets, or machine-specific absolute paths. Keep reusable camera parameters in `config/` and document any new calibration file name and intended camera setup.

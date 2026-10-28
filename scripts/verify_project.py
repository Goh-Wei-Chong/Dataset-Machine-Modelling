#!/usr/bin/env python3
"""Execute and verify the preparation and modelling notebooks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "prepare": "ML_Assignment_1.ipynb",
    "models": "ML_Assignment_2.ipynb",
}
MIN_FIGURES = {"prepare": 11, "models": 13}
RAW_SCHEMAS = {
    "hr_data.csv": [
        "employee_id", "department", "region", "education", "gender",
        "recruitment_channel", "no_of_trainings", "age", "previous_year_rating",
        "length_of_service", "KPIs_met >80%", "awards_won?", "avg_training_score",
        "is_promoted",
    ],
    "listings.csv": [
        "id", "name", "host_id", "host_name", "neighbourhood_group",
        "neighbourhood", "latitude", "longitude", "room_type", "price",
        "minimum_nights", "number_of_reviews", "last_review", "reviews_per_month",
        "calculated_host_listings_count", "availability_365",
    ],
}
PREPARED_SCHEMAS = {
    "hr_data_new.csv": [
        "department", "region", "education", "gender", "recruitment_channel",
        "no_of_trainings", "age", "previous_year_rating", "length_of_service",
        "KPIs_met >80%", "awards_won?", "avg_training_score", "is_promoted",
        "age_group", "good_conduct",
    ],
    "listings_new.csv": [
        "neighbourhood", "latitude", "longitude", "room_type", "price",
        "minimum_nights", "number_of_reviews", "last_review", "reviews_per_month",
        "calculated_host_listings_count", "availability_365",
    ],
}
EXPECTED_ROWS = {
    "hr_data.csv": 54808, "listings.csv": 7907,
    "hr_data_new.csv": 9336, "listings_new.csv": 6024,
}

# This runs after all source cells, in the same kernel. It never fits a model.
# The temporary cell is removed before either canonical notebook is saved.
MODEL_PROBE = r'''
import json as _verify_json
import sys as _verify_sys
import numpy as _verify_np
from sklearn.metrics import accuracy_score as _verify_accuracy, r2_score as _verify_r2

def _verify_require(condition, message):
    if not condition:
        raise ValueError(message)

_verify_result = {"classification": {}, "regression": {}, "splits": {},
                  "kernel_executable": _verify_sys.executable}
for _verify_prefix, _verify_rows, _verify_features in [
    ("1", 9336, 14), ("3", 9336, 13), ("5", 9336, 7),
    ("2", 6024, 10), ("4", 6024, 9),
]:
    _verify_test_rows = int(_verify_np.ceil(_verify_rows * 0.3))
    for _verify_part, _verify_n in [("train", _verify_rows - _verify_test_rows),
                                   ("test", _verify_test_rows)]:
        _verify_xname = "X" + _verify_prefix + "_" + _verify_part
        _verify_yname = "y" + _verify_prefix + "_" + _verify_part
        _verify_x = _verify_np.asarray(globals()[_verify_xname], dtype=float)
        _verify_y = _verify_np.asarray(globals()[_verify_yname], dtype=float)
        _verify_require(_verify_x.shape == (_verify_n, _verify_features),
                        "Unexpected shape for " + _verify_xname)
        _verify_require(_verify_y.shape == (_verify_n,),
                        "Unexpected shape for " + _verify_yname)
        _verify_require(_verify_np.isfinite(_verify_x).all() and _verify_np.isfinite(_verify_y).all(),
                        "Nonfinite split values: " + _verify_xname)
        _verify_result["splits"][_verify_xname] = list(_verify_x.shape)

for _verify_name, _verify_prefix in [
    ("log_final", "3"), ("dtc_final", "1"), ("mlpc_final", "1"),
    ("rfc_final", "5"), ("svc_final", "1"), ("xgc_final", "1"),
]:
    _verify_x = globals()["X" + _verify_prefix + "_test"]
    _verify_y = _verify_np.asarray(globals()["y" + _verify_prefix + "_test"])
    _verify_pred = _verify_np.asarray(globals()[_verify_name].predict(_verify_x))
    _verify_require(_verify_pred.shape == _verify_y.shape and _verify_np.isfinite(_verify_pred).all(),
                    "Invalid predictions from " + _verify_name)
    _verify_require(_verify_np.isin(_verify_y, [0, 1]).all()
                    and _verify_np.isin(_verify_pred, [0, 1]).all(),
                    "Invalid classification labels from " + _verify_name)
    _verify_score = float(_verify_accuracy(_verify_y, _verify_pred))
    _verify_require(_verify_np.isfinite(_verify_score) and 0 <= _verify_score <= 1,
                    "Invalid test accuracy from " + _verify_name)
    _verify_result["classification"][_verify_name] = {
        "accuracy": _verify_score, "test_rows": len(_verify_pred),
    }

for _verify_name, _verify_prefix in [
    ("lin_final", "4"), ("dtr_final", "2"), ("mlpr_final", "2"),
    ("rfr_final", "2"), ("svr_final", "2"), ("xgr_final", "2"),
]:
    _verify_x = globals()["X" + _verify_prefix + "_test"]
    _verify_y = _verify_np.asarray(globals()["y" + _verify_prefix + "_test"], dtype=float)
    _verify_pred = _verify_np.asarray(globals()[_verify_name].predict(_verify_x), dtype=float)
    _verify_require(_verify_pred.shape == _verify_y.shape and _verify_np.isfinite(_verify_pred).all(),
                    "Invalid predictions from " + _verify_name)
    _verify_rmse = float(_verify_np.sqrt(_verify_np.mean((_verify_y - _verify_pred) ** 2)))
    _verify_score = float(_verify_r2(_verify_y, _verify_pred))
    _verify_require(_verify_np.isfinite(_verify_rmse) and _verify_rmse >= 0
                    and _verify_np.isfinite(_verify_score),
                    "Invalid regression metrics from " + _verify_name)
    _verify_result["regression"][_verify_name] = {
        "rmse": _verify_rmse, "r2": _verify_score, "test_rows": len(_verify_pred),
    }
print("__PROJECT_VERIFICATION__" + _verify_json.dumps(_verify_result, allow_nan=False))
'''


class VerificationError(Exception):
    """A required project check failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def versions() -> dict[str, str]:
    result = {"python": sys.version.split()[0]}
    for package in (
        "nbclient", "nbformat", "ipykernel", "numpy", "pandas", "scipy",
        "scikit-learn", "matplotlib", "seaborn", "statsmodels", "xgboost",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    return result


def local_environment(run_dir: Path, args) -> None:
    # Set paths before importing Jupyter, IPython, or Matplotlib.
    for variable, directory in {
        "JUPYTER_CONFIG_DIR": "jupyter/config",
        "JUPYTER_DATA_DIR": "jupyter/data",
        "JUPYTER_RUNTIME_DIR": "jupyter/runtime",
        "IPYTHONDIR": "ipython", "MPLCONFIGDIR": "matplotlib",
        "XDG_CACHE_HOME": "cache", "PYTHONPYCACHEPREFIX": "pycache",
        "JOBLIB_TEMP_FOLDER": "joblib",
    }.items():
        path = run_dir / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    os.environ["MODEL_SEARCH_PROFILE"] = "extended" if args.extended_search else "standard"
    os.environ["MODEL_SEARCH_JOBS"] = str(args.jobs)


def read_table(name: str, columns: list[str], numeric: bool):
    import numpy as np
    import pandas as pd

    table = pd.read_csv(ROOT / name)
    require(table.columns.tolist() == columns, f"{name}: unexpected columns or column order")
    require(len(table) == EXPECTED_ROWS[name], f"{name}: expected {EXPECTED_ROWS[name]:,} rows, got {len(table):,}")
    if numeric:
        require(all(pd.api.types.is_numeric_dtype(dtype) for dtype in table.dtypes), f"{name}: nonnumeric columns")
        require(np.isfinite(table.to_numpy(dtype=float)).all(), f"{name}: missing or nonfinite values")
    if name == "hr_data_new.csv":
        require(table.is_promoted.value_counts().to_dict() == {0: 4668, 1: 4668}, "HR class balance changed")
    return table


def prepared_tables():
    return {name: read_table(name, columns, True) for name, columns in PREPARED_SCHEMAS.items()}


def compare_prepared(baseline, current) -> None:
    import pandas as pd

    for name in PREPARED_SCHEMAS:
        try:
            pd.testing.assert_frame_equal(
                baseline[name], current[name], check_dtype=False,
                check_exact=False, rtol=1e-12, atol=1e-12,
            )
        except AssertionError as exc:
            raise VerificationError(f"{name}: regenerated values, columns, or row order differ from the baseline") from exc


def table_summary(tables) -> dict:
    return {
        name: {"rows": len(table), "columns": table.columns.tolist(), "sha256": sha256(ROOT / name)}
        for name, table in tables.items()
    }


def execute_notebook(mode: str, notebook, args, run_dir: Path, result: dict) -> None:
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    started = time.monotonic()
    code_indices = [
        index for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code" and cell.source.strip()
    ]
    require(bool(code_indices), f"{NOTEBOOKS[mode]} contains no executable cells")
    ordinals = {index: number for number, index in enumerate(code_indices, start=1)}
    completed = set()
    starts = {}
    result.update(status="failed", code_cell_count=len(code_indices))
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)

    def on_cell_start(cell, cell_index):
        if cell_index in ordinals:
            starts[cell_index] = time.monotonic()
            print(f"{mode} [{ordinals[cell_index]}/{len(code_indices)}] Cell {cell_index + 1}: start", flush=True)

    def on_cell_executed(cell, cell_index, execute_reply):
        if cell_index in ordinals:
            elapsed = time.monotonic() - starts[cell_index]
            status = execute_reply.get("content", {}).get("status", "unknown")
            if status == "ok":
                completed.add(cell_index)
            print(f"{mode} [{ordinals[cell_index]}/{len(code_indices)}] Cell {cell_index + 1}: {status} ({elapsed:.1f}s)", flush=True)

    manager = KernelManager(
        kernel_name="python3", connection_file=str(run_dir / f"jupyter/runtime/{mode}-kernel.json"),
    )
    # Resolving a venv interpreter symlink would select the base Python instead.
    manager.kernel_spec.argv = [
        os.path.abspath(sys.executable), "-m", "ipykernel_launcher", "-f", "{connection_file}",
    ]
    client = NotebookClient(
        notebook, km=manager, timeout=args.timeout,
        allow_errors=False, force_raise_errors=True, skip_cells_with_tag="",
        record_timing=True, on_cell_start=on_cell_start, on_cell_executed=on_cell_executed,
    )
    try:
        print(f"Executing {NOTEBOOKS[mode]} with {sys.executable}", flush=True)
        client.execute(cwd=str(ROOT), env=os.environ.copy())
        require(completed == set(code_indices), f"{NOTEBOOKS[mode]}: not every nonblank code cell completed")
        require(
            all(notebook.cells[index].execution_count is not None for index in code_indices),
            f"{NOTEBOOKS[mode]}: a nonblank code cell has no execution count",
        )
        require(
            not any(output.get("output_type") == "error" for cell in notebook.cells for output in cell.get("outputs", [])),
            f"{NOTEBOOKS[mode]}: error outputs remain",
        )
        result["figure_count"] = sum(
            "image/png" in output.get("data", {})
            for cell in notebook.cells for output in cell.get("outputs", [])
        )
        require(result["figure_count"] >= MIN_FIGURES[mode], f"{NOTEBOOKS[mode]}: expected at least {MIN_FIGURES[mode]} figures, got {result['figure_count']}")
        if mode == "models":
            print("Verifying all 12 final models and their held-out split shapes", flush=True)
            probe = nbformat.v4.new_code_cell(MODEL_PROBE)
            notebook.cells.append(probe)
            try:
                client.execute_cell(probe, len(notebook.cells) - 1, store_history=False)
                text = "".join(output.get("text", "") for output in probe.get("outputs", []))
                prefix = "__PROJECT_VERIFICATION__"
                payload = next((line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)), None)
                require(payload is not None, "Model verification probe returned no result")
                result["models"] = json.loads(payload)
                require(result["models"]["kernel_executable"] == os.path.abspath(sys.executable), "Kernel used a different Python interpreter")
                require(len(result["models"]["classification"]) == 6 and len(result["models"]["regression"]) == 6, "Expected six classification and six regression results")
            finally:
                notebook.cells.pop()
        result["status"] = "passed"
    finally:
        result["executed_code_cell_count"] = len(completed)
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)
        try:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        finally:
            if client.kc is not None:
                client.kc.stop_channels()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=["all", "prepare", "models"], default="all", help="notebooks to execute (default: all, preparation first)")
    parser.add_argument("--timeout", type=positive_int, default=1800, help="maximum seconds per cell (default: 1800)")
    parser.add_argument("--extended-search", action="store_true", help="use the larger model hyperparameter search grids")
    parser.add_argument("--jobs", type=positive_int, default=1, help="parallel model-search workers (default: 1)")
    args = parser.parse_args()
    modes = list(NOTEBOOKS) if args.notebook == "all" else [args.notebook]
    for name in [*RAW_SCHEMAS, *PREPARED_SCHEMAS, *(NOTEBOOKS[mode] for mode in modes)]:
        if not (ROOT / name).is_file():
            print(f"Verification failed: required file is missing: {ROOT / name}", file=sys.stderr)
            return 1

    run_dir = ROOT / ".verification"
    local_environment(run_dir, args)
    started = time.monotonic()
    raw_hashes = {name: sha256(ROOT / name) for name in RAW_SCHEMAS}
    source_hashes = {name: sha256(ROOT / name) for name in (NOTEBOOKS[mode] for mode in modes)}
    report = {
        "status": "failed", "mode": args.notebook,
        "search_profile": os.environ["MODEL_SEARCH_PROFILE"], "jobs": args.jobs,
        "versions": versions(), "raw_sha256": raw_hashes, "notebooks": {},
    }
    notebooks = {}
    exit_code = 1
    try:
        import nbformat

        raw_tables = {name: read_table(name, columns, False) for name, columns in RAW_SCHEMAS.items()}
        report["raw_tables"] = table_summary(raw_tables)
        baseline = prepared_tables()
        report["prepared_baseline"] = table_summary(baseline)
        for mode in modes:
            name = NOTEBOOKS[mode]
            notebook = nbformat.read(ROOT / name, as_version=4)
            notebooks[name] = notebook
            result = report["notebooks"][name] = {}
            execute_notebook(mode, notebook, args, run_dir, result)
            compare_prepared(baseline, prepared_tables())
            require(all(sha256(ROOT / name) == digest for name, digest in raw_hashes.items()), "A raw CSV input changed during execution")
        report["prepared_tables"] = table_summary(prepared_tables())
        require(all(sha256(ROOT / name) == digest for name, digest in source_hashes.items()), "A notebook was edited during verification; fresh outputs were not saved")
        # Stage every result before replacing the selected canonical notebooks.
        for name, notebook in notebooks.items():
            nbformat.write(notebook, run_dir / name)
        for name in notebooks:
            os.replace(run_dir / name, ROOT / name)
        report["status"] = "passed"
        exit_code = 0
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"Verification failed: {report['error']}", file=sys.stderr, flush=True)
        if notebooks:
            import nbformat
            for name, notebook in notebooks.items():
                nbformat.write(notebook, run_dir / ("failed-" + name))
        if isinstance(exc, KeyboardInterrupt):
            exit_code = 130
    finally:
        report["raw_inputs_unchanged"] = all(
            (ROOT / name).is_file() and sha256(ROOT / name) == digest
            for name, digest in raw_hashes.items()
        )
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        (run_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if exit_code == 0:
        print(f"Verification passed in {report['elapsed_seconds']:.1f}s. Saved fresh notebook outputs and .verification/report.json", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

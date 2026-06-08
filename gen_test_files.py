
"""
Generate deterministic test fixtures, organized by module.
Examples:
    python tests/gen_test_files.py
    python tests/gen_test_files.py --module io
    python tests/gen_test_files.py --output-dir tests/data
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Callable, Dict
import numpy as np


def _write_csv(path: Path, rows: list[list[object]], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write(delimiter.join("" if v is None else str(v) for v in row) + "\n")


def _gen_io_test_data(output_dir: Path) -> None:
    """Generate fixtures for numcompute.io tests."""
    output_dir = output_dir / "io"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Standard comma-separated dataset with one header row and missing values.
    _write_csv(
        output_dir / "io_basic_missing.csv",
        [
            ["c1", "c2", "c3"],
            [1, 2, 3],
            [4, None, 6],
            [7, 8, 9],
            [None, 11, 12],
        ],
        delimiter=",",
    )

    # Semicolon-separated dataset for delimiter tests.
    _write_csv(
        output_dir / "io_semicolon_missing.csv",
        [
            ["a", "b", "c"],
            [10, 20, 30],
            [40, None, 60],
            [70, 80, 90],
        ],
        delimiter=";",
    )

    # Dataset with extra metadata line + header to validate skip_rows.
    _write_csv(
        output_dir / "io_two_header_rows.csv",
        [
            ["metadata", "for", "tests"],
            ["x", "y", "z"],
            [1, 1, 1],
            [2, None, 2],
            [3, 3, 3],
        ],
        delimiter=",",
    )

    # Dataset with no missing values as a control sample.
    _write_csv(
        output_dir / "io_no_missing.csv",
        [
            ["p", "q", "r"],
            [0.1, 1.1, 2.1],
            [3.1, 4.1, 5.1],
            [6.1, 7.1, 8.1],
        ],
        delimiter=",",
    )

def _gen_sort_search_test_data(output_dir: Path) -> None:
    """Generate fixtures for numcompute.sort_search tests."""
    output_dir = output_dir / "sort_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        output_dir / "sort_values_1d.npy",
        np.array([5, 1, 3, 3, 2, 9, 2], dtype=float),
    )
    np.save(
        output_dir / "sort_values_2d.npy",
        np.array(
            [
                [3, 2, 1],
                [1, 2, 3],
                [2, 1, 3],
                [2, 2, 2],
            ],
            dtype=float,
        ),
    )

    np.save(
        output_dir / "multi_key_data.npy",
        np.array(
            [
                [2, 1, 10],
                [1, 2, 30],
                [2, 1, 5],
                [1, 1, 20],
                [2, 2, 0],
            ],
            dtype=float,
        ),
    )

    np.save(
        output_dir / "topk_values.npy",
        np.array([4, 9, 1, 7, 3, 8, 6, 2, 5], dtype=float),
    )

    np.save(
        output_dir / "search_sorted.npy",
        np.array([1, 2, 2, 4, 7, 9], dtype=float),
    )


def _gen_rank_test_data(output_dir: Path) -> None:
    """Generate fixtures for numcompute.rank tests."""
    output_dir = output_dir / "rank"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        output_dir / "rank_values.npy",
        np.array([100, 50, 50, 20, 20, 20], dtype=float),
    )

    np.save(
        output_dir / "percentile_values.npy",
        np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float),
    )


def _gen_metrics_test_data(output_dir: Path) -> None:
    """Generate fixtures for numcompute.metrics tests."""
    output_dir = output_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        output_dir / "cls_y_true.npy",
        np.array([1, 0, 1, 1, 0, 0, 1], dtype=int),
    )
    np.save(
        output_dir / "cls_y_pred.npy",
        np.array([1, 0, 0, 1, 0, 1, 1], dtype=int),
    )

    np.save(
        output_dir / "multi_y_true.npy",
        np.array([0, 1, 2, 2, 1, 0], dtype=int),
    )
    np.save(
        output_dir / "multi_y_pred.npy",
        np.array([0, 2, 2, 1, 1, 0], dtype=int),
    )

    np.save(
        output_dir / "reg_y_true.npy",
        np.array([3.0, -0.5, 2.0, 7.0], dtype=float),
    )
    np.save(
        output_dir / "reg_y_pred.npy",
        np.array([2.5, 0.0, 2.0, 8.0], dtype=float),
    )

    np.save(
        output_dir / "roc_y_true.npy",
        np.array([0, 0, 1, 1], dtype=int),
    )
    np.save(
        output_dir / "roc_y_score.npy",
        np.array([0.1, 0.4, 0.35, 0.8], dtype=float),
    )


def _gen_utils_test_data(output_dir: Path) -> None:
    """Generate fixtures for numcompute.utils tests."""
    output_dir = output_dir / "utils"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        output_dir / "utils_X.npy",
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
    )

    np.save(
        output_dir / "utils_Y.npy",
        np.array(
            [
                [1.0, 1.0],
                [2.0, 0.0],
            ],
            dtype=float,
        ),
    )
def _gen_preprocessing_test_data(output_dir: Path) -> None:
    """Generate fixtures for numcompute.preprocessing tests."""
    output_dir = output_dir / "preprocessing"
    output_dir.mkdir(parents=True, exist_ok=True)

    
    np.save(
        output_dir / "preprocessing_normal.npy",
        np.array([[2.0, 4.0],
                  [6.0, 8.0],
                  [10.0, 12.0]], dtype=float)
    )

    
    np.save(
        output_dir / "preprocessing_equal.npy",
        np.array([[5.0, 5.0],
                  [5.0, 5.0],
                  [5.0, 5.0]], dtype=float)
    )

    
    np.save(
        output_dir / "preprocessing_nan.npy",
        np.array([[1.0, 2.0],
                  [np.nan, 4.0],
                  [5.0, np.nan]], dtype=float)
    )

   
    np.save(
        output_dir / "pipeline_nan.npy",
        np.array([[1.0, 2.0],
                  [np.nan, 4.0],
                  [5.0, 6.0]], dtype=float)
    )

    
    np.save(
        output_dir / "preprocessing_categories.npy",
        np.array(["cat", "dog", "bird", "cat"])
    )

def _gen_pipeline_test_data(output_dir: Path) -> None:
    """Generate fixtures for numcompute.pipeline tests."""
    output_dir = output_dir / "pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    
    np.save(
        output_dir / "pipeline_normal.npy",
        np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]], dtype=float)
    )

   
    np.save(
        output_dir / "pipeline_nan.npy",
        np.array([[1.0,    2.0],
                  [np.nan, 4.0],
                  [5.0,    6.0]], dtype=float)
    )

def _gen_stats_test_data(output_dir: Path) -> None:
    """Generate fixtures for numcompute.stats tests."""
    output_dir = output_dir / "stats"
    output_dir.mkdir(parents=True, exist_ok=True)

    
    np.save(
        output_dir / "stats_normal.npy",
        np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    )

    
    np.save(
        output_dir / "stats_nan.npy",
        np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=float)
    )

    
    np.save(
        output_dir / "stats_histogram.npy",
        np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=float)
    )

GENERATORS: Dict[str, Callable[[Path], None]] = {
    "io": _gen_io_test_data,
    "sort_search": _gen_sort_search_test_data,
    "rank": _gen_rank_test_data,
    "metrics": _gen_metrics_test_data,
    "utils": _gen_utils_test_data,
    "preprocessing": _gen_preprocessing_test_data,
    "pipeline": _gen_pipeline_test_data,
    "stats": _gen_stats_test_data,
}


def generate(output_dir: Path, module: str | None = None) -> None:
    if module is not None:
        try:
            GENERATORS[module](output_dir)
            print(f'Generating {module} - SUCCESS')
        except KeyError as exc:
            supported = ", ".join(sorted(GENERATORS))
            raise ValueError(f"Unknown module '{module}'. Supported: {supported}") from exc
        return

    print('Generating:')
    for gen in GENERATORS.values():
        try:
            gen(output_dir)
            print(f"  {gen.__name__[5:]} - SUCCESS")
        except Exception as exc:
            print(f"  {gen.__name__[5:]} - FAILED: {exc}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate test fixture data by module.")
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        help="Generate only one module's fixtures (e.g., io). Default: generate all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated fixtures. Default: tests/data",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    tests_dir = Path(__file__).resolve().parent
    data_dir = args.output_dir if args.output_dir is not None else tests_dir / "data"
    generate(data_dir, module=args.module)


if __name__ == "__main__":
    main()

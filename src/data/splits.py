from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data.path_utils import collect_json_paths
from typing import Iterable, Union

PathLike = Union[str, Path]

def split_paths(paths, val_size=0.2, seed=42):
    if len(paths) < 2:
        # too small to split safely
        return paths, []
    return train_test_split(paths, test_size=val_size, random_state=seed)

def write_paths_txt(paths: Iterable[PathLike], out_txt: PathLike) -> None:
    out_txt = Path(out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(p) for p in paths) + "\n"   # final newline is nice
    out_txt.write_text(text, encoding="utf-8")

def write_split_files(
    train_paths: Iterable[PathLike],
    val_paths: Iterable[PathLike],
    out_dir: PathLike,
    train_name: str = "train_paths.txt",
    val_name: str = "val_paths.txt",
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_txt = out_dir / train_name
    val_txt   = out_dir / val_name

    write_paths_txt(train_paths, train_txt)
    write_paths_txt(val_paths, val_txt)
    return train_txt, val_txt

import shutil, re, random
from pathlib import Path
import json


from colorama import Fore, Style
# B, U, R = '\033[1m', '\033[4m', '\033[0m'
# RED, GREEN, BLUE = Fore.RED, Fore.GREEN, Fore.BLUE
# RESET = Style.RESET_ALL
def print_color(msg:str, clr=Fore.RED):
    if   clr in ['RED',' Red', 'red', 'r']      :  clr = Fore.RED
    elif clr in ['BLUE', 'Blue', 'blue', 'b']   :  clr = Fore.BLUE
    elif clr in ['YELLOW', 'yellow', 'y']       :  clr = Fore.YELLOW
    elif clr in ['BLUE', 'Blue', 'blue', 'r']   :  clr = Fore.BLUE
    elif clr in ['GREEN', 'Green', 'green', 'g']:  clr = Fore.GREEN
    print( f"{clr}{msg}{Style.RESET_ALL}")

# ----------------------------------------------------------------------------
# Files and Paths Utils
# -----------------------------------------------------------------------------

def _make_unique_dir(root, base_name, **kwargs):
    """  Create a unique subdir under root for base_name, adding (2), (3), ... if needed.
    :param root : str|Path,  Parent directory.
    :param base_name : str, Desired subdirectory name.
    no_space : bool, optional (via kwargs)  If True, use base_name-(2) style; otherwise base_name (2).
    """
    no_space = kwargs.get("no_space", False)
    sep = "" if no_space else " "

    root = Path(root)
    clip_name = base_name
    i = 2
    clip_dir = root / clip_name

    while clip_dir.exists():
        clip_name = f"{base_name}{sep}({i})"
        clip_dir = root / clip_name
        i += 1

    clip_dir.mkdir(parents=True, exist_ok=True)
    return clip_dir, clip_name


def clear_dir(path, missing_ok:bool = False) -> None:
    """  Delete all files and subdirectories inside `path`, but keep `path` itself.
    Any error (nonexistent path, not a directory, permissions, etc.)
    is printed and swallowed, so it won't stop the program.
    ----------
    :param path : str|Path  Directory whose contents will be removed.
    :param missing_ok : bool, default False
            If False, raise FileNotFoundError when path doesn't exist.
            If True, silently do nothing when path doesn't exist.
    """
    try:
        p = Path(path)
    except Exception as e:
        print(f"clear_dir: invalid path {path!r}: {e}")
        return

    try:
        if not p.exists():
            print(f"clear_dir: path does not exist: {p}")
            return

        if not p.is_dir():
            print(f"clear_dir: not a directory: {p}")
            return
    except Exception as e:
        print(f"clear_dir: failed to inspect path {p!r}: {e}")
        return

    for child in p.iterdir():
        try:
            # Don't follow symlinks, just remove the link itself
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                # Weird/unknown type – try to unlink anyway
                child.unlink(missing_ok=True)
        except Exception as e:
            print(f"clear_dir: failed to remove {child!r}: {e}")


def correct_path(path:str|Path, project_root: str|Path|None=None):
    """  Try to resolve a path defined with relative prefixes (e.g. ../../a/b/c)
    by matching only its *tail* (a/b/c) somewhere under the project root.
    Rules:
    - Ignores leading ../ components
    - Searches recursively under project_root
    - If exactly one match is found → return corrected Path
    - If zero or multiple matches → print one-line warning and return None
    """
    if path is None:
        return None

    p = Path(path)
    tail = Path(*[x for x in p.parts if x not in ('..', '.')])

    root = Path(project_root) if project_root is not None else Path.cwd()
    matches = list(root.rglob(tail.as_posix()))

    if len(matches) == 1:
        # return matches[0]
        return str(matches[0]) if isinstance(path,str) else matches[0]
    elif len(matches) == 0:
        print(f"[correct_path] NOT FOUND: {tail}")
    else:
        print(f"[correct_path] AMBIGUOUS ({len(matches)} matches): {tail}")
    return None


def get_unique_name(file_name:str|Path, n:int=3) -> Path:
    """ Return a unique file name.
    Rules:  If file does not exist → return as is.
    If exists:  my_file.txt      -> my_file_001.txt  (padding = n)
                my_file_01.txt   -> my_file_02.txt   (padding preserved = 2)
                my_file_45.txt   -> my_file_46.txt   (padding preserved = 2)
    """

    file_path = Path(file_name)
    if not file_path.exists():
        return file_path

    parent, stem, suffix = file_path.parent, file_path.stem, file_path.suffix

    match = re.search(r'_(\d+)$', stem)

    if match:
        base = stem[:match.start()]
        num_str = match.group(1)
        counter = int(num_str) + 1
        padding = len(num_str)  # preserve existing padding
    else:
        base = stem
        counter = 1
        padding = n  # use default padding

    while True:
        new_name = f"{base}_{counter:0{padding}d}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1


# ***** json handling ***** #x`1
def load_json_frames(json_path: str | Path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # frames =
    return data['frames'] # frames


def json_equal(f1, f2, keys=None):
# def json_equal(f1, f2, ignore_header=False):
    with open(f1) as a, open(f2) as b:
        # if ignore_header:
        #     return json.load(a).get('frames') == json.load(b).get('frames')
        if keys:
            j1, j2 = json.load(a), json.load(b)
            found = [j1.get(k) == j2.get(k)  for k in collection(keys)]
            return  all (found)
            return json.load(a).get('frames') == json.load(b).get('frames')
        return json.load(a) == json.load(b)

def compare_json_dirs(d1, d2, soft_compare=False, **kwargs):
    """ Compare two JSON directories;
        in soft mode only common filenames are checked."""
    d1, d2 = Path(d1),  Path(d2)

    if not d1.is_dir(): return False, f"can't find 1st path: {d1.name}"
    if not d2.is_dir(): return False, f"can't find 2nd path: {d2.name}"

    files1 = {p.name: p for p in d1.rglob('*.json')}
    files2 = {p.name: p for p in d2.rglob('*.json')}

    if not soft_compare:
        #* Strict mode: file sets must match exactly
        if set(files1.keys()) != set(files2.keys()):
            return False, 'Different file sets'
        compare_list = files1.keys()
    else:
        #* Soft mode: compare only common names
        compare_list = set(files1.keys()) & set(files2.keys())

    if not compare_list:
        return False, 'No common JSON files to compare'

    unequal_count =0
    for name in compare_list:
        print(f"Comparing {name} ")
        if  json_equal(files1[name], files2[name]):
            print_color(f"\r - equal ", 'b')
        else:
            print_color(f"\r - differ")
            unequal_count += 1
            if kwargs.get('break',False):
                return False, f'Mismatch in {name}'
    if unequal_count > 0:
        return False, f" {len(compare_list) - unequal_count} out of {len(compare_list)} are equal"

    return True, "Directories identical (based on comparison mode)" #24


def compare_json_samples(l1, l2, smp, **kwargs):
    """Randomly compare JSON samples between two lists/dirs and print match statistics."""

    list1 = list(Path(l1).rglob('*.json')) if isinstance(l1, (str, Path)) else l1
    list2 = list(Path(l2).rglob('*.json')) if isinstance(l2, (str, Path)) else l2
    # print_color(list1);
    print_color(list2, 'b')

    if not list1:
        print('No files in first input')
        return

    # Determine number of samples
    if isinstance(smp, float) and (0 <= smp <= 1):
        n_draw = max(1, int(len(list1) * smp)) if smp > 0 else 0
    elif isinstance(smp, int) and 0 < smp :
        n_draw = min(int(smp), len(list1))
    else:
        raise ValueError('smp must a float between 0 and 1 or positive int ')

    sampled = random.sample(list1, n_draw) if n_draw > 0 else []
    print_color(sampled);
    # Map list2 by filename
    map2 = {p.name: p for p in list2}

    if kwargs.get('print_list', False):
        print("\t File: \t\t |\tFound\t|\tEqual ")
    found,identical = 0, 0
    for p1 in sampled:
        p2 = map2.get(Path(p1).name)
        if p2:
            found += 1
            eql = json_equal(p1, p2)
            if eql:
                identical += 1
        if kwargs.get('print_list', False):
            print(f"{Path(p1).name:20}- {'Yes' if p2 else '---'}\t{eql} ")

    percent = 100*(identical/found) if found > 0 else 0
    print(f"\nComparison result:\n {n_draw} files drawn\n"
          f" {found} found in second set\n  {identical} identical ({percent:.1f}%)")


# ***** pth handling ***** #
#*86->63
def get_epoch_pth(dir_path:str|Path, epoch:int|str|None='best') -> str|None:
    """ Return path to desired checkpoint (pth file) in dir_path.
    :param epoch:- int : desired epoch; if exact file not found, pick the closest
                         *later* epoch_XX.pth, or the last available if none later.
                 - 'best' (default): return a 'best' checkpoint if present
                         (file name containing 'best'). If none, fall back to last
                       epoch_XX.pth and print a warning.
                 - 'last': return the last epoch_XX.pth (highest epoch number).
    """
    #* Helper: extract epoch number from names like 'epoch_25.pth'
    def _epoch_of(name: str) -> int | None:
        if name.startswith('epoch_') and name.endswith('.pth'):
            num_part = name[len('epoch_'):-len('.pth')]
            try:
                return int(num_part)
            except ValueError:
                return None

    def all_pairs():
        epc_pairs = [(p.name, _epoch_of(p.name)) for p in pth_files]
        return [(p, e) for p, e in epc_pairs if e is not None]

    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise FileNotFoundError(f'Checkpoint directory not found: {dir_path}')

    pth_files = sorted(dir_path.glob('*.pth'))
    if not pth_files:
        # raise FileNotFoundError(f'No .pth files found in {dir_path}')
        print(f"[Error] No pth files found in {dir_path}")
        return None

    #* --- 'best' case ---
    if (isinstance(epoch, str) and epoch == 'best') or epoch is None:
        pth = [p for p in pth_files if 'best' in p.name]
        if pth:  # If several, pick the newest by mtime
            best_pth = max(pth, key=lambda p: p.stat().st_mtime)
            return str(best_pth.name)
        print("[WARN] No best checkpoint found, falling back to last epoch.")
        epoch = 'last'  # fallthrough

    #* --- 'last' case ---
    if isinstance(epoch, str) and epoch == 'last':
        epoch_pairs = all_pairs()
        if not epoch_pairs:
            # Only weird filenames – fall back to newest
            latest = max(pth_files, key=lambda p: p.stat().st_mtime)
            return str(latest)
        last_pth, _ = max(epoch_pairs, key=lambda pe: pe[1])
        return last_pth

    #* --- numeric (or numeric-string) epoch request ---
    if not isinstance(epoch, int):
        print(f"Unsupported epoch spec: {epoch!r}")

    #* Exact match first
    pth = f'epoch_{epoch}.pth'
    if (dir_path/pth).is_file():
        return pth
    #* Otherwise, search for closest later epoch_XX.pth
    epoch_pairs = all_pairs()
    closest_pth, _ = min(epoch_pairs,key=lambda pe: (abs(pe[1] - epoch), -pe[1]), )
    return str(closest_pth)


# ***** Basic Log handling ***** #
def _save_log(lines, log_name, log_type: str | None = None):
    """Save log lines to a file; log_type reserved for future formatting tweaks."""
    if not lines:   return

    # Default: plain text, one line per entry
    if log_type is None or log_type == "default":
        log_path = Path(log_name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as lf:
            for line in lines:
                lf.write(line + "\n")
    else:
        # Placeholder for future formats
        log_path = Path(log_name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as lf:
            for line in lines:
                lf.write(line + "\n")

def _load_log_lines(log_source):
    """Normalize log source (path or list) into list of stripped lines."""
    if isinstance(log_source, (str, Path)):
        lp = Path(log_source)
        if not lp.is_file():
            return []
        with lp.open("r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    elif isinstance(log_source, list):
        return [str(ln).strip() for ln in log_source if str(ln).strip()]
    return []


# ***** Collection casting ***** #
def as_collection(x):
    """  If x is a collection (list/tuple/set/dict/range/numpy array/torch tensor/etc.)
    return it as-is. Otherwise, wrap it in a single-element list.
    (*) Strings/bytes are treated as scalars (wrapped).
    (*) Multi-dim numpy / torch arrays are returned as-is (no special handling).
    """
    from collections.abc import Iterable

    if isinstance(x, (str, bytes, bytearray)):
        return [x]    #* treat strings/bytes as scalars, not collections

    #* optional numpy/torch support without hard dependency
    np_types = ()
    torch_types = ()
    try:
        import numpy as np
        np_types = (np.ndarray,)
    except Exception:
        pass
    try:
        import torch
        torch_types = (torch.Tensor,)
    except Exception:
        pass

    collection_types = (list, tuple, set, frozenset, range, dict) + np_types + torch_types

    #* common concrete collection types
    if isinstance(x, collection_types):
        return x

    #* other iterables (e.g. generators); treat as collections and return as-is
    if isinstance(x, Iterable):
        return x

    #* scalar fallback
    return [x]

collection = as_collection


# ----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
def tst_get_pth(tst_path:str=None, **kwargs):

    tst_ls = kwargs.get('tst_ls', [None, 'best', 'last', 10, 25, 17, 51, 81, -1])
    d = Path(tst_path)
    for t in tst_ls:
        print(get_epoch_pth(d, t))
    d = d.parent
    print(get_epoch_pth(d)),  print(get_epoch_pth(d, epoch=10))


if __name__ == "__main__":
    pass
    # tst_get_pth("/mnt/local-data/Python/Projects/weSmart/work_dirs/tsm_r50_bbfrm")
    f1 = Path("/home/ejwolf/clouds/erez.wolfson/wesmart-share/data/json/RWF-2000/train_pos/cy1gi3ZJb_c_4.json")
    f2 = Path("/home/ejwolf/clouds/erez.wolfson/wesmart-share/data/json/test/pos_004.json")
    print(f2.parent.is_dir())
    print(json_equal(f1, f2))
    print(json_equal(f1, f2, keys=['frames','fps', 'step']))

#277(2,3,2)-> 260

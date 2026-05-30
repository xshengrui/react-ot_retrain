import argparse
import csv
import logging
import pathlib
import pickle
import re
import sys
import tarfile
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm


"""
Example:
python infer_gdb_raw_xyz.py \
  --checkpoint checkpoint/RPSB-FT-Schedule/leftnet-ts_guess_NEBCI-xtb-ema-3739d3534f81/sb-epoch=199-val_ep_scaled_err=0.0441.ckpt \
  --checkpoint rpsb_ts1x_mix_dim-0.ckpt \
  --tar ./reactot/data/GDB-10-rxn_raw.tar.gz \
  --tar ./reactot/data/GDB-17-rxn_raw.tar.gz \
  --output-root ./outputs/gdb_raw_inference \
  --solver ode \
  --method midpoint \
  --batch-size 72 \
  --nfe 100
"""


ATOM_TO_Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
}
Z_TO_ATOM = {val: key for key, val in ATOM_TO_Z.items()}
FRAGMENTS = ("reactant", "transition_state", "product")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_logger(log_dir: pathlib.Path) -> None:
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = open(log_dir / "log.txt", "a", encoding="utf-8")
    file_console = Console(file=log_file, width=150)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        force=True,
        handlers=[RichHandler(), RichHandler(console=file_console)],
    )


def move_to_device(obj, target_device):
    if torch.is_tensor(obj):
        return obj.to(target_device, non_blocking=True)
    if isinstance(obj, Mapping):
        return type(obj)((key, move_to_device(val, target_device)) for key, val in obj.items())
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(move_to_device(val, target_device) for val in obj))
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return type(obj)(move_to_device(val, target_device) for val in obj)
    if hasattr(obj, "to"):
        return obj.to(target_device)
    return obj


def parse_xyz_text(text: str):
    lines = [line.strip() for line in text.splitlines()]
    if not lines:
        raise ValueError("empty xyz file")
    natoms = int(lines[0])
    atom_lines = lines[2 : 2 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(f"expected {natoms} atom lines, got {len(atom_lines)}")

    symbols = []
    coords = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"invalid xyz atom line: {line}")
        symbol = parts[0]
        if symbol not in ATOM_TO_Z:
            raise ValueError(f"unsupported atom {symbol}; expected one of {sorted(ATOM_TO_Z)}")
        symbols.append(symbol)
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.asarray(coords, dtype=np.float32)


def kabsch_transform(source: np.ndarray, target: np.ndarray):
    source_center = source.mean(axis=0, keepdims=True)
    target_center = target.mean(axis=0, keepdims=True)
    source0 = source - source_center
    target0 = target - target_center
    cov = source0.T @ target0
    u, _, vt = np.linalg.svd(cov)
    det = np.linalg.det(vt.T @ u.T)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = np.sign(det) if det != 0 else 1.0
    rotation = vt.T @ correction @ u.T
    return source_center, target_center, rotation.astype(np.float32)


def apply_transform(coords: np.ndarray, source_center, target_center, rotation):
    return ((coords - source_center) @ rotation.T + target_center).astype(np.float32)


def prepare_coordinates(r_pos, p_pos, ts_pos, align_mode: str):
    r_centered = (r_pos - r_pos.mean(axis=0, keepdims=True)).astype(np.float32)
    if align_mode == "none":
        return (
            r_centered,
            (p_pos - p_pos.mean(axis=0, keepdims=True)).astype(np.float32),
            (ts_pos - ts_pos.mean(axis=0, keepdims=True)).astype(np.float32),
        )
    if align_mode != "kabsch":
        raise ValueError(f"unknown align_mode={align_mode}")

    source_center, target_center, rotation = kabsch_transform(p_pos, r_centered)
    p_aligned = apply_transform(p_pos, source_center, target_center, rotation)
    ts_aligned = apply_transform(ts_pos, source_center, target_center, rotation)
    return r_centered, p_aligned, ts_aligned


def init_molecule_dict():
    return {
        "num_atoms": [],
        "charges": [],
        "fragments": [],
        "positions": [],
        "rxn": [],
        "wB97x_6-31G(d).energy": [],
        "wB97x_6-31G(d).atomization_energy": [],
        "wB97x_6-31G(d).forces": [],
        "formula": [],
        "xtb_positions": [],
    }


def formula_from_symbols(symbols):
    parts = []
    for symbol in sorted(set(symbols), key=lambda s: (s != "C", s != "H", s)):
        count = symbols.count(symbol)
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def add_fragment(data, frag, symbols, coords, rxn_id):
    charges = [ATOM_TO_Z[symbol] for symbol in symbols]
    formula = formula_from_symbols(symbols)
    data[frag]["num_atoms"].append(len(symbols))
    data[frag]["charges"].append(charges)
    data[frag]["fragments"].append([0 for _ in symbols])
    data[frag]["positions"].append(coords.astype(np.float32))
    data[frag]["rxn"].append(rxn_id)
    data[frag]["wB97x_6-31G(d).energy"].append(0.0)
    data[frag]["wB97x_6-31G(d).atomization_energy"].append(np.float64(0.0))
    data[frag]["wB97x_6-31G(d).forces"].append(np.zeros_like(coords, dtype=np.float32))
    data[frag]["formula"].append(formula)
    data[frag]["xtb_positions"].append(coords.astype(np.float32))


def reaction_sort_key(name: str):
    match = re.search(r"reaction(\d+)", name, re.IGNORECASE)
    return int(match.group(1)) if match else name


def build_test_pkl_from_tar(tar_path: pathlib.Path, output_datadir: pathlib.Path, align_mode: str):
    data = {frag: init_molecule_dict() for frag in FRAGMENTS}
    data.update(
        {
            "single_fragment": [],
            "use_ind": [],
            "ts_guess": [],
            "ts_guess_sbv1": [],
            "ts_guess_true": [],
            "ts_guess_NEBCI-xtb": [],
        }
    )
    skipped = []

    reactions = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        for member in tqdm(members, desc=f"read {tar_path.name}"):
            path = pathlib.PurePosixPath(member.name)
            if path.name not in {"R.xyz", "P.xyz", "TS.xyz"}:
                continue
            base = str(path.parent)
            payload = tar.extractfile(member).read().decode("utf-8")
            reactions.setdefault(base, {})[path.name] = payload

    for base in tqdm(sorted(reactions, key=reaction_sort_key), desc=f"prepare {tar_path.name}"):
        files = reactions[base]
        if not {"R.xyz", "P.xyz", "TS.xyz"}.issubset(files):
            skipped.append((base, "missing P.xyz or TS.xyz"))
            continue
        try:
            r_symbols, r_pos = parse_xyz_text(files["R.xyz"])
            p_symbols, p_pos = parse_xyz_text(files["P.xyz"])
            ts_symbols, ts_pos = parse_xyz_text(files["TS.xyz"])
            if r_symbols != p_symbols or r_symbols != ts_symbols:
                raise ValueError("atom order differs across R/P/TS")
            r_pre, p_pre, ts_pre = prepare_coordinates(r_pos, p_pos, ts_pos, align_mode)
        except Exception as exc:
            skipped.append((base, str(exc)))
            continue

        rxn_id = pathlib.PurePosixPath(base).name
        add_fragment(data, "reactant", r_symbols, r_pre, rxn_id)
        add_fragment(data, "product", r_symbols, p_pre, rxn_id)
        add_fragment(data, "transition_state", r_symbols, ts_pre, rxn_id)
        ts_guess = ((r_pre + p_pre) / 2).astype(np.float32)
        data["ts_guess"].append(ts_guess)
        data["ts_guess_sbv1"].append(ts_guess)
        data["ts_guess_true"].append(ts_pre.astype(np.float32))
        data["ts_guess_NEBCI-xtb"].append(ts_guess)
        data["single_fragment"].append(1)
        data["use_ind"].append(len(data["use_ind"]))

    output_datadir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_datadir / "test.pkl"
    with open(pkl_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return pkl_path, skipped, len(data["use_ind"])


def load_model(checkpoint_path, datadir):
    from reactot.trainer.pl_trainer import SBModule

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = SBModule(**hparams)
    model.load_state_dict(state_dict)
    model = model.eval().to(device)
    model.training_config["use_sampler"] = False
    model.training_config["swapping_react_prod"] = False
    model.training_config["datadir"] = str(datadir)
    model.training_config["num_workers"] = 0
    model.setup(stage="test", device=str(device), swapping_react_prod=False)
    return model.to(device)


def write_single_xyz(xyz_path, atomic_numbers, coords, mode="w"):
    with open(xyz_path, mode, encoding="utf-8") as handle:
        handle.write(f"{len(atomic_numbers)}\n\n")
        for atomic_number, coord in zip(atomic_numbers, coords):
            symbol = Z_TO_ATOM[int(atomic_number)]
            x, y, z = coord
            handle.write(f"{symbol} {x:.8f} {y:.8f} {z:.8f}\n")


def write_batch_xyz(output_dir, rxn_ids, x0_size, atomic_numbers, r_pos, pred_ts_pos, true_ts_pos, p_pos):
    output_dir.mkdir(parents=True, exist_ok=True)
    start = 0
    for rxn_id, natoms in zip(rxn_ids, x0_size.tolist()):
        end = start + natoms
        batch_atomic_numbers = atomic_numbers[start:end]
        reactant_coords = r_pos[start:end].detach().cpu().numpy()
        pred_ts_coords = pred_ts_pos[start:end].detach().cpu().numpy()
        true_ts_coords = true_ts_pos[start:end].detach().cpu().numpy()
        product_coords = p_pos[start:end].detach().cpu().numpy()

        write_single_xyz(output_dir / f"{rxn_id}_pred_ts.xyz", batch_atomic_numbers, pred_ts_coords)
        write_single_xyz(output_dir / f"{rxn_id}_true_ts.xyz", batch_atomic_numbers, true_ts_coords)
        rxn_xyz_path = output_dir / f"{rxn_id}_rxn.xyz"
        write_single_xyz(rxn_xyz_path, batch_atomic_numbers, reactant_coords, mode="w")
        write_single_xyz(rxn_xyz_path, batch_atomic_numbers, pred_ts_coords, mode="a")
        write_single_xyz(rxn_xyz_path, batch_atomic_numbers, product_coords, mode="a")
        start = end


def infer_and_save(model, batch_size, output_dir, dryrun=False):
    loader = model.test_dataloader(bz=batch_size)
    rxn_ids = loader.dataset.reactant["rxn"]
    rows = []
    sample_offset = 0
    iterator = tqdm(enumerate(loader), total=len(loader), desc="infer")
    for _, batch in iterator:
        batch = move_to_device(batch, device)
        representations, _ = batch
        true_ts_pos = representations[1]["pos"].detach().cpu()
        r_pos, pred_ts_pos, p_pos, x0_size, x0_other, rmsds = model.eval_sample_batch(
            batch,
            return_all=True,
        )

        batch_count = len(rmsds)
        batch_rxn_ids = rxn_ids[sample_offset : sample_offset + batch_count]
        atomic_numbers = x0_other[:, -1].long().detach().cpu().numpy()
        write_batch_xyz(
            output_dir=output_dir,
            rxn_ids=batch_rxn_ids,
            x0_size=x0_size.detach().cpu(),
            atomic_numbers=atomic_numbers,
            r_pos=r_pos,
            pred_ts_pos=pred_ts_pos,
            true_ts_pos=true_ts_pos,
            p_pos=p_pos,
        )
        for rxn_id, rmsd in zip(batch_rxn_ids, rmsds):
            rows.append({"rxn": rxn_id, "rmsd": float(rmsd)})
        sample_offset += batch_count
        if dryrun:
            break
    return rows


def safe_stem(path: pathlib.Path):
    name = path.name
    for suffix in (".tar.gz", ".ckpt", ".pkl"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", name)


def run_one(opt, checkpoint_path: pathlib.Path, tar_path: pathlib.Path):
    log = logging.getLogger(__name__)
    run_name = f"{safe_stem(checkpoint_path)}__{safe_stem(tar_path)}"
    run_dir = pathlib.Path(opt.output_root) / run_name
    data_dir = run_dir / "processed_data"
    xyz_dir = run_dir / "xyz"

    pkl_path, skipped, n_samples = build_test_pkl_from_tar(tar_path, data_dir, opt.align_mode)
    log.info(f"prepared {n_samples} reactions -> {pkl_path}")
    if skipped:
        skipped_path = run_dir / "skipped.csv"
        skipped_path.parent.mkdir(parents=True, exist_ok=True)
        with open(skipped_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["reaction", "reason"])
            writer.writerows(skipped)
        log.info(f"skipped {len(skipped)} reactions -> {skipped_path}")

    model = load_model(str(checkpoint_path), data_dir)
    model.nfe = opt.nfe
    model.ddpm.opt = opt
    rows = infer_and_save(model, opt.batch_size, xyz_dir, dryrun=opt.dryrun)

    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rxn", "rmsd"])
        writer.writeheader()
        writer.writerows(rows)

    rmsds = [row["rmsd"] for row in rows]
    log.info(f"checkpoint={checkpoint_path}")
    log.info(f"tar={tar_path}")
    log.info(f"xyz_output_dir={xyz_dir}")
    log.info(f"metrics={metrics_path}")
    log.info(f"mean={np.mean(rmsds):.5f}, median={np.median(rmsds):.5f}, len={len(rmsds)}")


def main(opt):
    setup_logger(pathlib.Path(opt.log_dir))
    log = logging.getLogger(__name__)
    log.info("===== Start =====")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"device={device}, align_mode={opt.align_mode}")

    for checkpoint in opt.checkpoint:
        for tar_path in opt.tar:
            run_one(opt, pathlib.Path(checkpoint), pathlib.Path(tar_path))
    log.info("===== End =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint path. Repeat this option to run multiple checkpoints.",
    )
    parser.add_argument(
        "--tar",
        action="append",
        default=None,
        help="GDB raw tar.gz path. Repeat this option to run multiple datasets.",
    )
    parser.add_argument("--output-root", default="./outputs/gdb_raw_inference")
    parser.add_argument("--log-dir", default=".log")
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--nfe", type=int, default=100)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--align-mode", choices=["kabsch", "none"], default="kabsch")
    parser.add_argument("--solver", type=str, choices=["ddpm", "ei", "ode"], default="ode")
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--diz", type=str, default="linear", choices=["linear", "quad"])
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--method", type=str, default="midpoint")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)

    args = parser.parse_args()
    if args.tar is None:
        args.tar = [
            "./reactot/data/GDB-10-rxn_raw.tar.gz",
            "./reactot/data/GDB-17-rxn_raw.tar.gz",
        ]
    main(args)

# import argparse
# import logging
# import pathlib
# import sys

# import numpy as np
# import torch
# from rich.console import Console
# from rich.logging import RichHandler
# from tqdm import tqdm


# # device = "cuda:0"
# device = "cpu"


# def setup_logger(log_dir: pathlib.Path) -> None:
#     log_dir.mkdir(exist_ok=True, parents=True)

#     log_file = open(log_dir / "log.txt", "a")
#     file_console = Console(file=log_file, width=150)
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(message)s",
#         datefmt="[%X]",
#         force=True,
#         handlers=[RichHandler(), RichHandler(console=file_console)],
#     )


# def load_model(
#     checkpoint_path,
#     datadir="./reactot/data/t1x",
#     split="test",
# ):
#     from reactot.trainer.pl_trainer import SBModule

#     print(checkpoint_path)

#     checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
#     hparams = checkpoint.get("hyper_parameters", {})
#     state_dict = checkpoint.get("state_dict", checkpoint)

#     model = SBModule(**hparams)
#     model.load_state_dict(state_dict)
#     model = model.eval()
#     model = model.to(device)

#     model.training_config["use_sampler"] = False
#     model.training_config["swapping_react_prod"] = False
#     model.training_config["datadir"] = datadir

#     stage = "test" if split == "test" else "fit"
#     model.setup(stage=stage, device=device, swapping_react_prod=False)
#     return model





# def get_dataloader(model, split, batch_size):
#     if split == "train":
#         return model.train_dataloader(bz=batch_size)
#     if split == "val":
#         return model.val_dataloader(bz=batch_size, shuffle=False)
#     if split == "test":
#         return model.test_dataloader(bz=batch_size)
#     raise ValueError(f"Unsupported split: {split}")


# def write_single_xyz(xyz_path, atomic_numbers, coords, mode="w"):
#     atom_map = {
#         1: "H",
#         6: "C",
#         7: "N",
#         8: "O",
#         9: "F",
#     }
#     with open(xyz_path, mode) as f:
#         f.write(f"{len(atomic_numbers)}\n\n")
#         for atomic_number, coord in zip(atomic_numbers, coords):
#             symbol = atom_map[int(atomic_number)]
#             x, y, z = coord
#             f.write(f"{symbol} {x} {y} {z}\n")


# def write_batch_xyz(output_dir, rxn_ids, x0_size, atomic_numbers, r_pos, ts_pos, p_pos):
#     output_dir.mkdir(parents=True, exist_ok=True)
#     start = 0
#     for rxn_id, natoms in zip(rxn_ids, x0_size.tolist()):
#         end = start + natoms
#         batch_atomic_numbers = atomic_numbers[start:end]
#         reactant_coords = r_pos[start:end].cpu().numpy()
#         ts_coords = ts_pos[start:end].cpu().numpy()
#         product_coords = p_pos[start:end].cpu().numpy()

#         ts_xyz_path = output_dir / f"{rxn_id}_ts.xyz"
#         rxn_xyz_path = output_dir / f"{rxn_id}_rxn.xyz"

#         write_single_xyz(ts_xyz_path, batch_atomic_numbers, ts_coords, mode="w")
#         write_single_xyz(rxn_xyz_path, batch_atomic_numbers, reactant_coords, mode="w")
#         write_single_xyz(rxn_xyz_path, batch_atomic_numbers, ts_coords, mode="a")
#         write_single_xyz(rxn_xyz_path, batch_atomic_numbers, product_coords, mode="a")
#         start = end


# def infer_and_save(model, loader, split, output_dir, dryrun=False):
#     rmsds = []
#     rxn_ids = loader.dataset.reactant["rxn"]
#     sample_offset = 0
#     iterator = enumerate(loader)
#     total = len(loader)

#     if not dryrun:
#         iterator = tqdm(iterator, total=total)

#     for batch_idx, batch in iterator:
#         r_pos, ts_pos, p_pos, x0_size, x0_other, batch_rmsds = model.eval_sample_batch(
#             batch,
#             return_all=True,
#         )

#         batch_size = len(batch_rmsds)
#         batch_rxn_ids = rxn_ids[sample_offset: sample_offset + batch_size]
#         atomic_numbers = x0_other[:, -1].long().cpu().numpy()
#         write_batch_xyz(
#             output_dir=output_dir,
#             rxn_ids=batch_rxn_ids,
#             x0_size=x0_size.cpu(),
#             atomic_numbers=atomic_numbers,
#             r_pos=r_pos,
#             ts_pos=ts_pos,
#             p_pos=p_pos,
#         )
#         rmsds.extend(batch_rmsds)
#         sample_offset += batch_size

#         if dryrun:
#             break

#     return rmsds


# def main(opt):
#     setup_logger(pathlib.Path(".log"))
#     log = logging.getLogger(__name__)

#     log.info("===== Start =====")
#     log.info("Command used:\n{}".format(" ".join(sys.argv)))

#     model = load_model(
#         checkpoint_path=opt.checkpoint,
#         datadir=opt.datadir,
#         split=opt.split,
#     )
#     loader = get_dataloader(model, opt.split, opt.batch_size)

#     model.nfe = opt.nfe
#     model.ddpm.opt = opt
#     output_dir = pathlib.Path(opt.output_dir)

#     rmsds = infer_and_save(
#         model=model,
#         loader=loader,
#         split=opt.split,
#         output_dir=output_dir,
#         dryrun=opt.dryrun,
#     )

#     log.info(f"split={opt.split}, datadir={opt.datadir}")
#     log.info(f"xyz_output_dir={output_dir}")
#     log.info(f"mean={np.mean(rmsds):.5f}, median={np.median(rmsds):.5f}, {len(rmsds)=}")
#     log.info("===== End =====")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--batch-size", type=int, default=72)
#     parser.add_argument("--nfe", type=int, default=100)
#     parser.add_argument("--save", type=str, default="debug")
#     parser.add_argument("--dryrun", action="store_true")

#     parser.add_argument("--solver", type=str, choices=["ddpm", "ei", "ode"])
#     parser.add_argument("--checkpoint", type=str, required=True)
#     parser.add_argument("--datadir", type=str, default="./reactot/data/t1x")
#     parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
#     parser.add_argument("--output-dir", type=str, default="./outputs/test_xyz")

#     parser.add_argument("--order", type=int, default=1)
#     parser.add_argument("--diz", type=str, default="linear", choices=["linear", "quad"])
#     parser.add_argument("--normalize", action="store_true")

#     parser.add_argument("--method", type=str, default="midpoint")
#     parser.add_argument("--atol", type=float, default=1e-2)
#     parser.add_argument("--rtol", type=float, default=1e-2)

#     opt = parser.parse_args()
#     main(opt)

# """

# 终版
# python our_evaluation.py \
#   --solver ode \
#   --method midpoint \
#   --checkpoint checkpoint/RPSB-FT-Schedule/leftnet-ts_guess_NEBCI-xtb-ema-3739d3534f81/sb-epoch=199-val_ep_scaled_err=0.0441.ckpt \
#   --datadir ./reactot/data/data_new_split \
#   --split test \
#   --batch-size 72 \
#   --nfe 100



# python our_evaluation_xyz.py \
#   --solver ode \
#   --method midpoint \
#   --checkpoint checkpoint/RPSB-FT-Schedule/leftnet-ts_guess_NEBCI-xtb-ema-3739d3534f81/sb-epoch=199-val_ep_scaled_err=0.0441.ckpt \
#   --datadir ./reactot/data/data_new_split \
#   --split test \
#   --output-dir ./outputs/t1x_test_xyz \
#   --batch-size 72 \
#   --nfe 100
  
# python our_evaluation_xyz.py \
#   --solver ode \
#   --method midpoint \
#   --checkpoint checkpoint/react/leftnet-ts_guess_NEBCI-xtb-ema-f8e73e8d54f6/sb-epoch=199-val_ep_scaled_err=0.0463.ckpt \
#   --datadir ./reactot/data/data_new_split \
#   --split test \
#   --output-dir ./outputs/ot_only_t1x \
#   --batch-size 72 \
#   --nfe 100


# python our_evaluation_xyz.py \
#   --solver ode \
#   --method midpoint \
#   --checkpoint checkpoint/react-mix/None/sb-epoch=199-val_ep_scaled_err=0.0731.ckpt \
#   --datadir ./reactot/data/t1x_rgd1_mix \
#   --split test \
#   --output-dir ./outputs/ot_mix \
#   --batch-size 72 \
#   --nfe 100
  
  
# python our_evaluation_xyz.py \
#   --solver ode \
#   --method midpoint \
#   --checkpoint checkpoint/react-mix/None/sb-epoch=199-val_ep_scaled_err=0.0591.ckpt \
#   --datadir ./reactot/data/t1x_rgd1_mix \
#   --split test \
#   --output-dir ./outputs/ot_mix \
#   --batch-size 72 \
#   --nfe 100



# python our_evaluation_xyz.py \
#   --solver ode \
#   --method midpoint \
#   --checkpoint  checkpoint/react-mix-dim/None/sb-epoch=159-val_ep_scaled_err=0.0496.ckpt\
#   --datadir ./reactot/data/t1x_rgd1_mix \
#   --split test \
#   --output-dir ./outputs/ot_mix_dim \
#   --batch-size 72 \
#   --nfe 100
  
  
# """


import argparse
import logging
import pathlib
import sys
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def setup_logger(log_dir: pathlib.Path) -> None:
    log_dir.mkdir(exist_ok=True, parents=True)

    log_file = open(log_dir / "log.txt", "a")
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


def load_model(
    checkpoint_path,
    datadir="./reactot/data/t1x",
    split="test",
):
    from reactot.trainer.pl_trainer import SBModule

    print(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    state_dict = checkpoint.get("state_dict", checkpoint)

    model = SBModule(**hparams)
    model.load_state_dict(state_dict)
    model = model.eval()
    model = model.to(device)

    model.training_config["use_sampler"] = False
    model.training_config["swapping_react_prod"] = False
    model.training_config["datadir"] = datadir

    stage = "test" if split == "test" else "fit"
    model.setup(stage=stage, device=str(device), swapping_react_prod=False)
    model = model.to(device)
    return model


def get_dataloader(model, split, batch_size):
    if split == "train":
        return model.train_dataloader(bz=batch_size)
    if split == "val":
        return model.val_dataloader(bz=batch_size, shuffle=False)
    if split == "test":
        return model.test_dataloader(bz=batch_size)
    raise ValueError(f"Unsupported split: {split}")


def write_single_xyz(xyz_path, atomic_numbers, coords, mode="w"):
    atom_map = {
        1: "H",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
    }
    with open(xyz_path, mode) as f:
        f.write(f"{len(atomic_numbers)}\n\n")
        for atomic_number, coord in zip(atomic_numbers, coords):
            symbol = atom_map[int(atomic_number)]
            x, y, z = coord
            f.write(f"{symbol} {x} {y} {z}\n")


def write_batch_xyz(output_dir, rxn_ids, x0_size, atomic_numbers, r_pos, ts_pos, p_pos):
    output_dir.mkdir(parents=True, exist_ok=True)
    start = 0
    for rxn_id, natoms in zip(rxn_ids, x0_size.tolist()):
        end = start + natoms
        batch_atomic_numbers = atomic_numbers[start:end]
        reactant_coords = r_pos[start:end].detach().cpu().numpy()
        ts_coords = ts_pos[start:end].detach().cpu().numpy()
        product_coords = p_pos[start:end].detach().cpu().numpy()

        ts_xyz_path = output_dir / f"{rxn_id}_ts.xyz"
        rxn_xyz_path = output_dir / f"{rxn_id}_rxn.xyz"

        write_single_xyz(ts_xyz_path, batch_atomic_numbers, ts_coords, mode="w")
        write_single_xyz(rxn_xyz_path, batch_atomic_numbers, reactant_coords, mode="w")
        write_single_xyz(rxn_xyz_path, batch_atomic_numbers, ts_coords, mode="a")
        write_single_xyz(rxn_xyz_path, batch_atomic_numbers, product_coords, mode="a")
        start = end


def infer_and_save(model, loader, split, output_dir, dryrun=False):
    rmsds = []
    rxn_ids = loader.dataset.reactant["rxn"]
    sample_offset = 0
    iterator = enumerate(loader)
    total = len(loader)

    if not dryrun:
        iterator = tqdm(iterator, total=total)

    for batch_idx, batch in iterator:
        batch = move_to_device(batch, device)
        r_pos, ts_pos, p_pos, x0_size, x0_other, batch_rmsds = model.eval_sample_batch(
            batch,
            return_all=True,
        )

        batch_size = len(batch_rmsds)
        batch_rxn_ids = rxn_ids[sample_offset: sample_offset + batch_size]
        atomic_numbers = x0_other[:, -1].long().detach().cpu().numpy()
        write_batch_xyz(
            output_dir=output_dir,
            rxn_ids=batch_rxn_ids,
            x0_size=x0_size.detach().cpu(),
            atomic_numbers=atomic_numbers,
            r_pos=r_pos,
            ts_pos=ts_pos,
            p_pos=p_pos,
        )
        rmsds.extend(batch_rmsds)
        sample_offset += batch_size

        if dryrun:
            break

    return rmsds


def main(opt):
    setup_logger(pathlib.Path(".log"))
    log = logging.getLogger(__name__)

    log.info("===== Start =====")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"device={device}")

    model = load_model(
        checkpoint_path=opt.checkpoint,
        datadir=opt.datadir,
        split=opt.split,
    )
    loader = get_dataloader(model, opt.split, opt.batch_size)

    model.nfe = opt.nfe
    model.ddpm.opt = opt
    output_dir = pathlib.Path(opt.output_dir)

    rmsds = infer_and_save(
        model=model,
        loader=loader,
        split=opt.split,
        output_dir=output_dir,
        dryrun=opt.dryrun,
    )

    log.info(f"split={opt.split}, datadir={opt.datadir}")
    log.info(f"xyz_output_dir={output_dir}")
    log.info(f"mean={np.mean(rmsds):.5f}, median={np.median(rmsds):.5f}, {len(rmsds)=}")
    log.info("===== End =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--nfe", type=int, default=100)
    parser.add_argument("--save", type=str, default="debug")
    parser.add_argument("--dryrun", action="store_true")

    parser.add_argument("--solver", type=str, choices=["ddpm", "ei", "ode"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--datadir", type=str, default="./reactot/data/t1x")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default="./outputs/test_xyz")

    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--diz", type=str, default="linear", choices=["linear", "quad"])
    parser.add_argument("--normalize", action="store_true")

    parser.add_argument("--method", type=str, default="midpoint")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)

    opt = parser.parse_args()
    main(opt)

"""
python our_evaluation_xyz.py \
  --solver ode \
  --method midpoint \
  --checkpoint checkpoint/react-mix-dim/None/sb-epoch=159-val_ep_scaled_err=0.0496.ckpt \
  --datadir ./reactot/data/t1x_rgd1_mix \
  --split test \
  --output-dir ./outputs/ot_mix_dim \
  --batch-size 72 \
  --nfe 100
  
  
  
  

"""

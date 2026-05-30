import argparse
import logging
import pathlib
import sys

import numpy as np
import torch
from rich.console import Console
from rich.logging import RichHandler


device = "cuda:0"
# device = "cpu"


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
    model.setup(stage=stage, device=device, swapping_react_prod=False)
    return model


def get_dataloader(model, split, batch_size):
    if split == "train":
        return model.train_dataloader(bz=batch_size)
    if split == "val":
        return model.val_dataloader(bz=batch_size, shuffle=False)
    if split == "test":
        return model.test_dataloader(bz=batch_size)
    raise ValueError(f"Unsupported split: {split}")


def main(opt):
    setup_logger(pathlib.Path(".log"))
    log = logging.getLogger(__name__)

    log.info("===== Start =====")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))

    model = load_model(
        checkpoint_path=opt.checkpoint,
        datadir=opt.datadir,
        split=opt.split,
    )
    loader = get_dataloader(model, opt.split, opt.batch_size)

    model.nfe = opt.nfe
    model.ddpm.opt = opt

    if opt.dryrun:
        batch = next(iter(loader))
        _, _, _, _, _, rmsds = model.eval_sample_batch(
            batch,
            return_all=True,
        )
    else:
        _, rmsds = model.eval_rmsd(
            loader,
            write_xyz=False,
            bz=opt.batch_size,
            refpath=f"ref_ts/{opt.split}",
            localpath=f"{opt.solver}-{opt.method}/{opt.split}/nfe{opt.nfe}/",
        )

    log.info(f"split={opt.split}, datadir={opt.datadir}")
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

    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--diz", type=str, default="linear", choices=["linear", "quad"])
    parser.add_argument("--normalize", action="store_true")

    parser.add_argument("--method", type=str, default="midpoint")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)

    opt = parser.parse_args()
    main(opt)


"""
python our_evaluation.py \
  --solver ode \
  --method midpoint \
  --checkpoint checkpoint/RPSB-FT-Schedule/leftnet-ts_guess_NEBCI-xtb-ema-3739d3534f81/sb-epoch=199-val_ep_scaled_err=0.0441.ckpt \
  --datadir ./reactot/data/data_new_split \
  --split test \
  --batch-size 72 \
  --nfe 100                #正常结果
  
  
  
  
python our_evaluation.py \
  --solver ode \
  --method midpoint \
  --checkpoint  checkpoint/react-mix/None/sb-epoch=199-val_ep_scaled_err=0.0591.ckpt\
  --datadir reactot/data/t1x_rgd1_mix \
  --split test \
  --batch-size 72 \
  --nfe 100                #没报错det，但就是低
  
  
python our_evaluation.py \
  --solver ode \
  --method midpoint \
  --checkpoint  checkpoint/react-mix/None/sb-epoch=199-val_ep_scaled_err=0.0591.ckpt\
  --datadir reactot/data/data_new_split \
  --split test \
  --batch-size 72 \
  --nfe 100      #没报错det，但就是低
  
  
  
  
  python our_evaluation.py \
  --solver ode \
  --method midpoint \
  --checkpoint  \
  --datadir reactot/data/data_new_split \
  --split test \
  --batch-size 72 \
  --nfe 100
  
  

python our_evaluation.py \
  --solver ode \
  --method midpoint \
  --checkpoint  checkpoint/react-mix/None/sb-epoch=199-val_ep_scaled_err=0.0591.ckpt\
  --datadir reactot/data/t1x_rgd1_mix2 \
  --split test \
  --batch-size 72 \
  --nfe 100      #报错了detected nan
  
  
  
  
python our_evaluation.py \
  --solver ode \
  --method midpoint \
  --checkpoint  rpsb_ts1x_mix_dim.ckpt\
  --datadir reactot/data/t1x_rgd1_mix \
  --split test \
  --batch-size 72 \
  --nfe 100      
  
  
  
python our_evaluation.py \
  --solver ode \
  --method midpoint \
  --checkpoint  rpsb_ts1x_mix_dim.ckpt\
  --datadir reactot/data/t1x_rgd1_mix \
  --split test \
  --batch-size 72 \
  --nfe 100      
  

python our_evaluation.py \
  --solver ode \
  --method midpoint \
  --checkpoint  checkpoint/react-mix-dim/None/sb-epoch=199-val_ep_scaled_err=0.1100.ckpt\
  --datadir reactot/data/t1x_rgd1_mix \
  --split test \
  --batch-size 72 \
  --nfe 100     
  
  
python our_evaluation.py \
  --solver ode \
  --method midpoint \
  --checkpoint    checkpoint/react-mix-dim/None/sb-epoch=159-val_ep_scaled_err=0.0496.ckpt\
  --datadir reactot/data/t1x_rgd1_mix \
  --split test \
  --batch-size 72 \
  --nfe 100       


python our_evaluation.py \
  --solver ode \
  --method midpoint \
  --checkpoint   checkpoint/react-mix/None/sb-epoch=147-val_ep_scaled_err=0.0401.ckpt \
  --datadir reactot/data/t1x_rgd1_mix \
  --split test \
  --batch-size 72 \
  --nfe 100   
  
  
  
python our_evaluation.py \
  --solver ode \
  --method midpoint \
  --checkpoint   rpsb_ts1x_mix_dim-0.ckpt \
  --datadir reactot/data/t1x_rgd1_mix \
  --split test \
  --batch-size 72 \
  --nfe 100 
  
  
 

"""

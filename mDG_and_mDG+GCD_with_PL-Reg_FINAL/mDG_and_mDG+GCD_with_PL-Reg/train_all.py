import argparse
import collections
import random
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from sconf import Config
from prettytable import PrettyTable

from domainbed.datasets import get_dataset
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.writers import get_writer
from domainbed.lib.logger import Logger
from domainbed.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Domain generalization", allow_abbrev=False)
    parser.add_argument("name", type=str)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--data_dir", type=str, default="datadir/")
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument(
        "--trial_seed",
        type=int,
        default=0,
        help="Trial number (used for seeding split_dataset and random_hparams).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of steps. Default is dataset-dependent."
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        help="Checkpoint every N steps. Default is dataset-dependent.",
    )
    parser.add_argument("--test_envs", type=int, nargs="+", default=None)
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--model_save", default=None, type=int, help="Model save start step")
    #  parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tb_freq", default=10)
    parser.add_argument("--mask_range",  type=float, default=0.1)
    parser.add_argument("--setting_name", type=str, default="CE")

    parser.add_argument("--debug", action="store_true", help="Run w/ debug mode")
    parser.add_argument("--show", action="store_true", help="Show args and hparams w/o run")
    parser.add_argument(
        "--evalmode",
        default="fast",
        help="[fast, all]. if fast, ignore train_in datasets in evaluation time.",
    )
    parser.add_argument("--prebuild_loader", action="store_true", help="Pre-build eval loaders")
    args, left_argv = parser.parse_known_args()
    args.deterministic = True

    # setup hparams
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    keys = ["config.yaml"] + args.configs
    keys = [open(key, encoding="utf8") for key in keys]

    hparams = Config(*keys, default=hparams)
    hparams.argv_update(left_argv)

    # if not hparams.use_MIRO:
    #     print('+' * 100)
    #     print('using ERM hparams')
    #     hparams = hparams_registry.default_hparams('ERM', args.dataset)
    #     keys = ["config.yaml"] + args.configs
    #     keys = [open(key, encoding="utf8") for key in keys]
    #
    #     hparams = Config(*keys, default=hparams)
    #     hparams.argv_update(left_argv)

    # setup debug
    if args.debug:
        args.checkpoint_freq = 5
        args.steps = 10
        args.name += "_debug"

    timestamp = misc.timestamp()
    args.unique_name = f"{timestamp}_{args.name}"

    # path setup
    args.work_dir = Path(".")
    args.data_dir = Path(args.data_dir)



    # TODO
    # args.out_root = args.work_dir / Path("train_output_test") / args.dataset
    args.out_root = args.work_dir / Path("train_output_test") / args.dataset


    args.out_dir = args.out_root/ args.unique_name

    args.out_dir.mkdir(exist_ok=True, parents=True)



    writer = get_writer(args.out_root / "runs" / args.unique_name)
    logger = Logger.get(args.out_dir / "log.txt")
    if args.debug:
        logger.setLevel("DEBUG")
    cmd = " ".join(sys.argv)
    logger.info(f"Command :: {cmd}")

    logger.nofmt("Environment:")
    logger.nofmt("\tPython: {}".format(sys.version.split(" ")[0]))
    logger.nofmt("\tPyTorch: {}".format(torch.__version__))
    logger.nofmt("\tTorchvision: {}".format(torchvision.__version__))
    logger.nofmt("\tCUDA: {}".format(torch.version.cuda))
    logger.nofmt("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    logger.nofmt("\tNumPy: {}".format(np.__version__))
    logger.nofmt("\tPIL: {}".format(PIL.__version__))
    filename = f"{args.dataset}_res.txt"
    log_save_path = args.out_root / filename


    # Different to DomainBed, we support CUDA only.
    assert torch.cuda.is_available(), "CUDA is not available"

    logger.nofmt("Args:")
    for k, v in sorted(vars(args).items()):
        logger.nofmt("\t{}: {}".format(k, v))

    logger.nofmt("HParams:")
    for line in hparams.dumps().split("\n"):
        logger.nofmt("\t" + line)

    if args.show:
        exit()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic

    # Dummy datasets for logging information.
    # Real dataset will be re-assigned in train function.
    # test_envs only decide transforms; simply set to zero.
    dataset, _in_splits, _out_splits = get_dataset([0], args, hparams)
    print('+' * 100)
    print(dataset.num_classes)
    if args.mask_range > 0:
        mask_range = int(dataset.num_classes * (1 - args.mask_range))
    else:
        mask_range = args.mask_range
    print('+' * 100)
    print(mask_range)
    # print dataset information
    logger.nofmt("Dataset:")
    logger.nofmt(f"\t[{args.dataset}] #envs={len(dataset)}, #classes={dataset.num_classes}")
    for i, env_property in enumerate(dataset.environments):
        logger.nofmt(f"\tenv{i}: {env_property} (#{len(dataset[i])})")
    logger.nofmt("")

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    logger.info(f"n_steps = {n_steps}")
    logger.info(f"checkpoint_freq = {checkpoint_freq}")

    org_n_steps = n_steps
    n_steps = (n_steps // checkpoint_freq) * checkpoint_freq + 1
    logger.info(f"n_steps is updated to {org_n_steps} => {n_steps} for checkpointing")

    if not args.test_envs:
        args.test_envs = [[te] for te in range(len(dataset))]


    # args.test_envs = [[0]]
    # args.test_envs = args.test_envs[::-1]
    # args.test_envs = [[3]]
    logger.info(f"Target test envs = {args.test_envs}")

    ###########################################################################
    # Run
    ###########################################################################
    all_records = []
    results = collections.defaultdict(list)

    for test_env in args.test_envs:
        with open(log_save_path, 'a') as f:
            f.write(f"test_env: {test_env}")
            f.write('\n')
            f.write(f"Command: {' '.join(sys.argv)}")
            f.write('\n')
            f.write("Unique name: %s" % args.unique_name)
            f.write('\n')
            f.write("Out path: %s" % args.out_dir)
            f.write('\n')
            f.write("Algorithm: %s" % args.algorithm)
            f.write('\n')
            f.write("Dataset: %s" % args.dataset)
            f.write('\n')

        res, records = train(
            test_env,
            args=args,
            hparams=hparams,
            n_steps=n_steps,
            checkpoint_freq=checkpoint_freq,
            logger=logger,
            writer=writer,
            mask_range=mask_range,
            log_save_path=log_save_path
        )
        all_records.append(records)
        for k, v in res.items():
            results[k].append(v)

    # log summary table


    logger.info("=== Summary ===")
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info("Unique name: %s" % args.unique_name)
    logger.info("Out path: %s" % args.out_dir)
    logger.info("Algorithm: %s" % args.algorithm)
    logger.info("Dataset: %s" % args.dataset)



    table = PrettyTable(["Selection"] + dataset.environments + ["Avg."])

    with open(log_save_path, 'a') as f:
        f.write("=== Summary ===")
        f.write('\n')
        f.write(f"Command: {' '.join(sys.argv)}")
        f.write('\n')

    for key, row in results.items():
        # row.append(np.mean(row))
        # row = [acc for acc in row]
        # table.add_row([key] + str(row))
        print(key, row)
        with open(log_save_path, 'a') as f:
            f.write(f"{key} = {row}")
            f.write('\n')
    # logger.nofmt(table)

if __name__ == "__main__":
    main()

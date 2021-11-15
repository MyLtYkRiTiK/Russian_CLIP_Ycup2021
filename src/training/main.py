import gc
import json
import logging
import os
import shutil
from pathlib import Path
from time import gmtime, strftime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from src.clip.clip import load
from src.clip.model import convert_weights, CLIP
from src.training.data import get_data
from src.training.logger import setup_primary_logging, setup_worker_logging
from src.training.scheduler import cosine_lr
from src.training.train import train


def convert_models_to_fp32(model):
    for p in model.parameters( ):
        p.data = p.data.float( )
        if p.grad:
            p.grad.data = p.grad.data.float( )


def is_master(args):
    return (not args.distributed) or args.gpu == 0 or args.dp


def main_worker(gpu, ngpus_per_node, log_queue, args):
    args.gpu = gpu
    args.rank = gpu
    setup_worker_logging(args.rank, log_queue, args.log_level)

    # Log and save params.
    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")
    shutil.copyfile(os.path.join(args.checkpoint_parent_path, "out.log"),
                    os.path.join(args.checkpoint_parent_path, "tensorboard", "out.log"))
    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    if args.dp:
        args.batch_size *= args.world_size

    if args.gpu is not None:
        logging.info(f"Use GPU: {args.gpu} for training")
        torch.cuda.set_device(args.gpu)

    # Do not use skip_reset unless you want to use on of the CLIP model
    if args.openai_pretrained:
        model, preprocess_train, preprocess_val = load(
            args.model,
            jit=False,
            is_train=True)
    else:
        model_config_file = Path(__file__).parent / f"{args.model.replace('/', '-')}.json"
        print('Loading model from', model_config_file)
        assert os.path.exists(model_config_file)
        with open(model_config_file, 'r') as f:
            model_info = json.load(f)
        model = CLIP(**model_info)
        convert_weights(model)

        model_orig, preprocess_train, _ = load(
            args.model,
            jit=False,
            is_train=True)

        model.visual = model_orig.visual
        del model_orig
        gc.collect( )

    if args.precision == "amp" or args.precision == "fp32" or args.gpu is None:
        convert_models_to_fp32(model)

    if not torch.cuda.is_available( ):
        model.float( )
        logging.warning("using CPU, this will be slow")
    else:
        model.cuda(args.gpu)
        if args.precision == "fp16":
            convert_weights(model)

        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        if args.dp:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)

        if args.precision == "fp16":
            convert_weights(model)

    data = get_data(args, (preprocess_train, preprocess_val))

    exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n: not exclude(n)

    named_parameters = list(model.named_parameters( ))
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    if args.metadata_file is None:
        optimizer = None
        scheduler = None
    else:
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    scaler = GradScaler( ) if args.precision == "amp" else None

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items( )))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items( )}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    cudnn.deterministic = False

    args.save_logs = (args.logs is not None and args.logs != '' and args.logs.lower( ) != 'none') and (
            (not args.distributed) or args.gpu == 0
    )
    writer = None
    if args.save_logs and args.tensorboard:
        writer = SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project="open-clip",
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if args.distributed:
        for p in model.module.visual.parameters( ):
            p.requires_grad = False
    else:
        for p in model.visual.parameters( ):
            p.requires_grad = False

    for epoch in range(start_epoch, args.epochs):
        if args.gpu == 0:
            logging.info(f'Start epoch {epoch}')
        train(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        steps = data["train"].dataloader.num_batches * (epoch + 1)

        # Saving checkpoints.
        if args.save_logs and (args.gpu == 0 or (not args.distributed)):
            if (epoch + 1) == args.epochs or (
                    args.save_frequency > 0 and ((epoch + 1) % args.save_frequency) == 0
            ):
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "name": args.name,
                        "state_dict": model.state_dict( ),
                        "optimizer": optimizer.state_dict( ),
                    },
                    os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt"),
                )
            if (epoch + 1) % 5 == 0:
                model_eval = model.eval( )
                torch.save(
                    {
                        "state_dict": model_eval.state_dict( ),
                    },
                    os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_light.pt"),
                )
                shutil.move(os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_light.pt"),
                            os.path.join(args.checkpoint_parent_path, f"epoch_{epoch + 1}_light.pt"))

    if args.wandb and (args.gpu == 0 or (not args.distributed)):
        wandb.finish( )


class Args:
    save_frequency = 1
    report_to = 'tensorboard'
    metadata_file = 'path_to_file/metadata.json'
    images_directory = "path_to_dataset/dataset_small/"
    warmup = 10000
    batch_size = 128
    wd = 0.1
    epochs = 30
    model = 'ViT-B/32'
    precision = 'amp'
    openai_pretrained = False
    debug = False
    gpu = 0
    lr = 4.0e-7
    workers = 14
    name = None
    dataset_type = 'ycup'
    logs = './logs/'
    skip_aggregate = False
    aggregate = True  # not args.skip_aggregate
    copy_codebase = False
    dp = False
    dist_backend = 'nccl'
    dist_url = "tcp://127.0.0.1:6100"
    beta1 = 0.9
    beta2 = 0.98
    eps = 1.0e-6
    resume = None
    val_data = None
    use_bn_sync = False


def main():
    args = Args( )

    # get the name of the experiments
    if args.name is None:
        args.name = strftime(
            f"lr={args.lr}_"
            f"wd={args.wd}_"
            f"agg={args.aggregate}_"
            f"model={args.model}_"
            f"batchsize={args.batch_size}_workers={args.workers}_date=%Y-%m-%d-%H-%M-%S",
            gmtime( ),
        )

    if args.copy_codebase:
        import sys, subprocess
        from shutil import copytree, ignore_patterns
        new_code_path = os.path.join(args.logs, args.name, "code")
        if os.path.exists(new_code_path):
            print(
                f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
            )
            return -1
        print(f"Copying codebase to {new_code_path}")
        current_code_path = os.path.realpath(__file__)
        for _ in range(3):
            current_code_path = os.path.dirname(current_code_path)
        copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
        print("Done copying code.")
        os.environ["PYTHONPATH"] = f"{os.environ['PYTHONPATH']}:{os.path.join(new_code_path, 'src')}"
        main_file = os.path.join(new_code_path, "src", "", "main.py")
        argv = sys.argv
        argv.remove('--copy-codebase')
        argv.extend(['--name', args.name])
        command = [sys.executable] + argv
        print("Executing command:", " ".join(command))
        subprocess.check_call(command)
        return 1

    args.log_path = os.path.join(args.logs, args.name, "out.log")
    if os.path.exists(args.log_path):
        print(
            "Error. Experiment already exists. Use --name {} to specify a new experiment."
        )
        return -1

    assert args.precision in ['amp', 'fp16', 'fp32']

    args.ngpus_per_node = torch.cuda.device_count( )

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to

    args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    args.checkpoint_parent_path = os.path.join(args.logs, args.name)
    for dirname in [args.tensorboard_path, args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    # Set multiprocessing type to spawn.
    # This is important for logging to work with multiprocessing.
    torch.multiprocessing.set_start_method("spawn")

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level)

    # Distributed training = training on more than one GPU.
    # Also easily possible to extend to multiple nodes & multiple GPUs.
    args.distributed = (args.gpu is None) and torch.cuda.is_available( ) and (not args.dp)
    if args.distributed:
        ngpus_per_node = torch.cuda.device_count( )
        args.world_size = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, log_queue, args))
    else:
        if args.dp:
            args.gpu = args.multigpu[0]
            args.world_size = len(args.multigpu)
        else:
            args.world_size = 1
        main_worker(args.gpu, None, log_queue, args)


if __name__ == "__main__":
    main( )

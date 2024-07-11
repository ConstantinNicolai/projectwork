from models.ResMLP import ResMLP

import os
import time
import re
import pickle
from types import SimpleNamespace
from typing import NamedTuple, Callable

import numpy as np

import torch
from torch import nn
from torch import Tensor, distributed
from torch.utils.data import DataLoader

from tqdm import tqdm, trange

from deep_sparse_nine import Linear

from timm.data.mixup import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy

from util.model_training_state import (
    ModelTrainingState,
    load_pretrained,
    save_checkpoint,
)
from util.data_loader import generate_data_loader
from util.open import open_file_or_stdout
from util.average import Average

from util.attr_dict import AttrDict
from util.grouped_parser import GroupedArgumentParser


def set_to_print_only_by_force():
    builtin_printer = __builtins__.printer

    def altered_printer(*args, **kwargs):
        if kwargs.pop("force", False):
            builtin_printer(*args, **kwargs)

    __builtins__.printer = altered_printer


def set_sparsity(state: ModelTrainingState, sparsity: float) -> None:
    for name, layer in state.model.named_modules():
        if isinstance(layer, Linear):
            layer.update_sparsity(sparsity)


def initialize_distributed():
    distargs = AttrDict()

    if "LOCAL_RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("Not using distributed training")
        distargs.device = torch.device("cuda")
        distargs.distributed = False
        distargs.rank = 0
        distargs.gpu = 0
        return distargs

    distargs.rank = int(os.environ["LOCAL_RANK"])
    distargs.worldsize = int(os.environ["WORLD_SIZE"])
    distargs.gpu = distargs.rank % torch.cuda.device_count()

    print(
        "Distributed training: initializing rank {}, gpu {}, worldsize = {}".format(
            distargs.rank, distargs.gpu, distargs.worldsize
        ),
        flush=True,
    )
    torch.cuda.set_device(distargs.gpu)
    distargs.device = torch.device("cuda", distargs.gpu)
    distargs.dist_backend = "nccl"
    distargs.distributed = True
    distributed.init_process_group(
        backend=distargs.dist_backend,
        init_method="env://",
        world_size=distargs.worldsize,
        rank=distargs.rank,
    )
    distributed.barrier()

    if distargs.rank != 0:
        set_to_print_only_by_force()

    return distargs


def get_loss_function(mixup_settings: SimpleNamespace, gpu: int):
    use_cutmix = (
        mixup_settings.cutmix > 0
        or mixup_settings.mixup > 0
        or mixup_settings.cutmix_minmax is not None
    )
    # Loss function changes when using cutmix and or label smoothing
    if use_cutmix:
        return SoftTargetCrossEntropy().cuda(gpu)
    elif mixup_settings.smoothing:
        return LabelSmoothingCrossEntropy(
            smoothing=mixup_settings.smoothing
        ).cuda(gpu)

    return nn.CrossEntropyLoss().cuda(gpu)


def train(
    trainloader: DataLoader,
    state: ModelTrainingState,
    loss_function: Callable,
    mixup_fn: Callable,
    epoch: int,
    gpu: int,
    distributed: bool,
    silent: bool,
) -> float:
    state.model.train()
    lss = Average("loss", ":.4f")
    with tqdm(
        trainloader,
        unit="batch",
        desc=f"training epoch {epoch}",
        position=0,
        disable=silent,
    ) as tbatch:
        for images, targets in tbatch:
            if torch.cuda.is_available():
                images = images.cuda(gpu, non_blocking=True)
                targets = targets.cuda(gpu, non_blocking=True)

            if mixup_fn is not None:
                images, targets = mixup_fn(images, targets)

            output = state.model(images)
            loss = loss_function(output, targets)

            state.optimizer.zero_grad()

            lss.update(loss.item(), images.size(0))

            loss.backward()
            state.optimizer.step()

            tbatch.set_postfix(
                loss=f"{lss.val:.4f} -> avg({lss.avg:.4f})",
            )
    lss.synchronize(distributed, gpu)
    return lss.avg


class ValidationAccuracy(NamedTuple):
    acc1: float
    acc5: float
    trainacc1: float
    trainacc5: float


def validate(
    testloader: DataLoader,
    trainloader: DataLoader | None,
    model,
    gpu: int,
    distributed: bool,
    silent: bool,
) -> ValidationAccuracy:
    model.eval()

    top1 = Average("top1", ":.4f")
    top5 = Average("top5", ":.4f")
    traintop1 = Average("traintop1", fmt=":.4f")
    traintop5 = Average("traintop1", fmt=":.4f")

    with torch.no_grad():
        with tqdm(
            testloader,
            unit="batch",
            desc="validating",
            position=0,
            disable=silent,
        ) as tbatch:
            for images, targets in tbatch:
                images = images.cuda(gpu, non_blocking=True)
                targets = targets.cuda(gpu, non_blocking=True)

                output = model(images)

                acc1, acc5 = accuracy(output, targets, topk=(1, 5))
                top1.update(float(acc1[0]), images.size(0))
                top5.update(float(acc5[0]), images.size(0))

                tbatch.set_postfix(
                    accuracy=f"@Top-1: {top1.val:.4f} -> avg({top1.avg:.4f}), @Top-5: {top5.val:.4f} -> avg({top5.avg:.4f}))"
                )

        if trainloader:
            with tqdm(
                trainloader,
                unit="batch",
                desc="validating w/o data augmentation",
                position=0,
                disable=silent,
            ) as tbatch:
                for images, targets in tbatch:
                    images = images.cuda(gpu, non_blocking=True)
                    targets = targets.cuda(gpu, non_blocking=True)
                    output = model(images)

                    trainacc1, trainacc5 = accuracy(
                        output, targets, topk=(1, 5)
                    )
                    traintop1.update(trainacc1.item(), images.size(0))
                    traintop5.update(trainacc5.item(), images.size(0))
                    tbatch.set_postfix(
                        accuracy=f"@Top-1: {traintop1.val:.4f} -> avg({traintop1.avg:.4f}), @Top-5: {traintop5.val:.4f} -> avg({traintop5.avg:.4f}))"
                    )

    top1.synchronize(distributed, gpu)
    top5.synchronize(distributed, gpu)
    traintop1.synchronize(distributed, gpu)
    traintop5.synchronize(distributed, gpu)

    return ValidationAccuracy(top1.avg, top5.avg, traintop1.avg, traintop5.avg)


def accuracy(output: Tensor, target: Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = (
                correct[:k]
                .reshape(1, -1)
                .view(-1)
                .float()
                .sum(0, keepdim=True)
            )
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def initialize_cutmix(args, num_classes: int):
    cutmixargs = {}
    cutmixargs["mixup_alpha"] = args.mixup
    cutmixargs["cutmix_alpha"] = args.cutmix
    cutmixargs["cutmix_minmax"] = args.cutmix_minmax
    cutmixargs["prob"] = args.mixup_prob
    cutmixargs["switch_prob"] = args.mixup_switch_prob
    cutmixargs["mode"] = args.mixup_mode
    cutmixargs["correct_lam"] = True
    cutmixargs["label_smoothing"] = args.smoothing
    cutmixargs["num_classes"] = num_classes

    use = args.cutmix > 0 or args.mixup > 0 or args.cutmix_minmax is not None
    mixup_fn = Mixup(**cutmixargs) if use else None
    return mixup_fn


class Timer:
    def __init__(self, trainingsize: int, num_batches: int):
        self.num_batches = trainingsize / num_batches
        self.trainingsize = trainingsize

    def start(self):
        self.starttime = time.time()

    def stop(self):
        self.epoch = time.time() - self.starttime
        self.batch = self.epoch / self.num_batches
        self.image = self.epoch / self.trainingsize


def define_filenames(files_and_paths: SimpleNamespace, arch: str):
    if files_and_paths.checkpoint_dir:
        checkpoint_dir = os.path.join(
            files_and_paths.checkpoint_dir, files_and_paths.exp_name
        )
        checkpoint_file = os.path.join(checkpoint_dir, arch)
    else:
        checkpoint_file = ""

    if files_and_paths.exp_name:
        result_file = os.path.join(
            files_and_paths.result_dir, files_and_paths.exp_name + ".csv"
        )
        meta_file = os.path.join(
            files_and_paths.result_dir, files_and_paths.exp_name + ".meta"
        )
    else:
        result_file = ""
        meta_file = ""

    print(f"Saving checkpoints to {checkpoint_file}")
    print(f"Writing results to {result_file}")
    print(f"Writing metadata to {meta_file}")

    return checkpoint_file, result_file, meta_file


def save_distributions(model, epoch: int, files_and_paths: SimpleNamespace):
    # TODO: Check what has changed in the definition of the model and change accordingly
    expression = re.compile(
        r"blocks\.([0-9]+)\.(xchannel|xpatch)\.([0-9]?)\.?(weight|bias)"
    )

    data = []
    for name, param in model.named_parameters():
        if match := re.search(expression, name):
            depth = match.group(1)
            layer_type = match.group(2)
            layer_index = match.group(3)
            argtype = match.group(4)
            hist = torch.histogram(param.detach().cpu())
            data.append(
                {
                    "depth": depth,
                    "layer_type": layer_type,
                    "layer_index": layer_index,
                    "argtype": argtype,
                    "hist": hist,
                }
            )
    filename = os.path.join(
        files_and_paths.result_dir,
        "distributions",
        files_and_paths.exp_name,
        f"{epoch}.dist",
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pickle.dump(
        data,
        open(
            filename,
            "wb",
        ),
    )


def log(
    result_file: str,
    epoch: int,
    timer: Timer,
    validation_accuracy: ValidationAccuracy,
    train_loss: float,
):
    with open_file_or_stdout(
        result_file,
    ) as fp:
        fp.write(
            f"{epoch}, "
            f"{timer.epoch}, "
            f"{timer.batch}, "
            f"{timer.image}, "
            f"{validation_accuracy.acc1}, "
            f"{validation_accuracy.acc5}, "
            f"{validation_accuracy.trainacc1}, "
            f"{validation_accuracy.trainacc5}, "
            f"{train_loss}\n"
        )


def setup_and_run_training(args: dict):
    system_settings = args["system settings"]
    model_parameter = args["model parameter"]
    pretrained_network = args["pretrained network"]
    learning_parameter = args["learning parameter"]
    files_and_paths = args["files and paths"]
    metadata = args["meta data"]
    mixup_settings = args["mixup settings"]

    torch.manual_seed(system_settings.seed)
    np.random.seed(system_settings.seed)

    distributed_parameters = initialize_distributed()

    arch = f"resmlp{model_parameter.depth}"
    model = ResMLP(
        num_classes=model_parameter.num_classes,
        depth=model_parameter.depth,
        input_shape=[
            3,
            model_parameter.image_size,
            model_parameter.image_size,
        ],
        patch_shape=model_parameter.patch_size,
        embedding_dimension=model_parameter.embedding_dim,
        device=distributed_parameters.device,
    )
    optimizer = create_optimizer(learning_parameter, model)
    lr_scheduler, _ = create_scheduler(learning_parameter, optimizer)
    state = ModelTrainingState(arch, model, optimizer, lr_scheduler)

    # TODO: Figure out what should be the behaviour regarding the consistency of the loaded model and the
    # parsed arguments
    if pretrained_network.pretrained:
        load_pretrained(
            file_dir=pretrained_network.pretrained,
            gpu=distributed_parameters.gpu,
            model=model,
        )
        print(f"pretrained changing head to {model_parameter.num_classes}")
        print(model.head)
        model.head = nn.Linear(
            in_features=args.embedding_dim,
            out_features=model_parameter.num_classes,
        ).cuda(distributed_parameters.gpu)

    trainloader, testloader = generate_data_loader(
        dataset_type=metadata.data_set,
        data_root=files_and_paths.data_dir,
        batch_size=model_parameter.batch_size,
        input_size=model_parameter.image_size,
    )

    loss_function = get_loss_function(
        mixup_settings, distributed_parameters.gpu
    )
    mixup_fn = initialize_cutmix(mixup_settings, model_parameter.num_classes)

    checkpoint_file, result_file, meta_file = define_filenames(
        files_and_paths, arch
    )

    startEpoch = state.epoch + 1

    timer = Timer(len(trainloader), model_parameter.batch_size)

    for epoch in trange(
        startEpoch,
        learning_parameter.epochs,
        position=1,
        desc=f"training model: {arch}",
        disable=metadata.silent,
    ):
        state.epoch = epoch
        if distributed_parameters.distributed:
            trainloader.sample.set_epoch(epoch)

        # TODO: Determine what should be the behavior of a loaded(pretrained) model
        # in case: epoch > model_parameter.sparsity_at_epoch
        if epoch == model_parameter.sparsity_at_epoch:
            set_sparsity(state, model_parameter.sparsity)

        timer.start()

        train_loss = train(
            trainloader,
            state,
            loss_function,
            mixup_fn,
            epoch,
            distributed_parameters.gpu,
            distributed_parameters.distributed,
            metadata.silent,
        )

        timer.stop()

        validation_accuracy = validate(
            testloader,
            trainloader if model_parameter.traintest else None,
            state.model,
            distributed_parameters.gpu,
            distributed_parameters.distributed,
            metadata.silent,
        )

        if distributed_parameters.rank == 0:
            log(result_file, epoch, timer, validation_accuracy, train_loss)

            if state.best_acc1 < validation_accuracy.acc1:
                state.best_acc1 = validation_accuracy.acc1
                save_checkpoint(state, checkpoint_file)

            if (
                files_and_paths.save_distributions != 0
                and epoch % files_and_paths.save_distributions == 0
            ):
                save_distributions(state.model, epoch, files_and_paths)


if __name__ == "__main__":
    parser = GroupedArgumentParser(description="ResMLP Training script")

    files_and_paths = parser.add_argument_group("files and paths")

    files_and_paths.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="Full path to directory for checkpoints",
    )
    files_and_paths.add_argument(
        "--save_distributions",
        type=int,
        default=0,
        help="Save weight and bias distributions every N epochs (0 to disable)",
    )
    files_and_paths.add_argument(
        "--result_dir",
        type=str,
        default="",
        help="Path to where results and metadata is written to",
    )
    files_and_paths.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Path to input data (e.g. ImageNet)",
    )
    files_and_paths.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Experiment's name for meta data",
    )

    pretrained_network = parser.add_argument_group("pretrained network")

    pretrained_network.add_argument(
        "--pretrained",
        type=str,
        default="",
        help="Path to pretrained state dict",
    )
    pretrained_network.add_argument(
        "--pretrained_classes",
        type=int,
        default=1000,
        help="Pretrained model's number of classes. Used to appy state dict (default: 1000)",
    )

    metadata = parser.add_argument_group("meta data")

    metadata.add_argument(
        "--round",
        type=int,
        default=0,
        help="What round of the experiment are we on",
    )

    metadata.add_argument(
        "--data_set",
        type=str,
        default="IMAGENET",
        choices=["IMAGENET", "CIFAR10", "CIFAR100"],
        # nargs=1,
        help="Dataset to be used",
    )
    metadata.add_argument(
        "--silent",
        action="store_true",
        help="disable printing status bar to sdout",
    )

    model_parameter = parser.add_argument_group("model parameter")

    model_parameter.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (default: 32)"
    )
    model_parameter.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Size of input images (default: 224)",
    )
    model_parameter.add_argument(
        "--patch_size", type=int, default=16, help="Patch Size (default: 16)"
    )
    model_parameter.add_argument(
        "--embedding_dim",
        type=int,
        default=384,
        help="Dimension of embedding vectors (default: 384)",
    )
    model_parameter.add_argument(
        "--depth", type=int, default=12, help="Network depth (default: 12)"
    )
    model_parameter.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="Number of targe classes (default: 1000)",
    )
    model_parameter.add_argument(
        "--sparsity",
        type=float,
        default=0.0,
        help="Percentage of activations to drop for backprop."
        " 0 to disable (default: 0)",
    )
    model_parameter.add_argument(
        "--sparsity_at_epoch",
        type=int,
        default=0,
        help="Sparsify training after this epoch" " 0 to disable (default: 0)",
    )
    model_parameter.add_argument(
        "--granularity",
        type=int,
        default=384,
        help="Granularity of sparse structures to apply",
    )
    model_parameter.add_argument(
        "--traintest",
        action="store_true",
        help="Additionally also validate training set w/o augmentation to compare train and test acc",
    )

    learning_parameter = parser.add_argument_group("learning parameter")

    learning_parameter.add_argument(
        "--epochs", type=int, default=100, help="Epochs (default: 100)"
    )
    learning_parameter.add_argument(
        "--weight_decay",
        type=float,
        default=0.2,
        help="Weight decay (default: 0.2)",
    )
    learning_parameter.add_argument(
        "--decay_rate",
        type=float,
        default=0.1,
        help="LR decay rate (default: 0.1)",
    )
    learning_parameter.add_argument(
        "--lr", type=float, default=5e-3, help="Learning rate (default: 5e-3)"
    )
    learning_parameter.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="Minimal learning rate for scheduler (default: 1e-5)",
    )
    learning_parameter.add_argument(
        "--warmup_lr",
        type=float,
        default=1e-6,
        help="Warmup learning rate(default: 1e-5)",
    )
    learning_parameter.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Warmup epochs for the scheduler(default: 5)",
    )
    learning_parameter.add_argument(
        "--cooldown_epochs",
        type=int,
        default=10,
        help="Cooldown epochs for the scheduler(default: 10)",
    )
    learning_parameter.add_argument(
        "--sched",
        type=str,
        default="cosine",
        help="LR scheduler used (default: cosine)",
    )

    learning_parameter.add_argument(
        "--opt",
        type=str,
        default="FusedLAMB",
        help="optimizer to use (default: FusedLAMB)",
    )
    learning_parameter.add_argument(
        "--momentum", type=float, default=5e-3, help="Momentum (default: 5e-3)"
    )

    mixup_settings = parser.add_argument_group("mixup settings")

    mixup_settings.add_argument(
        "--mixup",
        type=float,
        default=0.8,
        help="Alpha value for mixup, set to 0 for no mixup (default: 0.8)",
    )
    mixup_settings.add_argument(
        "--cutmix",
        type=float,
        default=1.0,
        help="Alpha value for cutmix, set to 0 for no mixup (default: 1.0)",
    )
    mixup_settings.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="Cutmix min.max ratio, overrides alpha values and enalbes cutmix set to 0 for no mixup (default: None)",
    )
    mixup_settings.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability for performing cutmix/mixup, when one/both is/are enabled (default: 1.0)",
    )
    mixup_settings.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of swtiching from cutmix to mixup when both are enabled (default: 0.5)",
    )
    mixup_settings.add_argument(
        "--mixup-mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix. Per "batch", "pair", or "element" (default: "batch")',
    )
    mixup_settings.add_argument(
        "--mixup_off_epoch",
        type=int,
        default=0,
        help="Turn off mixup after this epoch, disabled if 0 (default: 0)",
    )
    mixup_settings.add_argument(
        "--smoothing",
        type=float,
        default=0.1,
        help="Label smoothing (default= 0.1)",
    )

    system_settings = parser.add_argument_group("system settings")

    system_settings.add_argument(
        "--deterministic",
        action="store_true",
        help="fix seed for pytorch and numpy (also sets CUDNN mode to deterministic which may lower performance)",
    )
    system_settings.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Manual seed, ignored if --deterministic is unset (default: 0)",
    )
    system_settings.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loader",
    )

    parser.set_defaults(deterministic=False)
    parser.set_defaults(silent=False)
    parser.set_defaults(traintest=False)
    args = parser.parse_args()

    setup_and_run_training(args)

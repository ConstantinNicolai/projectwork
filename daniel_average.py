import torch
from torch.distributed import barrier, all_reduce


class Average(object):
    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def __str__(self):
        ret = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return ret.format(**self.__dict__)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self, distributed: bool, gpu: int) -> None:
        """
        Synchronize average across nodes and processes.
        nccl backend only supports gpu communication -> first wite to cuda
        tensor then do all redure
        """
        if distributed:
            device = torch.device("cuda", gpu)
            box = torch.tensor(
                [self.sum, self.count], dtype=torch.float64, device=device
            )
            barrier()
            all_reduce(box)
            self.sum = float(box[0])
            self.count = int(box[1])
            self.avg = self.sum / self.count
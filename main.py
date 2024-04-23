import argparse
import logging
import os
import resource
from contextlib import suppress


class ConfigParser(argparse.Namespace):
    import configargparse

    class CommandParser(configargparse.ArgumentParser):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.command = None

        def acton(self, cfg):
            self.command = cfg(self)
            return self

    def __init__(self, cmd=None):
        import train
        import merge
        import buildcfg

        super().__init__()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command', help='sub-command help', required=True,
                                           parser_class=self.CommandParser)
        subparsers.add_parser('train', aliases=['test']).acton(train.ConfigCommand)
        subparsers.add_parser('merge').acton(merge.ConfigCommand)
        subparsers.add_parser('buildcfg', aliases=['cfg']).acton(buildcfg.ConfigCommand)

        args, argv = parser.parse_known_args(cmd, namespace=self)
        cmd_parser = subparsers.choices.get(args.command)
        args.command = cmd_parser.command


class SetupEnvironment:
    def __call__(self, args: ConfigParser):
        print(args)
        # if argv := args.argv:
        #     print('unrecognized arguments: %s' % ' '.join(argv), file=sys.stderr)
        return args.command(args)

    # A function to set up the running environment for the training
    def __init__(self, cudaMallocAsync=True):
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except ValueError:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft < hard:
                resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

        if cudaMallocAsync:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync'

        import numpy as np
        import torch

        with suppress(ImportError):
            from torch.backends import cuda, cudnn
            # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
            # in PyTorch 1.12 and later.
            cuda.matmul.allow_tf32 = True
            # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
            cudnn.allow_tf32 = True

        torch.set_default_dtype(torch.float32)
        if not torch.cuda.is_available():
            logging.getLogger(__name__).warning('CUDA is not available. Using CPU instead.')

        # Set the seed for generating random numbers.
        np.random.seed(np.bitwise_xor(*np.atleast_1d(np.asarray(torch.seed(), dtype=np.uint64)).view(np.uint32)).item())

        import pyximport
        pyximport.install()


if __name__ == "__main__":
    SetupEnvironment()(ConfigParser())  # command(config_parser())

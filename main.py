import argparse
import io
import logging
import os
from contextlib import suppress, nullcontext, redirect_stdout
from io import StringIO
from typing import Union, Text


class TextIO(str):
    def __new__(cls, *args, **kw):
        obj = super().__new__(cls, *args, **kw)
        obj.file = StringIO()
        return obj

    @staticmethod
    def open(s, *args, **kwargs):
        return nullcontext(s.file) if isinstance(s, TextIO) else open(s, *args, **kwargs)


class ConfigParser(argparse.Namespace):
    import configargparse

    class CommandParser(configargparse.ArgumentParser):
        def __init__(self, *args, **kwargs):
            self.config_dict = dict()
            self.command = None
            self.argv = None
            super().__init__(*args, config_file_open_func=TextIO.open, **kwargs)

        def get_parser_cfg(self, args, header):
            s: Union[TextIO, Text] = TextIO(header)
            with s.file as f, StringIO(f'[{header}]  # ') as h, redirect_stdout(h):
                h.seek(0, io.SEEK_END)
                self.write_config_file(args, (s,))
                return h.getvalue() + f.getvalue()

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
        self.command: str | ConfigParser.CommandParser = subparsers.choices[args.command]
        self.command.argv = argv


class SetupEnvironment:
    def __call__(self, args: ConfigParser):
        command_parser = args.command
        del args.command
        argv = command_parser.argv
        try:
            return command_parser.command(args, argv=argv)
        except TypeError:
            if argv:
                command_parser.error('unrecognized arguments: %s' % ' '.join(argv))
            return command_parser.command(args)

    # A function to set up the running environment for the training
    def __init__(self, cudaMallocAsync=True):
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
            # The performance might improve if the benchmarking feature is enabled
            cudnn.benchmark = True

        torch.set_default_dtype(torch.float32)
        if not torch.cuda.is_available():
            logging.getLogger(__name__).warning('CUDA is not available. Using CPU instead.')

        # Set the seed for generating random numbers.
        np.random.seed(np.bitwise_xor(*np.atleast_1d(np.asarray(torch.seed(), dtype=np.uint64)).view(np.uint32)).item())

        import pyximport
        pyximport.install()


if __name__ == "__main__":
    SetupEnvironment()(ConfigParser())  # command(config_parser())

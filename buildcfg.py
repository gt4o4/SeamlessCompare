import tempfile
from io import StringIO
from pathlib import Path

import configargparse

import merge
import train


def get_parser_cfg(parser, args):
    with tempfile.NamedTemporaryFile(mode='r') as f, StringIO() as s:
        parser.write_config_file(args, (f.name,))
        print('[Merge Configuration File]', file=s)
        print(f.read(), file=s)
        return s.getvalue()


class ConfigCommand:
    def __init__(self, parser):
        parser.add_argument('--source', type=Path, help='source config file path')
        parser.add_argument('--target', type=Path, help='target config file path')
        merge.ConfigCommand(parser)
        self.parser = parser

    def __call__(self, args):
        source_cmd = train.ConfigCommand(parser := configargparse.ArgumentParser())
        source_args = parser.parse_args(args=(), config_file_contents=args.source.read_text())
        # train_cmd(train_args)
        merge_args, argv = self.parser.parse_known_args(
            args=(), config_file_contents=get_parser_cfg(parser, source_args))
        print(get_parser_cfg(self.parser, merge_args))

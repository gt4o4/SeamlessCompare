import os
from pathlib import Path

import merge
import train


def find_ckpt(basedir, expname):
    logfolder = Path(basedir, expname)
    ret = sorted(logfolder.glob(f'**/{expname}.th'), key=os.path.getmtime, reverse=True)
    return ret[0] if ret else None


class ConfigCommand:
    def __init__(self, parser):
        parser.add_argument('source', type=Path, help='source config file path')
        parser.add_argument('target', type=Path, help='target config file path')
        merge.ConfigCommand(parser.add_argument_group('Merge', 'Merge config files'))
        self.parser = parser

    def __call__(self, args):
        merge_parser = type(self.parser)().acton(merge.ConfigCommand)
        merge_args = merge_parser.build_args_command(merge_parser.parse_args(
            args=(), config_file_contents=self.parser.get_parser_cfg(args, 'Merge')))

        config_source = self.train_ckpt_config(args.source.read_text(), 'Source')
        config_target = self.train_ckpt_config(args.target.read_text(), 'Target')

        merge_args, argv = merge_parser.parse_known_args(args=merge_args, config_file_contents=config_source)
        print(merge_parser.get_parser_cfg(merge_args, 'Merge'))

    def train_ckpt_config(self, config_contents, header):
        parser = type(self.parser)().acton(train.ConfigCommand)
        source_args = parser.parse_args(args=(), config_file_contents=config_contents)

        if not source_args.ckpt:
            source_args.ckpt = find_ckpt(source_args.basedir, source_args.expname)
        source_args.render_only = source_args.render_test = source_args.ckpt is not None
        source_args.render_train = source_args.render_path = False

        parser.command(source_args := parser.parse_args(args=parser.build_args_command(source_args)))
        return parser.get_parser_cfg(source_args, header)

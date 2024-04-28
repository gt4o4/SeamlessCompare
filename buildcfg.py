import os
from pathlib import Path

import merge
import train


def find_ckpt(basedir, expname):
    logfolder = Path(basedir, expname)
    ret = sorted(logfolder.glob(f'**/{expname}.th'), key=os.path.getmtime, reverse=True)
    return ret[0] if ret else None


def build_args_command(parser, source_args):
    new_args = []
    for a in parser._actions:
        new_val = getattr(source_args, a.dest, None) if a.dest is not None else None
        if new_val is not None and a.default != new_val and str(a.default) != str(new_val):
            k, *_ = parser.get_possible_config_keys(a)
            try:
                arg = parser.convert_item_to_command_line_arg(a, k, new_val)
            except (AssertionError, ValueError):
                arg = parser.convert_item_to_command_line_arg(a, k, str(new_val))
            new_args.extend(arg)
    return new_args


class ModelConfiger:
    def __init__(self, parser_cls):
        self.parser_cls = parser_cls

    def train_ckpt_config(self, config_contents, header):
        parser = self.parser_cls()
        train.ConfigCommand(parser)
        source_args = parser.parse_args(args=(), config_file_contents=config_contents)
        if not source_args.ckpt:
            source_args.ckpt = find_ckpt(source_args.basedir, source_args.expname)
        source_args.render_only = source_args.render_test = source_args.ckpt is not None
        source_args.render_train = source_args.render_path = False
        args = build_args_command(parser, source_args)

        parser = self.parser_cls()
        source_cmd = train.ConfigCommand(parser)
        parser_args = parser.parse_args(args)
        output_config = parser.get_parser_cfg(parser_args, header)

        source_cmd(parser_args)
        return output_config


class ConfigCommand:
    def __init__(self, parser):
        parser.add_argument('source', type=Path, help='source config file path')
        parser.add_argument('target', type=Path, help='target config file path')
        merge.ConfigCommand(parser.add_argument_group('Merge', 'Merge config files'))
        self.parser = parser

    def __call__(self, args):
        parser_cls = type(self.parser)
        merge_parser = parser_cls()
        merge_cmd = merge.ConfigCommand(merge_parser)
        merge_args = merge_parser.parse_args(args=(), config_file_contents=self.parser.get_parser_cfg(args, 'Merge'))
        merge_args = build_args_command(merge_parser, merge_args)

        configer = ModelConfiger(parser_cls)
        config_source = configer.train_ckpt_config(args.source.read_text(), 'Source')
        config_target = configer.train_ckpt_config(args.target.read_text(), 'Target')
        merge_args, argv = self.parser.parse_known_args(args=(), config_file_contents=config_source)
        print(self.parser.get_parser_cfg(merge_args, 'Merge'))

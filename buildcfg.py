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
    for c, l in parser.get_source_to_settings_dict().items():
        for k, (a, v) in l.items():
            new_val = getattr(source_args, a.dest, None) if a.dest is not None else None
            def_val = parser.get_default(k)
            if def_val is not None and str(def_val) != str(new_val):
                try:
                    arg = parser.convert_item_to_command_line_arg(a, k, new_val)
                except (AssertionError, ValueError):
                    arg = parser.convert_item_to_command_line_arg(a, k, str(new_val))
                new_args.extend(arg)
    return new_args


class ModelConfiger:
    def __init__(self, parser_cls):
        self.parser = parser_cls()
        self.cmd = train.ConfigCommand(self.parser)

    def train_ckpt_config(self, config_contents):
        source_args = self.parser.parse_args(args=(), config_file_contents=config_contents)
        if not source_args.ckpt:
            source_args.ckpt = find_ckpt(source_args.basedir, source_args.expname)
        source_args.render_only = source_args.render_test = source_args.ckpt is not None
        source_args.render_train = source_args.render_path = False
        self.cmd(source_args)
        return build_args_command(self.parser, source_args)


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
        config_source = configer.train_ckpt_config(args.source.read_text())
        config_target = configer.train_ckpt_config(args.target.read_text())
        merge_args, argv = self.parser.parse_known_args(args=(), config_file_contents=config_source)
        print(self.parser.get_parser_cfg(merge_args, 'Merge'))

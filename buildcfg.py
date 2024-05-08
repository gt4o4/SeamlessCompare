import operator
import os
import shutil
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

import merge
import train


def find_ckpt(basedir, expname):
    logfolder = Path(basedir, expname)
    ret = sorted(logfolder.glob(f'**/{expname}.th'), key=os.path.getmtime, reverse=True)
    return ret[0] if ret else None


def copy_ckpt(source_ckpt, target_ckpt):
    return target_ckpt if target_ckpt.exists() else shutil.copyfile(source_ckpt, target_ckpt)


def generate_expname(source_name, target_name):
    common_match = SequenceMatcher(
        operator.methodcaller('isspace'), source_name, target_name).find_longest_match()
    expname = source_name[common_match.a:common_match.a + common_match.size].strip().strip('_') if \
        common_match.size else f"{source_name.strip().rstrip('_')}_{target_name.strip().strip('_')}"
    return f"{expname}_merge"


class ConfigCommand:
    def __init__(self, parser):
        parser.add_argument('source', type=Path, help='source config file path')
        parser.add_argument('target', type=Path, help='target config file path')
        merge.ConfigCommand(parser.add_argument_group('Merge', 'Merge config files'))
        self.parser = parser

    def __call__(self, args):
        source_cfg = self.train_ckpt_config(args.source.read_text(), 'Source')
        target_cfg = self.train_ckpt_config(args.target.read_text(), 'Target')
        merge_cfg = self.merge_ckpt_config(source_cfg, target_cfg, args, 'Merge')

        parser = type(self.parser)().acton(merge.ConfigCommand)
        merge_args = parser.parse_args(args=self.parser.build_args_command(args), config_file_contents=merge_cfg)
        merge_dir = Path(merge_args.basedir, merge_args.expname)

        with (merge_dir / f'{merge_args.expname}.txt').open(mode='w') as f:
            print(parser.get_parser_cfg(merge_args, 'Merge'), file=f)
            print('; '.join(source_cfg.splitlines(keepends=True)), file=f)
            print('; '.join(target_cfg.splitlines(keepends=True)), file=f)
            print('#', datetime.now(), file=f)

        return parser.command(merge_args)

    def train_ckpt_config(self, config_contents, header=None):
        parser = type(self.parser)().acton(train.ConfigCommand)
        source_args = parser.parse_args(args=(), config_file_contents=config_contents)

        if not source_args.ckpt:
            source_args.ckpt = find_ckpt(source_args.basedir, source_args.expname)
        source_args.render_only = source_args.render_test = source_args.ckpt is not None
        source_args.render_train = source_args.render_path = False

        parser.command(source_args := parser.parse_args(args=parser.build_args_command(source_args)))
        return parser.get_parser_cfg(source_args, header) if header else source_args

    def merge_ckpt_config(self, source_cfg, target_cfg, merge_args, header=None):
        parser = type(self.parser)().acton(merge.ConfigCommand)
        ignore_set = {'ckpt', 'expname'}

        source_args, _ = parser.parse_known_args(args=self.parser.build_args_command(merge_args),
                                                 config_file_contents=source_cfg)
        target_args, _ = parser.parse_known_args(args=parser.build_args_command(SimpleNamespace(**{
            k: v for k, v in source_args._get_kwargs() if k not in ignore_set})), config_file_contents=target_cfg)

        target_args.expname = generate_expname(source_args.expname, target_args.expname)

        merge_dir = Path(target_args.basedir, target_args.expname)
        merge_dir.mkdir(parents=True, exist_ok=True)
        copy_ckpt(source_args.ckpt, merge_dir / f'{target_args.expname}.th')
        target_args.ckpt = copy_ckpt(target_args.ckpt, merge_dir / os.path.basename(target_args.ckpt))

        merge_args = parser.parse_args(args=parser.build_args_command(target_args))
        return parser.get_parser_cfg(merge_args, header) if header else merge_args

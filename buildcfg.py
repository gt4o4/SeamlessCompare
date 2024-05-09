import json
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
    prefix = source_name[common_match.a:common_match.a + common_match.size].strip().strip('_') if \
        common_match.size else f"{source_name.strip().rstrip('_')}_{target_name.strip().strip('_')}"
    return f"{prefix}_merge", prefix


def build_transform(source_aabb, target_aabb, source_trans, target_trans):
    return vars(source_trans), vars(target_trans)


class ConfigCommand:
    def __init__(self, parser):
        parser.add_argument('source', type=Path, help='source config file path')
        parser.add_argument('target', type=Path, help='target config file path')
        merge.ConfigCommand(parser.add_argument_group('Merge', 'Merge config files'))
        self.parser = parser

    def __call__(self, args):
        merge_cfg, source_args, target_args = self.merge_ckpt_config(
            source_cfg=self.train_ckpt_config(source_cfg := args.source.read_text(), dryrun=True, header='Source'),
            target_cfg=self.train_ckpt_config(target_cfg := args.target.read_text(), dryrun=True, header='Target'),
            merge_args=self.parser.build_args_command(args), header='Merge')

        parser = type(self.parser)().acton(merge.ConfigCommand)
        merge_args = parser.parse_args(args=self.parser.build_args_command(args), config_file_contents=merge_cfg)
        merge_dir = Path(merge_args.basedir, merge_args.expname)

        source_cfg = self.train_ckpt_config(source_cfg, source_args, header='Source')
        target_cfg = self.train_ckpt_config(target_cfg, target_args, header='Target')
        with (merge_dir / f'{merge_args.expname}.txt').open(mode='w') as f:
            print(parser.get_parser_cfg(merge_args, 'Merge'), file=f)
            print('; '.join(source_cfg.splitlines(keepends=True)), file=f)
            print('; '.join(target_cfg.splitlines(keepends=True)), file=f)
            print('#', datetime.now(), file=f)

        return parser.command(merge_args)

    def train_ckpt_config(self, config_contents, config_args=(), header=None, dryrun=False):
        parser = type(self.parser)().acton(train.ConfigCommand)
        source_args = parser.parse_args(args=config_args, config_file_contents=config_contents)

        if not source_args.ckpt:
            source_args.ckpt = find_ckpt(source_args.basedir, source_args.expname)

        source_args.render_only = source_args.ckpt is not None
        source_args.render_train = source_args.render_test = source_args.render_path = False

        # if dryrun and source_args.transform and not source_args.render_only:
        #     source_args.transform.ns = None

        evaluator = parser.command(source_args)
        if not source_args.render_only and (new_ckpt := find_ckpt(source_args.basedir, source_args.expname)):
            source_args.ckpt = new_ckpt

        if not source_args.at_least_aabb:
            source_args.at_least_aabb = evaluator.tensorf.aabb.flatten().tolist()

        source_args = parser.build_args_command(source_args)
        if header:
            source_args = parser.get_parser_cfg(parser.parse_args(args=source_args), header)
        return source_args

    def merge_ckpt_config(self, source_cfg, target_cfg, merge_args, header=None):
        parser = type(self.parser)().acton(merge.ConfigCommand)
        ignore_set = {'ckpt', 'expname', 'at_least_aabb'}

        source_args, _ = parser.parse_known_args(args=merge_args, config_file_contents=source_cfg)
        merge_args, _ = parser.parse_known_args(args=parser.build_args_command(SimpleNamespace(**{
            k: v for k, v in source_args._get_kwargs() if k not in ignore_set})), config_file_contents=target_cfg)

        merge_args.expname, prefix = generate_expname(source_args.expname, merge_args.expname)

        merge_dir = Path(merge_args.basedir, merge_args.expname)
        merge_dir.mkdir(parents=True, exist_ok=True)

        source_args, target_args = self.transform_ckpt_config(
            source_cfg, target_cfg, merge_args, merge_dir / f'{prefix}_transforms.json')

        merge_args = parser.build_args_command(merge_args)
        if header:
            merge_args = parser.get_parser_cfg(parser.parse_args(args=merge_args), header)
        return merge_args, source_args, target_args

    def transform_ckpt_config(self, source_cfg, target_cfg, merge_args, filename):
        parser = type(self.parser)().acton(train.ConfigCommand)

        source_args, _ = parser.parse_known_args(args=(), config_file_contents=source_cfg)
        target_args, _ = parser.parse_known_args(args=(), config_file_contents=target_cfg)

        source_trans, target_trans = build_transform(
            source_args.at_least_aabb, target_args.at_least_aabb,
            *source_args.transform.get(source_args.expname, target_args.expname))

        source_name = Path(source_args.datadir).stem
        target_name = Path(target_args.datadir).stem
        prefix = os.path.commonprefix((source_name, target_name))
        with open(filename := os.fspath(filename), mode='w') as f:
            json.dump({
                source_name.removeprefix(prefix): source_trans,
                target_name.removeprefix(prefix): target_trans}, f, indent=4)

        # copy_ckpt(source_args.ckpt, merge_dir / PurePath(os.path.basename(merge_args.datadir)).with_suffix('.th'))
        # merge_args.ckpt = copy_ckpt(merge_args.ckpt, merge_dir / os.path.basename(merge_args.ckpt))
        source_args.expname = merge_args.expname
        source_args.basedir = target_args.basedir = merge_args.basedir
        source_args.transform = target_args.transform = merge_args.transform = filename
        source_args.ckpt = target_args.ckpt = target_args.at_least_aabb = None

        return parser.build_args_command(source_args), parser.build_args_command(target_args)

import json
import operator
import os
import shutil
from collections import namedtuple
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path, PurePath
from types import SimpleNamespace

import numpy as np
import torch

import merge
import train
from models import TransformFile

Vector3 = namedtuple('Vector3', ('x', 'y', 'z'))


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
        common_match.size >= 3 else f"{source_name.strip().rstrip('_')}_{target_name.strip().strip('_')}"
    return f"{prefix}_merge", prefix


def boundingBox(low: Vector3, high: Vector3):
    corner1 = Vector3(low.x, low.y, low.z)
    corner2 = Vector3(high.x, low.y, low.z)
    corner3 = Vector3(low.x, high.y, low.z)
    corner4 = Vector3(low.x, low.y, high.z)
    corner5 = Vector3(high.x, high.y, low.z)
    corner6 = Vector3(high.x, low.y, high.z)
    corner7 = Vector3(low.x, high.y, high.z)
    corner8 = Vector3(high.x, high.y, high.z)

    return corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8


class ConfigCommand:
    def __init__(self, parser):
        parser.add_argument('source', type=Path, help='source config file path')
        parser.add_argument('target', type=Path, help='target config file path')
        merge.ConfigCommand(parser.add_argument_group('Merge', 'Merge config files'))
        self.parser = parser

    def __call__(self, args):
        target_cfg = self.train_ckpt_config(args.target.read_text(), dryrun=True, header='Target')
        source_cfg = self.train_ckpt_config(None, self.check_aabb(args.source.read_text(), target_cfg), header='Source')
        merge_cfg = self.merge_ckpt_config(source_cfg, target_cfg,
                                           merge_args=self.parser.build_args_command(args), header='Merge')

        parser = type(self.parser)().acton(merge.ConfigCommand)
        merge_args = parser.parse_args(args=self.parser.build_args_command(args), config_file_contents=merge_cfg)
        merge_dir = Path(merge_args.basedir, merge_args.expname)

        with (merge_dir / f'{merge_args.expname}.txt').open(mode='w') as f:
            print(parser.get_parser_cfg(merge_args, 'Merge'), file=f)
            print('; '.join(source_cfg.splitlines(keepends=True)), file=f)
            print('; '.join(target_cfg.splitlines(keepends=True)), file=f)
            print('#', datetime.now(), file=f)

        return parser.command(merge_args)

    def train_ckpt_config(self, config_contents, config_args=(), header=None, dryrun=False):
        parser = type(self.parser)().acton(train.ConfigCommand)
        args = parser.parse_args(args=config_args, config_file_contents=config_contents)

        if not args.ckpt:
            args.ckpt = find_ckpt(args.basedir, args.expname)

        args.render_only = args.ckpt is not None
        args.render_train = args.render_test = args.render_path = False

        if dryrun and args.transform and not args.render_only:
            transform_type, _ = args.transform.get(Path(args.datadir).stem, PurePath(str(args.transform)).stem)
            transform_type.rot = transform_type.trans = None

        evaluator = parser.command(args)
        if not args.render_only and (new_ckpt := find_ckpt(args.basedir, args.expname)):
            args.ckpt = new_ckpt

        if not args.at_least_aabb:
            args.at_least_aabb = evaluator.tensorf.aabb.flatten().tolist()
        else:
            at_least_aabb = torch.as_tensor(args.at_least_aabb, device=evaluator.tensorf.aabb.device,
                                            dtype=evaluator.tensorf.aabb.dtype).reshape(2, 3)
            # assert evaluator.tensorf.aabb[0].le(at_least_aabb[0]).all() and \
            #        evaluator.tensorf.aabb[1].ge(at_least_aabb[1]).all(), 'Invalid at_least_aabb'

        args = parser.build_args_command(args)
        if header:
            args = parser.get_parser_cfg(parser.parse_args(args=args), header)
        return args

    def merge_ckpt_config(self, source_cfg, target_cfg, merge_args, header=None):
        parser = type(self.parser)().acton(merge.ConfigCommand)
        ignore_set = {'ckpt', 'expname', 'at_least_aabb'}

        source_args, _ = parser.parse_known_args(args=merge_args, config_file_contents=source_cfg)
        merge_args, _ = parser.parse_known_args(args=parser.build_args_command(SimpleNamespace(**{
            k: v for k, v in source_args._get_kwargs() if k not in ignore_set})), config_file_contents=target_cfg)

        expname, prefix = generate_expname(source_args.expname, merge_args.expname)

        merge_dir = Path(merge_args.basedir, expname)
        merge_dir.mkdir(parents=True, exist_ok=True)

        merge_args.transform = self.transform_ckpt_config(source_args.expname, merge_args,
                                                          merge_dir / f'{prefix}_transforms.json')
        merge_args.expname = expname

        # merger = parser.command.get_merger(merge_args)
        # if merge_args.transform is not None:
        #     merge_args.matrix = None

        merge_args = parser.build_args_command(merge_args)

        if header:
            merge_args = parser.get_parser_cfg(parser.parse_args(args=merge_args), header)
        return merge_args

    def transform_ckpt_config(self, source_name, merge_args, filename):
        target_name = merge_args.expname
        prefix = os.path.commonprefix((source_name, target_name))
        strip = '_'

        if merge_args.matrix:
            return None
        elif merge_args.transform:
            source_trans, target_trans = merge_args.transform.get(source_name, target_name)
            del target_trans.scale
            trdict = {source_name.removeprefix(prefix).strip(strip): vars(source_trans),
                      target_name.removeprefix(prefix).strip(strip): vars(target_trans)}
        else:
            trdict = {target_name.removeprefix(prefix).strip(strip): {}}

        with open(filename, mode='w') as f:
            json.dump(trdict, f, indent=4)

        # copy_ckpt(source_args.ckpt, merge_dir / PurePath(os.path.basename(merge_args.datadir)).with_suffix('.th'))
        # merge_args.ckpt = copy_ckpt(merge_args.ckpt, merge_dir / os.path.basename(merge_args.ckpt))
        # source_args.expname = merge_args.expname
        # source_args.basedir = target_args.basedir = merge_args.basedir
        # source_args.transform = target_args.transform = merge_args.transform = TransformFile(filename)
        # source_args.ckpt = target_args.ckpt = target_args.at_least_aabb = None

        return TransformFile(filename)

    def check_aabb(self, source_cfg, target_cfg):
        parser = type(self.parser)().acton(train.ConfigCommand)

        source_args, _ = parser.parse_known_args(args=(), config_file_contents=source_cfg)
        target_args, _ = parser.parse_known_args(args=(), config_file_contents=target_cfg)

        if source_args.transform:
            _, tgt_trans = source_args.transform.get(source_args.expname, target_args.expname)
            aabb = np.asarray(boundingBox(Vector3(*target_args.at_least_aabb[:3]),
                                          Vector3(*target_args.at_least_aabb[3:]))).reshape(-1, 3)
            aabb = np.hstack((aabb, np.ones((aabb.shape[0], 1))))
            aabb = np.matmul(aabb, tgt_trans.matrix().T)[..., :3]
            aabb_min, aabb_max = np.min(aabb, axis=0), np.max(aabb, axis=0)

            print('at_least_aabb = ', at_least_aabb := aabb_min.tolist() + aabb_max.tolist())
        else:
            at_least_aabb = target_args.at_least_aabb
            aabb_min, aabb_max = at_least_aabb[:3], at_least_aabb[3:]

        if aabb_ref := source_args.at_least_aabb:
            print('cur_aabb = ', aabb_ref[0].tolist() + aabb_ref[1].tolist())
            aabb_min = np.minimum(aabb_ref[0], aabb_min)
            aabb_max = np.maximum(aabb_ref[1], aabb_max)
            print('mix_aabb = ', aabb_min.tolist() + aabb_max.tolist())

        source_args.at_least_aabb = at_least_aabb
        return parser.build_args_command(source_args)

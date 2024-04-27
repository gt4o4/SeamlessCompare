import os
from pathlib import Path


def find_ckpt(basedir, expname):
    logfolder = Path(basedir, expname)
    ret = sorted(logfolder.glob(f'**/{expname}.th'), key=os.path.getmtime, reverse=True)
    return ret[0] if ret else None


def get_ckpt_config(config_contents, header, parser_cls):
    import train
    source_cmd = train.ConfigCommand(parser := parser_cls())
    source_args = parser.parse_args(args=(), config_file_contents=config_contents)
    if not source_args.ckpt:
        source_args.ckpt = find_ckpt(source_args.basedir, source_args.expname)
    source_args.render_only = source_args.render_test = source_args.ckpt is not None
    source_args.render_train = source_args.render_path = False
    source_cmd(source_args)
    return parser.get_parser_cfg(source_args, header)


class ConfigCommand:
    def __init__(self, parser):
        import merge
        parser.add_argument('--source', type=Path, help='source config file path')
        parser.add_argument('--target', type=Path, help='target config file path')
        merge.ConfigCommand(parser)
        self.parser = parser

    def __call__(self, args):
        parser_cls = type(self.parser)
        config_source = get_ckpt_config(args.source.read_text(), 'Source', parser_cls)
        config_target = get_ckpt_config(args.target.read_text(), 'Target', parser_cls)
        merge_args, argv = self.parser.parse_known_args(args=(), config_file_contents=config_source)
        print(self.parser.get_parser_cfg(merge_args, 'Merge'))

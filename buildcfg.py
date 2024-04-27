from pathlib import Path

import merge
import train


class ConfigCommand:
    def __init__(self, parser):
        parser.add_argument('--source', type=Path, help='source config file path')
        parser.add_argument('--target', type=Path, help='target config file path')
        merge.ConfigCommand(parser)
        self.parser = parser

    def __call__(self, args):
        source_cmd = train.ConfigCommand(parser := type(self.parser)())
        source_args = parser.parse_args(args=(), config_file_contents=args.source.read_text())
        # train_cmd(train_args)
        merge_args, argv = self.parser.parse_known_args(
            args=(), config_file_contents=parser.get_parser_cfg(source_args, '[Train Configuration File]'))
        print(self.parser.get_parser_cfg(merge_args, '[Merge Configuration File]'))

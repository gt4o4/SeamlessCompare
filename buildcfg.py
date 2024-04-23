import tempfile

import merge


class ConfigCommand(merge.ConfigCommand):
    def __init__(self, parser):
        super().__init__(parser)
        self.parser = parser

    def __call__(self, args):
        with tempfile.NamedTemporaryFile(mode='r') as f:
            self.parser.write_config_file(args, (f.name,))
            print(f.read())

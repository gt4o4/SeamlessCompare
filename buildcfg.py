import tempfile

import merge


class ConfigCommand:
    def __init__(self, parser):
        merge.ConfigCommand(parser)
        self.parser = parser

    def __call__(self, args, argv):
        with tempfile.NamedTemporaryFile(mode='r') as f:
            self.parser.write_config_file(args, (f.name,))
            print('[Merge Configuration File]')
            print(f.read())

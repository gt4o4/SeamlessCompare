import numpy as np
import torch

from engine.trainer import Trainer
from utils.opt import config_parser
from utils.fs import setup_wd


def main(parser):
    args, _ = parser.parse_known_args()
    print('args =', args, end='\n\n')

    run_dir, ckpt_dir, tb_dir = setup_wd(parser, args)

    trainer = Trainer(args, run_dir, ckpt_dir, tb_dir)
    
    if args.export_mesh:
        trainer.export_mesh()

    if args.render_only and (args.render_train or args.render_test or args.render_path):
        trainer.render_test(trainer.build_network())
    else:
        trainer.train()


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    np.random.seed(np.bitwise_xor(*np.atleast_1d(np.asarray(torch.seed(), dtype=np.uint64)).view(np.uint32)).item())

    main(config_parser())

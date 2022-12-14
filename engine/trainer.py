import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    pass
from tqdm import trange, tqdm

from data import dataset_dict
from models import MODEL_ZOO
from models.loss import TVLoss
from engine.eval import evaluation, evaluation_path
from utils.recon import convert_sdf_samples_to_ply
from utils.render import chunkify_render, N_to_reso, cal_n_samples
from utils.fs import seek_checkpoint
from utils.color import sort_palette


class SimpleSampler:
    def __init__(self, train_dataset, batch):
        total = train_dataset.all_rays.shape[0]
        w, h = train_dataset.img_wh
        self.dataset = train_dataset
        self.batch = batch
        self.curr = total
        self.ids = np.random.permutation(total)

    def apply_filter(self, func, *args, **kwargs):
        mask_filtered = func(self.dataset.all_rays[self.ids], *args, **kwargs)
        self.ids = self.ids[mask_filtered]
        self.curr = self.ids.shape[0]

    def nextids(self):
        total = self.ids.shape[0]
        self.curr += self.batch
        if self.curr + self.batch > total:
            np.random.shuffle(self.ids)
            self.curr = 0
        return self.ids[self.curr:self.curr + self.batch]

    def getbatch(self, device):
        ids = self.nextids()
        return self.dataset.all_rays[ids].to(device), self.dataset.all_rgbs[ids].to(device)


class Trainer:
    def __init__(self, args, run_dir, ckpt_dir, tb_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.renderer = chunkify_render

        self.args = args
        self.optimizer = None
        self.summary_writer = None
        self.trainingSampler = None
        self.logger = logging.getLogger(type(self).__name__)

        self.run_dir = run_dir
        self.ckpt_dir = ckpt_dir
        self.tb_dir = tb_dir

        # init dataset
        dataset = dataset_dict[args.dataset_name]
        self.train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
        self.test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)

        # init parameters
        self.aabb = self.train_dataset.scene_bbox.to(self.device)
        self.reso_cur = N_to_reso(args.N_voxel_init, self.aabb)
        self.reso_mask = None
        self.nSamples = min(args.nSamples, cal_n_samples(self.reso_cur, args.step_ratio))
        self.palette = self.build_palette(args.palette_path, args.shadingMode == 'PLT_AlphaBlend')

        print("[trainer init] aabb", self.aabb.tolist())
        print("[trainer init] num of render samples", self.nSamples)
        print("[trainer init] palette shape", self.palette.shape)

        # linear in logrithmic space
        self.N_voxel_list = torch.round(torch.exp(torch.linspace(
            np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(args.upsamp_list) + 1))).long().tolist()[1:]
        
        # loss function
        self.tvreg = TVLoss()

        self.Ortho_reg_weight = args.Ortho_weight
        print("[trainer init] initial Ortho_reg_weight", self.Ortho_reg_weight)
        self.L1_reg_weight = args.L1_weight_inital
        print("[trainer init] initial L1_reg_weight", self.L1_reg_weight)
        self.TV_weight_density, self.TV_weight_app = args.TV_weight_density, args.TV_weight_app
        print(f"[trainer init] initial TV_weight density: {self.TV_weight_density} appearance: {self.TV_weight_app}")

        if args.lr_decay_iters > 0:
            self.lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
        else:
            args.lr_decay_iters = args.n_iters
            self.lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)
        print("[trainer init] lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    def build_palette(self, filepath, is_sort_palette=False):
        # filepath = Path(filepath)
        # rgbs = self.train_dataset.all_rgbs
        # if self.train_dataset.white_bg:
        #     fg = torch.lt(rgbs, 1.).any(dim=-1)
        #     rgbs = rgbs[fg]
        # rgbs = rgbs.to(device='cpu', dtype=torch.double).numpy()
        # return sort_palette(rgbs, Hull_Simplification_determined_version(
        #     rgbs, filepath.stem + "-convexhull_vertices", error_thres=1. / 256.))
        palette = np.load(filepath)
        palette = torch.from_numpy(palette).float()
        if is_sort_palette:
            all_rgbs = self.train_dataset.all_rgbs
            return sort_palette(all_rgbs, palette)
        return palette

    def build_network(self):
        args = self.args

        ckpt_path = seek_checkpoint(args, self.ckpt_dir)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            kwargs = ckpt['kwargs']
            kwargs.update({'device': self.device})
            tensorf = MODEL_ZOO[args.model_name](**kwargs)
            tensorf.load(ckpt)
        else:
            n_lamb_sigma = args.n_lamb_sigma
            n_lamb_sh = args.n_lamb_sh
            near_far = self.train_dataset.near_far
            palette = self.palette

            tensorf = MODEL_ZOO[args.model_name](
                self.aabb, self.reso_cur, self.device,
                density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh,
                app_dim=args.data_dim_color, near_far=near_far,
                shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre,
                density_shift=args.density_shift, distance_scale=args.distance_scale,
                pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe,
                featureC=args.featureC, step_ratio=args.step_ratio,
                fea2denseAct=args.fea2denseAct, palette=palette)

        return tensorf

    # def plt_loss(self, plt_map, gt_train, weight=1.):
    #     pix, opq = plt_map[..., :3], plt_map[..., 3:]
    #     E_opaque = F.mse_loss(opq, self.ones.expand_as(opq), reduction='mean')
    #     loss = F.mse_loss(pix, gt_train, reduction='mean')
    #     return loss * weight, E_opaque * weight

    def train_one_batch(self, tensorf, iteration, rays_train, rgb_train):
        args = self.args
        white_bg = self.train_dataset.white_bg
        ndc_ray = args.ndc_ray
        
        loss_dict = {}

        rgb_map, _, _, weights, render_bufs = self.renderer(
            rays_train, tensorf, chunk=args.batch_size, N_samples=self.nSamples, white_bg=white_bg,
            ndc_ray=ndc_ray, device=self.device, is_train=True)

        # Loss
        img_loss = torch.mean((rgb_map[..., :3] - rgb_train) ** 2)
        loss_dict['img_loss'] = img_loss.clone().detach().item()

        total_loss = img_loss
        
        # Regularization
        if self.Ortho_reg_weight > 0:
            loss_reg_ortho = tensorf.vector_comp_diffs()
            total_loss += self.Ortho_reg_weight * loss_reg_ortho
            loss_dict['ortho_reg_loss'] = loss_reg_ortho.clone().detach().item()
        if self.L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += self.L1_reg_weight * loss_reg_L1
            loss_dict['L1_reg_loss'] = loss_reg_L1.clone().detach().item()

        if self.TV_weight_density > 0:
            self.TV_weight_density *= self.lr_factor
            loss_tv = tensorf.TV_loss_density(self.tvreg)
            total_loss = total_loss + loss_tv * self.TV_weight_density
            loss_dict['tv_loss_den'] = loss_tv.clone().detach().item()
        if self.TV_weight_app > 0:
            self.TV_weight_app *= self.lr_factor
            loss_tv = tensorf.TV_loss_app(self.tvreg)
            total_loss = total_loss + loss_tv * self.TV_weight_app
            loss_dict['tv_loss_app'] = loss_tv.clone().detach().item()

        loss_dict['total_loss'] = total_loss.detach().item()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # LR shrinkage
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * self.lr_factor

        return loss_dict

    def update_grid_resolution(self, tensorf, iteration):
        args = self.args
        # init resolution
        upsamp_list = args.upsamp_list
        update_AlphaMask_list = args.update_AlphaMask_list

        if iteration in update_AlphaMask_list:
            if self.reso_cur[0] * self.reso_cur[1] * self.reso_cur[2] < 256 ** 3:  # update volume resolution
                self.reso_mask = self.reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(self.reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                self.L1_reg_weight = args.L1_weight_rest
                print("[update_grid_resolution] set L1_reg_weight to", self.L1_reg_weight)

            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                self.trainingSampler.apply_filter(tensorf.filtering_rays)

        if iteration in upsamp_list:
            n_voxels = self.N_voxel_list.pop(0)
            self.reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            self.nSamples = min(args.nSamples, cal_n_samples(self.reso_cur, args.step_ratio))
            tensorf.upsample_volume_grid(self.reso_cur)

            if args.lr_upsample_reset:
                print("[update_grid_resolution] reset lr to initial")
                lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
            self.optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    def train(self):
        args = self.args
        white_bg = self.train_dataset.white_bg

        self.summary_writer = SummaryWriter(log_dir=self.tb_dir)

        tensorf = self.build_network()

        grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
        self.optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        torch.cuda.empty_cache()
        PSNRs, PSNRs_test = [], [0]

        self.trainingSampler = SimpleSampler(self.train_dataset, args.batch_size)
        if not args.ndc_ray:
            self.trainingSampler.apply_filter(tensorf.filtering_rays, bbox_only=True)

        print(f'=== training ======> {args.expname}')

        pbar = trange(args.n_iters, miniters=args.progress_refresh_every, file=sys.stdout)
        for iteration in pbar:
            ###### Core optimization ######
            batch_train = self.trainingSampler.getbatch(device=self.device)
            loss_dict = self.train_one_batch(tensorf, iteration, *batch_train)

            ###### Logging ######
            total_loss = loss_dict['total_loss']
            self.summary_writer.add_scalar('train/total_loss', total_loss, global_step=iteration)

            img_loss = loss_dict['img_loss']
            PSNRs.append(-10.0 * np.log(img_loss) / np.log(10.0))
            self.summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
            self.summary_writer.add_scalar('train/mse', img_loss, global_step=iteration)

            if 'ortho_reg_loss' in loss_dict:
                ortho_reg_loss = loss_dict['ortho_reg_loss']
                self.summary_writer.add_scalar('train/reg_ortho', ortho_reg_loss, global_step=iteration)
            if 'L1_reg_loss' in loss_dict:
                L1_reg_loss = loss_dict['L1_reg_loss']
                self.summary_writer.add_scalar('train/reg_L1', L1_reg_loss, global_step=iteration)

            if 'tv_loss_den' in loss_dict:
                tv_loss = loss_dict['tv_loss_den']
                self.summary_writer.add_scalar('train/reg_tv_density', tv_loss, global_step=iteration)
            if 'tv_loss_app' in loss_dict:
                tv_loss = loss_dict['tv_loss_app']
                self.summary_writer.add_scalar('train/reg_tv_app', tv_loss, global_step=iteration)

            # Print the current values of the losses.
            if iteration % args.progress_refresh_every == 0:
                pbar.set_description(
                    f'Iteration {iteration:05d}:'
                    + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                    + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                    + f' mse = {img_loss:.6f}'
                )
                PSNRs = []

            # Evaluation on testset
            if iteration % args.vis_every == args.vis_every - 1 and args.N_vis != 0:
                try:
                    print(f'== evaluation ======> {args.N_vis} views')
                    savePath = Path(self.run_dir, f'testset_vis_{iteration:06d}')
                    PSNRs_test = evaluation(self.test_dataset, tensorf, args, self.renderer, os.fspath(savePath), N_vis=args.N_vis,
                                            N_samples=self.nSamples, white_bg=white_bg, ndc_ray=args.ndc_ray, palette=self.palette,
                                            compute_extra_metrics=False, save_gt=True)
                    self.summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
                    print(f'=== continue training ======>')
                except Exception as e:
                    self.logger.warning(f'Evaluation failed: {e}')

            ###### Upsampling ######
            self.update_grid_resolution(tensorf, iteration)

        tensorf.save(f'{self.ckpt_dir}/{args.expname}_last.th')
        PSNRs_test = self.render_test(tensorf)
        self.summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=pbar.total)

    @torch.no_grad()
    def export_mesh(self):
        args = self.args
        ckpt = torch.load(args.ckpt, map_location=self.device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': self.device})
        tensorf = MODEL_ZOO[args.model_name](**kwargs)
        tensorf.load(ckpt)

        alpha, _ = tensorf.getDenseAlpha()
        convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply', bbox=tensorf.aabb.cpu(), level=0.005)

    @torch.no_grad()
    def render_test(self, tensorf):
        args = self.args
        white_bg = self.test_dataset.white_bg
        ndc_ray = args.ndc_ray

        logfolder = Path(self.run_dir)

        PSNRs_test = None
        if args.render_train:
            print(f'=== render train ======> {args.expname}')
            filePath = logfolder / 'render_train'
            PSNRs_test = evaluation(self.train_dataset, tensorf, args, self.renderer, os.fspath(filePath),  palette=self.palette,
                                    N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=self.device)
            print(f'mean psnr: {np.mean(PSNRs_test)}')

        if args.render_test:
            print(f'=== render test ======> {args.expname}')
            filePath = logfolder / 'render_test'
            PSNRs_test = evaluation(self.test_dataset, tensorf, args, self.renderer, os.fspath(filePath), palette=self.palette,
                                    N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=self.device)
            print(f'mean psnr: {np.mean(PSNRs_test)}')

        if args.render_path:
            filePath = logfolder / 'render_path'
            c2ws = self.test_dataset.render_path
            print('=== render path ======>', c2ws.shape)
            evaluation_path(self.test_dataset, tensorf, c2ws, self.renderer, os.fspath(filePath), N_samples=-1, palette=self.palette,
                            white_bg=white_bg, ndc_ray=ndc_ray, save_video=True, device=self.device)

        return PSNRs_test

import os
import os.path as osp
import math
import time
import glob
import torch

from data import create_dataloader
from models import define_model
from metrics import create_metric_calculator
from utils import dist_utils, base_utils, data_utils

from basicsr.data.data_util import read_img_seq
import datetime


def train(opt):
    # print configurations
    base_utils.log_info(f'{20 * "-"} Configurations {20 * "-"}')
    base_utils.print_options(opt)

    # create data loader
    train_loader = create_dataloader(opt, phase='train', idx='train')

    # build model
    model = define_model(opt)

    # set training params
    total_sample, iter_per_epoch = len(train_loader.dataset), len(train_loader)
    total_iter = opt['train']['total_iter']
    total_epoch = int(math.ceil(total_iter / iter_per_epoch))
    start_iter, iter = opt['train']['start_iter'], 0
    test_freq = opt['test']['test_freq']
    log_freq = opt['logger']['log_freq']
    ckpt_freq = opt['logger']['ckpt_freq']

    base_utils.log_info(f'Number of the training samples: {total_sample}')
    base_utils.log_info(f'{total_epoch} epochs needed for {total_iter} iterations')

    # train
    for epoch in range(total_epoch):
        if opt['dist']:
            train_loader.sampler.set_epoch(epoch)

        for data in train_loader:
            # update iter
            iter += 1
            curr_iter = start_iter + iter
            if iter > total_iter: break

            # prepare data
            model.prepare_training_data(data)

            # train a mini-batch
            model.train()

            # update running log
            model.update_running_log()

            # update learning rate
            model.update_learning_rate()

            # print messages
            if log_freq > 0 and curr_iter % log_freq == 0:
                msg = model.get_format_msg(epoch, curr_iter)
                base_utils.log_info(msg)

            # save model
            if ckpt_freq > 0 and curr_iter % ckpt_freq == 0:
                model.save(curr_iter)

            # evaluate model
            if test_freq > 0 and curr_iter % test_freq == 0:
                # set model index
                model_idx = f'G_iter{curr_iter}'

                # for each testset
                for dataset_idx in sorted(opt['dataset'].keys()):
                    # select test dataset
                    if 'test' not in dataset_idx: continue

                    ds_name = opt['dataset'][dataset_idx]['name']
                    base_utils.log_info(f'Testing on {ds_name} dataset')

                    # create data loader
                    test_loader = create_dataloader(
                        opt, phase='test', idx=dataset_idx)
                    test_dataset = test_loader.dataset
                    num_seq = len(test_dataset)

                    # create metric calculator
                    metric_calculator = create_metric_calculator(opt)

                    # infer a sequence
                    rank, world_size = dist_utils.get_dist_info()
                    for idx in range(rank, num_seq, world_size):
                        # fetch data
                        data = test_dataset[idx]

                        # prepare data
                        model.prepare_inference_data(data)

                        # infer
                        hr_seq = model.infer()

                        # save hr results
                        if opt['test']['save_res']:
                            res_dir = osp.join(
                                opt['test']['res_dir'], ds_name, model_idx)
                            res_seq_dir = osp.join(res_dir, data['seq_idx'])
                            data_utils.save_sequence(
                                res_seq_dir, hr_seq, data['frm_idx'], to_bgr=True)

                        # compute metrics for the current sequence
                        if metric_calculator is not None:
                            gt_seq = data['gt'].numpy()
                            metric_calculator.compute_sequence_metrics(
                                data['seq_idx'], gt_seq, hr_seq)

                    # save/print results
                    if metric_calculator is not None:
                        seq_idx_lst = [data['seq_idx'] for data in test_dataset]
                        metric_calculator.gather(seq_idx_lst)

                        if opt['test'].get('save_json'):
                            # write results to a json file
                            json_path = osp.join(
                                opt['test']['json_dir'], f'{ds_name}_avg.json')
                            metric_calculator.save(model_idx, json_path, override=True)
                        else:
                            # print directly
                            metric_calculator.display()


def edge_enhance_train(opt):
    base_utils.log_info(f'{20 * "-"} Configurations {20 * "-"}')
    base_utils.print_options(opt)

    # loading dataset
    lr_train_folder = opt['dataset']['train']['lr_seq_dir']
    gt_train_folder = opt['dataset']['train']['gt_seq_dir']

    subfolder_lr_train = sorted(glob.glob(osp.join(lr_train_folder, '*')))
    subfolder_gt_train = sorted(glob.glob(osp.join(gt_train_folder, '*')))

    total_iter = opt['train']['total_iter']
    start_iter = 0
    test_freq = opt['test']['test_freq']
    log_freq = opt['logger']['log_freq']
    ckpt_freq = opt['logger']['ckpt_freq']
    ckpt_freq = opt['logger']['ckpt_freq']

    # build model
    model = define_model(opt)

    # batch size
    bz = 1
    t_each_side = 2

    for epoch in range(100):
        for (subfolder, subfolder_gt) in zip(subfolder_lr_train, subfolder_gt_train):
            imgs_lq = read_img_seq(subfolder, return_imgname=False)
            imgs_gt = read_img_seq(subfolder_gt, return_imgname=False)

            for i in range(t_each_side, imgs_lq.shape[0], 1):
                # update iter
                curr_iter = start_iter + 1

                input = imgs_lq[i-t_each_side:i+t_each_side, :, :, :].unsqueeze(0).permute(0,1,3,4,2).cuda()
                target = imgs_gt[i-t_each_side:i+t_each_side, :, :, :].unsqueeze(0).permute(0,1,3,4,2).cuda()

                # prepare data
                model.prepare_training_data({"gt": target.cuda(), "lr": input.cuda()})

                # train a mini-batch
                model.train()

                # update running log
                model.update_running_log()

                # update learning rate
                model.update_learning_rate()

                # print messages
                if log_freq > 0 and curr_iter % log_freq == 0:
                    msg = model.get_format_msg(epoch, curr_iter)
                    base_utils.log_info(msg)

                # save model
                if ckpt_freq > 0 and curr_iter % ckpt_freq == 0:
                    model.save(curr_iter)


def test(opt):
    # logging
    base_utils.print_options(opt)

    # infer and evaluate performance for each model
    for load_path in opt['model']['generator']['load_path_lst']:
        # set model index
        model_idx = osp.splitext(osp.split(load_path)[-1])[0]

        # log
        base_utils.log_info(f'{"=" * 40}')
        base_utils.log_info(f'Testing model: {model_idx}')
        base_utils.log_info(f'{"=" * 40}')

        # create model
        opt['model']['generator']['load_path'] = load_path
        model = define_model(opt)

        # for each test dataset
        for dataset_idx in sorted(opt['dataset'].keys()):
            # select testing dataset
            if 'test' not in dataset_idx:
                continue

            ds_name = opt['dataset'][dataset_idx]['name']
            base_utils.log_info(f'Testing on {ds_name} dataset')

            # create data loader
            test_loader = create_dataloader(opt, phase='test', idx=dataset_idx)
            test_dataset = test_loader.dataset
            num_seq = len(test_dataset)

            # create metric calculator
            metric_calculator = create_metric_calculator(opt)

            # infer a sequence
            rank, world_size = dist_utils.get_dist_info()
            for idx in range(rank, num_seq, world_size):
                # fetch data
                data = test_dataset[idx]

                # prepare data
                model.prepare_inference_data(data)

                # infer
                hr_seq = model.infer()

                # save hr results
                if opt['test']['save_res']:
                    res_dir = osp.join(
                        opt['test']['res_dir'], ds_name, model_idx)
                    res_seq_dir = osp.join(res_dir, data['seq_idx'])
                    data_utils.save_sequence(
                        res_seq_dir, hr_seq, data['frm_idx'], to_bgr=True)

                # compute metrics for the current sequence
                if metric_calculator is not None:
                    gt_seq = data['gt'].numpy()
                    metric_calculator.compute_sequence_metrics(
                        data['seq_idx'], gt_seq, hr_seq)

            # save/print results
            if metric_calculator is not None:
                seq_idx_lst = [data['seq_idx'] for data in test_dataset]
                metric_calculator.gather(seq_idx_lst)

                if opt['test'].get('save_json'):
                    # write results to a json file
                    json_path = osp.join(
                        opt['test']['json_dir'], f'{ds_name}_avg.json')
                    metric_calculator.save(model_idx, json_path, override=True)
                else:
                    # print directly
                    metric_calculator.display()

            base_utils.log_info('-' * 40)


def profile(opt, lr_size, test_speed=False):
    # basic configs
    scale = opt['scale']
    device = torch.device(opt['device'])
    msg = '\n'

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False

    # logging
    base_utils.print_options(opt['model']['generator'])

    lr_size_lst = tuple(map(int, lr_size.split('x')))
    hr_size = f'{lr_size_lst[0]}x{lr_size_lst[1] * scale}x{lr_size_lst[2] * scale}'
    msg += f'{"*" * 40}\nResolution: {lr_size} -> {hr_size} ({scale}x SR)'

    # create model
    from models.networks import define_generator
    net_G = define_generator(opt).to(device)
    # base_utils.log_info(f'\n{net_G.__str__()}')

    # profile
    lr_size = tuple(map(int, lr_size.split('x')))
    gflops_dict, params_dict = net_G.profile(lr_size, device)

    gflops_all, params_all = 0, 0
    for module_name in gflops_dict.keys():
        gflops, params = gflops_dict[module_name], params_dict[module_name]
        msg += f'\n{"-" * 40}\nModule: [{module_name}]'
        msg += f'\n    FLOPs (10^9): {gflops:.3f}'
        msg += f'\n    Parameters (10^6): {params / 1e6:.3f}'
        gflops_all += gflops
        params_all += params
    msg += f'\n{"-" * 40}\nOverall'
    msg += f'\n    FLOPs (10^9): {gflops_all:.3f}'
    msg += f'\n    Parameters (10^6): {params_all / 1e6:.3f}\n{"*" * 40}'

    # test running speed
    if test_speed:
        n_test, tot_time = 30, 0
        for i in range(n_test):
            dummy_input_list = net_G.generate_dummy_data(lr_size, device)

            start_time = time.time()
            # ---
            net_G.eval()
            with torch.no_grad():
                _ = net_G.step(*dummy_input_list)
            torch.cuda.synchronize()
            # ---
            end_time = time.time()
            tot_time += end_time - start_time
        msg += f'\nSpeed: {n_test / tot_time:.3f} FPS (averaged over {n_test} runs)\n{"*" * 40}'

    base_utils.log_info(msg)


if __name__ == '__main__':
    # === parse arguments === #
    args = base_utils.parse_agrs()

    # === generic settings === #
    # parse configs, set device, set ramdom seed
    opt = base_utils.parse_configs(args)
    # set logger
    base_utils.setup_logger('base')
    # set paths
    base_utils.setup_paths(opt, args.mode)

    # === train === #
    if args.mode == 'train':
        train(opt)
    # === edge enhance train  === #
    elif args.mode == 'edge_enhance_train':
        edge_enhance_train(opt)

    # # === edge enhance train  === #
    # elif args.mode == 'edge_enhance_test':
    #     my_train(opt)

    # === test === #
    elif args.mode == 'test':
        test(opt)

    # === profile === #
    elif args.mode == 'profile':
        profile(opt, args.lr_size, args.test_speed)

    else:
        raise ValueError(f'Unrecognized mode: {args.mode} (train|test|profile)')

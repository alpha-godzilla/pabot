"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import numpy as np
import torch
from torchvision.utils import save_image
from models.utils import save_high_quality_tensor_image
from skimage.metrics import structural_similarity as ssim_fn
import os
import time

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    if opt.direction == 'AtoB':
        max_num_test = len(os.listdir(opt.dataroot+'/testA'))
    else:
        max_num_test = len(os.listdir(opt.dataroot+'/testB'))
    loss_pathes = []
    psnr_list, ssim_list, lpips_list = [], [], []
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').cuda()
    except ImportError:
        print('Warning: lpips not installed, LPIPS will be skipped.')
        lpips_fn = None
    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        st = time.time()
        model.forward()           # run inference
        et = time.time()
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, width=opt.display_winsize)

        # Compute PSNR / SSIM / LPIPS (fake_B vs real_B)
        fake = visuals['fake_B']
        real = visuals['real_B']
        # LPIPS on [-1,1] tensors
        if lpips_fn is not None:
            lpips_val = lpips_fn(fake.cuda(), real.cuda()).item()
            lpips_list.append(lpips_val)
        # Convert to [0,255] uint8 numpy
        fake_np = ((fake[0].detach().cpu().clamp(-1, 1).float() + 1) / 2 * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
        real_np = ((real[0].detach().cpu().clamp(-1, 1).float() + 1) / 2 * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
        # PSNR
        mse = np.mean((fake_np.astype(np.float64) - real_np.astype(np.float64)) ** 2)
        psnr_list.append(10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf'))
        # SSIM
        ssim_list.append(ssim_fn(fake_np, real_np, channel_axis=2, data_range=255))

    webpage.save()  # save the HTML

    # Print and save test metrics
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    msg = '\n========== Test Results (epoch %s) ==========\n' % opt.epoch
    msg += '  PSNR:  %.3f\n' % avg_psnr
    msg += '  SSIM:  %.4f\n' % avg_ssim
    if lpips_list:
        avg_lpips = np.mean(lpips_list)
        msg += '  LPIPS: %.4f\n' % avg_lpips
    msg += '  Num images: %d\n' % len(psnr_list)
    msg += '================================================\n'
    print(msg)
    metrics_path = os.path.join(web_dir, 'test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(msg)
    print('Metrics saved to', metrics_path)
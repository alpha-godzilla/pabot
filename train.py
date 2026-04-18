import time
import torch
import os
import re
from options.train_options import TrainOptions
from data import create_dataset, get_test_loaders, get_val_loaders
from models import create_model
from util.visualizer import Visualizer
from util.util import write_images
from models.utils import eval_loader, eval_val_metrics, SimpleLogger


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    wandb_run = None
    if getattr(opt, "use_wandb", False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=opt.wandb_project,
                entity=opt.wandb_entity,
                name=opt.wandb_run_name or opt.name,
                mode=opt.wandb_mode,
                config=vars(opt),
            )
            print(f'W&B enabled: run={wandb_run.name}')
        except ImportError:
            print('Warning: wandb is not installed. Install with: pip install wandb')
            wandb_run = None

    num_gpus = len(opt.gpu_ids)
    if num_gpus > 1 and opt.batch_size < num_gpus:
        raise ValueError(
            'For multi-GPU DataParallel, --batch_size must be >= number of GPUs. '
            f'Got batch_size={opt.batch_size}, num_gpus={num_gpus}.'
        )
    if num_gpus > 1 and opt.batch_size % num_gpus != 0:
        print(
            f'Warning: batch_size ({opt.batch_size}) is not divisible by num_gpus ({num_gpus}). '
            'Per-GPU load may be imbalanced.'
        )
    device = torch.device(f'cuda:{opt.gpu_ids[0]}') if num_gpus > 0 else torch.device('cpu')

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    test_loader_a, test_loader_b = get_test_loaders(opt)
    val_loader_a, val_loader_b = get_val_loaders(opt)
    # 修改选取逻辑
    chosen_ids = [50, 100, 150, 200] 
    fix_a = torch.stack([test_loader_a.dataset[i]['A'] for i in chosen_ids[:opt.display_size]]).to(device)
    fix_b = torch.stack([test_loader_b.dataset[i]['A'] for i in chosen_ids[:opt.display_size]]).to(device)
    # fix_a = torch.stack([test_loader_a.dataset[i]['A'] for i in range(opt.display_size)]).cuda()  # fixed test data
    # fix_b = torch.stack([test_loader_b.dataset[i]['A'] for i in range(opt.display_size)]).cuda()
    #print(fix_a.shape,fix_a.size(0)) torch.Size([16, 3, 256, 256]) 16
    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    test_logger = SimpleLogger(os.path.join(opt.run_dir, 'test.txt'))
    total_iters = 0                # the total number of training iterations

    # LPIPS model (initialize once)
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device)
    except ImportError:
        print('Warning: lpips not installed, LPIPS will be skipped. Install with: pip install lpips')
        lpips_fn = None
    best_psnr = 0.0
    best_phi_mse = None
    best_phi_epoch = None
    best_phi_applied = False

    def _scan_best_phi_mse_from_phase_states(run_dir):
        phase_pat = re.compile(r"^(\d+)_phase_state\.pth$")
        best_epoch_local = None
        best_mse_local = None
        if not os.path.isdir(run_dir):
            return best_epoch_local, best_mse_local
        for fname in os.listdir(run_dir):
            m = phase_pat.match(fname)
            if m is None:
                continue
            path = os.path.join(run_dir, fname)
            try:
                state = torch.load(path, map_location="cpu")
            except Exception:
                continue
            mse = state.get("phi_epoch_mse_loss", None)
            if mse is None:
                continue
            try:
                mse = float(mse)
            except (TypeError, ValueError):
                continue
            ep = int(m.group(1))
            if best_mse_local is None or mse < best_mse_local:
                best_mse_local = mse
                best_epoch_local = ep
        return best_epoch_local, best_mse_local

    if bool(getattr(opt, "continue_train", False)):
        best_phi_epoch, best_phi_mse = _scan_best_phi_mse_from_phase_states(opt.run_dir)
        if best_phi_mse is not None:
            print(f"[resume] current best phi-epoch-mse from history: epoch={best_phi_epoch}, mse={best_phi_mse:.6f}")

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        phi_step_count = 0
        phi_sample_count = 0
        phi_opt_time_sum = 0.0

        dataset.set_epoch(epoch)
        model.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                if num_gpus > 1:
                    model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            step_opt_time = time.time() - optimize_start_time
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if bool(getattr(model, "is_phi_pretrain_stage", False)):
                phi_step_count += 1
                phi_sample_count += int(batch_size)
                phi_opt_time_sum += float(step_opt_time)


            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)


            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                if wandb_run is not None:
                    wandb_run.log({
                        **{f"train/{k}": float(v) for k, v in losses.items()},
                        "train/epoch": int(epoch),
                        "train/epoch_iter": int(epoch_iter),
                        "train/total_iters": int(total_iters),
                        "train/optimize_time": float(optimize_time),
                        "train/data_time": float(t_data),
                    }, step=int(total_iters))

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % (opt.eval_epoch_freq) == 0 or epoch == 1:  # display images on visdom and save images to a HTML file
            images = model.interpolation(fix_a, fix_b)
            write_images(images, 12, opt.img_dir, postfix='%03d_interp' % epoch)
            images = model.sample(fix_a, fix_b)
            write_images(images, opt.display_size, opt.img_dir, postfix='%03d_sample' % epoch)

        # Save latest at every epoch end for reliable resume, and keep numbered
        # snapshots only every <save_epoch_freq> epochs to control disk usage.
        print('saving latest model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        if epoch % max(1, int(opt.save_epoch_freq)) == 0:
            print('saving epoch checkpoint at epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)

        # Track best phi by minimum epoch-MSE of phi loss during phi pretraining stage.
        stage_epoch = max(0, int(epoch) - int(getattr(opt, "epoch_count", 1)))
        phi_pretrain_epochs = int(getattr(opt, "phi_pretrain_epochs", 0))
        phi_pretrain_max_epochs = getattr(opt, "phi_pretrain_max_epochs", None)
        if phi_pretrain_max_epochs is None:
            max_phi_epochs = phi_pretrain_epochs
        else:
            max_phi_epochs = int(phi_pretrain_max_epochs)
        max_phi_epochs = max(0, max_phi_epochs)
        in_phi_stage = stage_epoch < max_phi_epochs

        phi_epoch_mse = model.get_phi_epoch_mse_loss() if hasattr(model, "get_phi_epoch_mse_loss") else None
        if in_phi_stage and phi_epoch_mse is not None:
            if best_phi_mse is None or float(phi_epoch_mse) < float(best_phi_mse):
                best_phi_mse = float(phi_epoch_mse)
                best_phi_epoch = int(epoch)
                model.save_networks('best_phi')
                print('==> Best Phi MSE: %.6f at epoch %d, saving best_phi model' % (best_phi_mse, best_phi_epoch))

        # Right after finishing phi-pretrain (e.g., phi50), swap to best_phi for warmup/normal stages.
        if (not in_phi_stage) and (not best_phi_applied) and best_phi_epoch is not None and max_phi_epochs > 0:
            print('loading best_phi checkpoint (epoch %d) for post-phi stages' % best_phi_epoch)
            model.load_networks('best_phi')
            best_phi_applied = True

        # Evaluate PSNR / SSIM / LPIPS on the validation set
        if epoch % opt.eval_epoch_freq == 0:
            if opt.direction == 'BtoA':
                val_src, val_tgt = val_loader_b, val_loader_a
            else:
                val_src, val_tgt = val_loader_a, val_loader_b
            results = eval_val_metrics(model, val_src, val_tgt, lpips_fn)
            test_logger.log(epoch, opt.n_epochs + opt.n_epochs_decay, results, verbose=True)
            if wandb_run is not None:
                wandb_run.log({
                    **{f"val/{k}": float(v) for k, v in results.items()},
                    "val/epoch": int(epoch),
                }, step=int(total_iters))
            if results['PSNR'] > best_psnr:
                best_psnr = results['PSNR']
                model.save_networks('best')
                test_logger.log_message('==> Best PSNR: %.3f at epoch %d, saving best model' % (best_psnr, epoch))
            if wandb_run is not None:
                wandb_run.log({"val/best_psnr": float(best_psnr)}, step=int(total_iters))

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        if phi_step_count > 0:
            phi_avg_step = phi_opt_time_sum / max(1, phi_step_count)
            phi_sps = phi_sample_count / max(1e-8, phi_opt_time_sum)
            print(
                '[phi-speed] epoch %d: steps=%d, samples=%d, avg_step=%.4fs, throughput=%.2f samples/s'
                % (epoch, phi_step_count, phi_sample_count, phi_avg_step, phi_sps)
            )
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
        if wandb_run is not None and len(model.optimizers) > 0:
            wandb_run.log({
                "train/lr": float(model.optimizers[0].param_groups[0]["lr"]),
                "train/epoch_end": int(epoch),
            }, step=int(total_iters))

    if wandb_run is not None:
        wandb_run.finish()

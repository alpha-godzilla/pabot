import torch
import torch.nn.functional as F
import os
import torch_fidelity
import glob
import cv2
from PIL import Image
import numpy as np
#----------------------------------------------------#
# evaluation modules
#----------------------------------------------------#
class SimpleLogger:
    def __init__(self, path):
        self.path = path

    def log(self, iteration, max_iteration, metric_dict, verbose=False):
        message = '[%03d/%03d] ' % (iteration, max_iteration)
        for key in metric_dict:
            message += '\t %s:%.3f \t' % (key, metric_dict[key])
        if verbose:
            print(message)
        record = open(self.path, 'a')
        record.write('\n' + message + '\n')
        record.close()

    def log_message(self, message, verbose=True):
        record = open(self.path, 'a')
        record.write('\n' + message + '\n')
        if verbose:
            print(message)
        record.close()
        
def rgb_to_grayscale(tensor):
    weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=tensor.dtype, device=tensor.device)
    weights = weights.view(1, 3, 1, 1)
    gray_tensor = (tensor * weights).sum(dim=1, keepdim=True)

    return gray_tensor

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    image_numpy = image_numpy.clip(0, 255)
    return image_numpy.astype(imtype)


def save_image_numpy(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

@torch.no_grad()
def eval_loader(model, test_loader_a, test_loader_b, output_directory, opt):
    fake_dir = os.path.join(output_directory, 'fake')
    if not os.path.exists(fake_dir):
        os.mkdir(fake_dir)
    if opt.direction == 'AtoB':
        test_loader = test_loader_a
        real_dir = os.path.join(opt.dataroot, 'testB')
        acc_loader = test_loader_b
    else:
        test_loader = test_loader_b
        real_dir = os.path.join(opt.dataroot, 'testA')
        acc_loader = test_loader_a

    for it, (data, acc_data) in enumerate(zip(test_loader,acc_loader)):
        fake = model.translate(data['A'].cuda(), acc_data['A'].cuda())
        path_fake = os.path.join(fake_dir, os.path.basename(data['A_paths'][0]).replace('jpg', 'png'))
        im = tensor2im(fake)
        save_image_numpy(im, path_fake)
    eval_dict = eval_method(real_dir, fake_dir)
    return eval_dict

def save_high_quality_tensor_image(tensor, path):
    im = tensor2im(tensor)
    save_image_numpy(im, path)


@torch.no_grad()
def eval_val_metrics(model, src_loader, tgt_loader, lpips_fn=None):
    """Compute PSNR, SSIM, LPIPS on paired validation set.

    Args:
        model: model with translate(x) method; if it also exposes extract_attention(x),
               metrics are computed on paired attention maps rather than raw images.
        src_loader: source domain DataLoader (input to translate)
        tgt_loader: target domain DataLoader (ground truth)
        lpips_fn: pre-initialized lpips.LPIPS model (optional)
    """
    from skimage.metrics import structural_similarity as ssim_fn

    psnr_list, ssim_list, lpips_list = [], [], []

    is_phi_eval = getattr(model, "is_phi_pretrain_stage", False)

    for src_data, tgt_data in zip(src_loader, tgt_loader):
        src_img = src_data['A'].cuda()
        tgt_img = tgt_data['A'].cuda()

        if is_phi_eval:
            # --- Phi-only Evaluation Logic (Attention Map PSNR) ---
            # Extract source features and predict with Transformer
            phi_in_domain = str(getattr(model.opt, "phi_input_domain", "attention")).strip().lower()
            out_src = model._extract_attention_features(src_img)
            # _extract_attention_features returns (attn, cls, feat) or (attn, cls) or (attn, feat) depending on flags
            # In DualVelocityStructModel, it returns (attn, cls, feat) if phi_input_domain="feature"
            # otherwise (attn, cls, None).
            attn_src = out_src[0]
            cls_src = out_src[1]
            feat_src = out_src[2] if len(out_src) > 2 else None

            # Input for Phi branch
            if phi_in_domain == "feature":
                phi_input = model._phi_build_input(attn_src, cond_feature_map=feat_src)
            else:
                phi_input = attn_src

            # Output is [B, 1, 28, 28] in [0, 1] range (sigmoid)
            fake_img = model.netPhi(phi_input)
            
            # Get Ground Truth CT Attention
            out_tgt = model.dino_extractor(tgt_img, return_cls_attn=True)
            tgt_img_map = out_tgt[0]
            # Normalize GT to [0, 1] for fair PSNR
            tgt_img_norm = model._normalize_struct_features(tgt_img_map)
            
            # Use native 28x28 resolution for evaluation (matching loss)
            fake_eval = fake_img * 2.0 - 1.0
            tgt_eval = tgt_img_norm * 2.0 - 1.0
        else:
            # --- Standard Image Translation Evaluation ---
            fake_eval = model.translate(src_img)  # [-1, 1]
            if hasattr(model, "extract_attention"):
                tgt_eval = model.extract_attention(tgt_img)  # [-1, 1]
            else:
                tgt_eval = tgt_img

        # LPIPS (expects [-1, 1])
        if lpips_fn is not None:
            if fake_eval.shape[-2:] != (224, 224):
                lpips_input_fake = F.interpolate(fake_eval, size=(224, 224), mode='bilinear', align_corners=False)
                lpips_input_tgt = F.interpolate(tgt_eval, size=(224, 224), mode='bilinear', align_corners=False)
            else:
                lpips_input_fake, lpips_input_tgt = fake_eval, tgt_eval
            lpips_val = lpips_fn(lpips_input_fake, lpips_input_tgt).item()
            lpips_list.append(lpips_val)

        # Convert to [0, 255] uint8 numpy for PSNR / SSIM
        fake_np = ((fake_eval[0].cpu().clamp(-1, 1).float() + 1) / 2 * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
        tgt_np  = ((tgt_eval[0].cpu().clamp(-1, 1).float() + 1) / 2 * 255).numpy().transpose(1, 2, 0).astype(np.uint8)

        # PSNR
        mse = np.mean((fake_np.astype(np.float64) - tgt_np.astype(np.float64)) ** 2)
        if mse == 0:
            psnr_list.append(float('inf'))
        else:
            psnr_list.append(10 * np.log10(255.0 ** 2 / mse))

        # SSIM
        ssim_val = ssim_fn(fake_np, tgt_np, channel_axis=2, data_range=255)
        ssim_list.append(ssim_val)

    results = {
        'PSNR': np.mean(psnr_list),
        'SSIM': np.mean(ssim_list),
    }
    if lpips_list:
        results['LPIPS'] = np.mean(lpips_list)
    return results


@torch.no_grad()
def eval_method(real_path, fake_path):
    print(real_path)
    print(fake_path)
    eval_dict = {}
    eval_args = {'fid': True, 'kid': True, 'kid_subset_size': 50, 'kid_subsets': 10, 'verbose': False, 'cuda': True}
    metric_dict_AB = torch_fidelity.calculate_metrics(input1=real_path, input2=fake_path, **eval_args)
    eval_dict['FID'] = metric_dict_AB['frechet_inception_distance']
    eval_dict['KID'] = metric_dict_AB['kernel_inception_distance_mean']*100.
    return eval_dict

def eval_maps(real_path, fake_path, thr1=5, thr2=10, name=''):
    reals = glob.glob(real_path + '/*')
    fakes = glob.glob(fake_path + '/*')

    reals = sorted(reals)
    fakes = sorted(fakes)
    print(real_path, fake_path)

    num_imgs = len(reals)
    corr5_count = 0.0
    corr10_count = 0.0
    pix_count = 0.0
    RMSE = 0.0
    for i in range(num_imgs):

        real = cv2.imread(reals[i])
        fake = cv2.imread(fakes[i])

        real = cv2.resize(real, (256, 256), interpolation=cv2.INTER_LINEAR)
        fake = cv2.resize(fake, (256, 256), interpolation=cv2.INTER_LINEAR)

        real = real.astype(np.float32)
        fake = fake.astype(np.float32)
        diff = np.abs(real - fake)

        max_diff = np.max(diff, axis=2)

        corr5_count = corr5_count + np.sum(max_diff < thr1)
        corr10_count = corr10_count + np.sum(max_diff < thr2)
        pix_count = pix_count + 256**2

        diff = (diff**2)/(256**2)
        diff = np.sum(diff)
        rmse = np.sqrt(diff)
        RMSE = RMSE + rmse

    RMSE = RMSE/num_imgs
    acc5 = corr5_count/pix_count*100.
    acc10 = corr10_count/pix_count*100.
    eval_dict = {'%s/rmse' % (name):RMSE,'%s/acc@%d'%(name, thr1):acc5, '%s/acc@%d'%(name, thr2):acc10}
    return eval_dict



def eval_city2parsing(real_path, fake_path):
    labels = [{'name':'road', 'catId':0, 'color': (128, 64, 128)},
              {'name':'sidewalk', 'catId':1, 'color': (244, 35, 232)},
              {'name':'building', 'catId':2, 'color': (70, 70, 70)},
              {'name':'wall', 'catId':3, 'color': (102, 102, 156)},
              {'name':'fence', 'catId':4, 'color': (190, 153, 153)},
              {'name':'pole', 'catId':5, 'color': (153, 153, 153)},
              {'name':'traffic_light', 'catId':6, 'color': (250, 170, 30)},
              {'name':'traffic_sign', 'catId':7, 'color': (220, 220, 0)},
              {'name':'vegetation', 'catId':8, 'color': (107, 142, 35)},
              {'name':'terrain', 'catId':9, 'color': (152, 251, 152)},
              {'name':'sky', 'catId':10, 'color': (70, 130, 180)},
              {'name':'person', 'catId':11, 'color': (220, 20, 60)},
              {'name':'rider', 'catId':12, 'color': (255, 0, 0)},
              {'name':'car', 'catId':13, 'color': (0, 0, 142)},
              {'name':'truck', 'catId':14, 'color': (0, 0, 70)},
              {'name':'bus', 'catId':15, 'color': (0, 60, 100)},
              {'name':'train', 'catId':16, 'color': (0, 80, 100)},
              {'name':'motorcycle', 'catId':17, 'color': (0, 0, 230)},
              {'name':'bicycle', 'catId':18, 'color': (119, 11, 32)},
              {'name':'ignore', 'catId':19, 'color': (0, 0, 0)}]

    reals = glob.glob(real_path+'/*jpg')
    fakes = glob.glob(fake_path+'/*png')
    reals = sorted(reals)
    fakes = sorted(fakes)
    num_imgs = len(reals)

    CM = np.zeros((19,19), dtype=np.float32)
    # test
    for i in range(num_imgs):
        real = cv2.imread(reals[i])
        fake = cv2.imread(fakes[i])

        real = cv2.resize(real, (128, 128), interpolation=cv2.INTER_NEAREST)
        fake = cv2.resize(fake, (128, 128), interpolation=cv2.INTER_NEAREST)

        pred = fake
        label = real


        label_dis = np.zeros((20, 128, 128), dtype=np.float32)
        pred_dis = np.zeros((20, 128, 128), dtype=np.float32)

        for j in range(20):
            color = labels[j]['color']
            label_diff = np.abs(label - color)
            pred_diff = np.abs(pred - color)

            label_diff = np.sum(label_diff, axis=2)
            pred_diff = np.sum(pred_diff, axis=2)

            label_dis[j,:,:] = label_diff
            pred_dis[j,:,:] = pred_diff

        label_id = np.argmin(label_dis, axis=0)
        pred_id = np.argmin(pred_dis, axis=0)

        for j in range(19):
            coord = np.where(label_id == j)
            pred_j = pred_id[coord]
            for k in range(19):
                CM[j,k] = CM[j, k] + np.sum(pred_j == k)


    pix_acc = 0
    mean_acc = 0
    mean_IoU = 0

    count = 0
    for i in range(19):
        count = count + CM[i, i]
    pix_acc = count / np.sum(CM)

    count = 0
    for i in range(19):
        temp = CM[i, :]
        count = count + CM[i,i]/(np.sum(temp) + 1e-6)
    mean_acc = count/19

    count = 0
    for i in range(19):
        temp_0 = CM[i, :]
        temp_1 = CM[:, i]
        count = count + CM[i, i]/(np.sum(temp_0) + np.sum(temp_1) - CM[i, i] + 1e-6)

    mean_IoU = count/19

    eval_dict = {'pix_acc':pix_acc, 'mean_acc':mean_acc, 'mean_IoU':mean_IoU}
    return eval_dict



if __name__ == '__main__':
    import sys
    out = eval_maps(sys.argv[1], sys.argv[2])
    print(out)

from torch import optim
from adabelief_pytorch import AdaBelief
from typing import Tuple
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import torch
import os
import matplotlib.pyplot as plt
import torchvision
import yaml

def optimizer_config(net, tools):
    opt_name = tools['name']
    learning_rate = float(tools['lr'])
    momentum = float(tools['momentum'])
    weight_decay = float(tools['weight_decay'])
    eps = float(tools['eps'])
    weight_decouple = tools['weight_decouple']
    sam_option = tools['sam']
    accumulate_iter = tools['accumulate_iter']

    t0 = int(tools['scheduler']['T_0'])
    t_mult = float(tools['scheduler']['T_mult'])
    eta_max =  float(tools['scheduler']['eta_max'])
    min_lr =  float(tools['scheduler']['min_lr'])
    t_up = int(tools['scheduler']['T_up'])
    gamma = float(tools['scheduler']['gamma'])

    if sam_option:
        if opt_name.lower() == 'sgd':
            base_opt = optim.SGD
            optimizer = SAM(net.parameters(), base_opt ,lr=learning_rate, momentum=momentum, weight_decay=weight_decay,
                            nesterov=True, accumulate_grad_batch=accumulate_iter)

        elif opt_name.lower() == 'adabelief':
            base_opt = AdaBelief
            optimizer = SAM(net.parameters(), base_opt ,lr=learning_rate, eps=eps, weight_decay=weight_decay,
                            weight_decouple=weight_decouple, rectify = True, print_change_log = False,
                            accumulate_grad_batch=accumulate_iter)

        elif opt_name.lower() == 'adam':
            base_opt = optim.Adam
            optimizer = SAM(net.parameters(), base_opt ,lr=learning_rate, eps=eps, weight_decay=weight_decay,
                            accumulate_grad_batch=accumulate_iter)

        else:
            base_opt = optim.Adam
            optimizer = SAM(net.parameters(), base_opt, lr=learning_rate, eps=eps, weight_decay=weight_decay,
                            accumulate_grad_batch=accumulate_iter)

    else:
        if opt_name.lower() == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=learning_rate,momentum=momentum, weight_decay=weight_decay,
                            nesterov=True)

        elif opt_name.lower() == 'adabelief':
            optimizer = AdaBelief(net.parameters() ,lr=learning_rate, eps=eps, weight_decay=weight_decay,
                            weight_decouple=weight_decouple, rectify=True, print_change_log = False)

        elif opt_name.lower() == 'adam':
            optimizer = optim.Adam(net.parameters(), lr=learning_rate, eps=eps, weight_decay=weight_decay)

        else:
            optimizer = optim.Adam(net.parameters(), lr=learning_rate, eps=eps, weight_decay=weight_decay)

    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=t0, cycle_mult=t_mult, max_lr=eta_max,
                                              min_lr=min_lr, warmup_steps=t_up, gamma=gamma)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.0001, last_epoch=100, )
    return optimizer, scheduler


def add_image_to_tensorboard(images, writer, description='images'):
    images = UnNormalize(images)
    img_grid = torchvision.utils.make_grid(images)
    # matplotlib_imshow(img_grid)
    writer.add_image(description, img_grid)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def plot_dataloader(batch, dataloader, config, classes):

    mean = config['mean']
    std = config['std']
    for i in range(batch):
        img, label = next(iter(dataloader))
        image = img[i, :, :, :].squeeze()
        un_normalize_img = UnNormalize(mean=mean, std=std)
        image = un_normalize_img(image)
        image = image.permute(1, 2, 0)
        plt.imshow(image)
        plt.title(classes[str(int(label[i]))])
        plt.show()


def partial_mixup(input_tensors: torch.Tensor,
                  gamma: float,
                  indices: torch.Tensor
                  ) -> torch.Tensor:
    if input_tensors.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input_tensors[indices]
    return input_tensors.mul(gamma).add(perm_input, alpha=1 - gamma)


# adopted from https://github.com/moskomule/mixup.pytorch
def mixup(input_tensors: torch.Tensor,
          target: torch.Tensor,
          gamma: float,
          ) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randperm(input_tensors.size(0), device=input_tensors.device, dtype=torch.long)
    return partial_mixup(input_tensors, gamma, indices), partial_mixup(target, gamma, indices)


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def save_best_model(accuracy, best_accuracy, epoch, first_save, net, optimizer, avg_loss, path, logger, timestamp):
    if accuracy > best_accuracy and epoch > first_save:
        best_accuracy = accuracy
        model_name = f'model_{timestamp}_{epoch + 1}_{best_accuracy}.pt'
        logger.info(model_name + '  saved\n')
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, os.path.join(path,'best_ckpt.pt'))

    return best_accuracy

def get_config(config):

    classes = config['classes']

    epochs = config['train']['epochs']
    do_every_train = config['train']['do_every']
    train_batch = config['train']['batch_size']
    mixed_precision = config['train']['mixed_precision']
    model_name = config['train']['model']['name']
    fine_tuned_flag = config['train']['model']['fine_tuned']

    device = config['device']

    sam_option = config['optimizer']['sam']

    train_data_path = config['data']['train']['path']
    csv_train_filename = config['data']['train']['csv_filename']

    valid_data_path = config['data']['validation']['path']
    csv_valid_filename = config['data']['validation']['csv_filename']
    valid_batch = config['data']['validation']['batch_size']
    do_every_valid = config['data']['validation']['do_every']
    valid_dataloader_flag = config['data']['validation']['dataloader_plot']
    first_save = config['data']['validation']['first_epoch_to_save']
    mix_up_tools = config['data']['train']['augmentation']['mix_up']


    return classes, epochs, do_every_train, train_batch, train_data_path, valid_data_path, csv_train_filename, \
        csv_valid_filename, model_name, fine_tuned_flag, mixed_precision,  device, sam_option, valid_batch,\
        do_every_valid, first_save, valid_dataloader_flag, mix_up_tools


def get_logger(logging, path):
    logger = logging.getLogger()
    logging.basicConfig(filename=path + '/logs.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    logger.addHandler(logging.StreamHandler())
    return logger

def write_config_file(config, path):
    with open(os.path.join(path, 'config.yml'), 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)


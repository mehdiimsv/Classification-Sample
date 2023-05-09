from utils import mixup, partial_mixup, add_image_to_tensorboard
import torch
import numpy as np
from tqdm import tqdm


def train_one_epoch(do_every, training_loader, optimizer, model, loss_fn, tb_writer, mix_up_tools, device,
                    sam, mixed_precision, scheduler, epoch, scheduler_tools, logger, accumulate_iter):

    running_loss = 0.
    last_loss = 0.
    train_step = 0
    scheduler_final_epoch_flag = scheduler_tools['flag']
    scheduler_final_epoch = scheduler_tools['scheduler_final_epoch']
    fixed_final_lr = scheduler_tools['fixed_final_lr']
    scheduler_flag = scheduler_tools['flag']

    for i, data in enumerate(tqdm(training_loader)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        mix_up = mix_up_tools['mixup']
        # input_only = mix_up_tools['input_only']
        # alpha = mix_up_tools['alpha']
        # if self.train:
        # if mix_up:
        #     if input_only:
        #         inputs = partial_mixup(inputs, np.random.beta(alpha + 1, alpha),
        #                                torch.randperm(inputs.size(0), device=inputs.device, dtype=torch.long))
        #     else:
        #         inputs, labels = mixup(inputs, labels, np.random.beta(alpha, alpha))

        #tensors, labels = mixup(inputs, labels, torch.distributions.beta.Beta(mix_up_tools,
        #                                                                       mix_up_tools).sample().item()) if mix_up and epoch < 50 else (
        #    inputs, labels)

        #if mix_up and train_step == 0 and epoch == 0:
        #    add_image_to_tensorboard(tensors.cpu(), tb_writer, 'training images (with mixup)')

        loss = sam_mixed_precision_handling(sam, mixed_precision, optimizer, model, inputs, loss_fn, labels, i, training_loader,
                                            accumulate_iter)

        train_step += 1
        # Gather data and report
        running_loss += loss.item()
        if i % do_every == do_every-1:
            last_loss = running_loss / do_every  # loss per batch
            tb_x = epoch * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    if scheduler_flag:
        if not scheduler_final_epoch_flag:
            scheduler.step()
        else:
            if epoch <= scheduler_final_epoch:
                scheduler.step()
            else:
                optimizer.param_groups[0]['lr'] = fixed_final_lr

    tb_writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch+1)

    return last_loss


def sam_mixed_precision_handling(sam, mixed_precision, optimizer, model, inputs, loss_fn, labels, batch_idx, data_loader,
                                 accumulate_iter):
    if sam and not mixed_precision:
        optimizer.zero_grad()
        outputs = model(inputs)
        first_loss = loss_fn(outputs, labels)
        first_loss.backward()
        optimizer.first_step(zero_grad=True)
        loss = loss_fn(model(inputs), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1)
        optimizer.second_step(zero_grad=True)

    elif sam and mixed_precision:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            first_loss = loss_fn(outputs, labels)

        first_loss.backward()
        optimizer.first_step(zero_grad=True)

        with torch.cuda.amp.autocast():
            loss = loss_fn(model(inputs), labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1)
        optimizer.second_step(zero_grad=True)

    elif not sam and mixed_precision:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

    else:
        # optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1)
        if ((batch_idx + 1) % accumulate_iter == 0) or (batch_idx + 1 == len(data_loader)):
            optimizer.step()
            optimizer.zero_grad()
        # optimizer.step()

    return loss

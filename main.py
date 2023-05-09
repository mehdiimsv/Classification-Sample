import logging
import time
import argparse
import warnings
from datetime import datetime

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from custom_dataset import MyDataset
from train import train_one_epoch
from utils import *
from model import model_config
from evaluation import evaluation


def main(config):
    warnings.filterwarnings("ignore", category=UserWarning)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'runs/{timestamp}'
    writer = SummaryWriter(path)

    write_config_file(config, path)

    classes, epochs, do_every_train, train_batch, train_data_path, valid_data_path, csv_train_filename,\
    csv_valid_filename, model_name, is_pretrained, mixed_precision, device, sam_option, valid_batch, do_every_valid,\
    first_save, valid_dataloader_flag, mix_up_tools = get_config(config)

    logger = get_logger(logging, path)

    logger.info(f"Using {device} device")
    logger.info(f"Model name is {model_name}")
    logger.info(f"Optimizer is {config['optimizer']['name']}")

    # transform = make_transform(config['data']['train']['augmentation'])

    train_dataset = MyDataset(train_data_path, csv_train_filename, is_train=True, **config['data']['train']
    ['augmentation'])
    valid_dataset = MyDataset(valid_data_path, csv_valid_filename, is_train=False, **config['data']['train']
    ['augmentation'])

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch, shuffle=True, pin_memory=True)

    if valid_dataloader_flag:
        plot_dataloader(train_batch, train_dataloader, config['data']['train']['augmentation']['normalize'], classes)

    train_number = len(train_dataset)
    valid_number = len(valid_dataset)
    logging.info(f'Number of train samples: {train_number}')
    logging.info(f'Number of validation samples: {valid_number}')

    # net = nn.Sequential(*list(net.children())[:-3])
    # model = nn.Sequential(
    #     model,
    #     nn.Softmax(1)
    # )

    num_classes = len(classes.keys())
    net = model_config(model_name, is_pretrained, num_classes, device, logger)

    optimizer, scheduler = optimizer_config(net, config['optimizer'])

    loss_fn = torch.nn.CrossEntropyLoss()
    best_f1 = 0
    for epoch in range(epochs):
        logger.info(f'\nEPOCH {epoch + 1}:')

        first_time = time.time()
        avg_loss = train_one_epoch(do_every_train, train_dataloader, optimizer, net, loss_fn, writer,
                                   mix_up_tools, device, sam_option, mixed_precision, scheduler, epoch,
                                   config['optimizer']['scheduler']['end_epoch'], logger,
                                   config['optimizer']['accumulate_iter'])

        logger.info(f"One epoch takes {time.time()-first_time} seconds")
        y_prediction, y_true = [], []
        if epoch % do_every_valid == do_every_valid - 1:
            running_v_loss = 0.0

            with torch.no_grad():
                for i, v_data in enumerate(valid_dataloader):
                    v_inputs, v_labels = v_data
                    y_true.extend(v_labels)
                    v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)

                    v_outputs = net(v_inputs)
                    y_prediction.extend(torch.max(v_outputs.data, 1)[1].cpu())

                    v_loss = loss_fn(v_outputs, v_labels)
                    running_v_loss += v_loss

            avg_v_loss = running_v_loss / (valid_number + 1)
            logger.info(f'LOSS train {avg_loss} valid {avg_v_loss}')

            writer.add_scalars('Training vs. Validation Loss/',
                               {'Training': avg_loss, 'Validation': avg_v_loss}, epoch + 1)

            f1 = evaluation(y_true, y_prediction, epoch, writer, logger, classes)
            best_f1 = save_best_model(f1, best_f1, epoch, first_save, net, optimizer, avg_v_loss, path, logger,
                                      timestamp)

        writer.flush()

    logger.info('\nTraining Process Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='All needed parameters')
    parser.add_argument('--config_path', required=False, default='./config.yaml',
                        help='Config file path')

    args = parser.parse_args()

    with open(args.config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file)

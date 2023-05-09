from torchmetrics import Accuracy, Recall, Precision, F1, ConfusionMatrix
import torch

def evaluation(y_true, y_prediction, epoch, tb_writer, logger, classes):

    classes = tuple(classes.values())
    correct_prediction = dict.fromkeys(classes, 0)
    total_prediction = dict.fromkeys(classes, 0)

    y_true = torch.tensor(y_true)
    y_prediction = torch.tensor(y_prediction)

    for label, prediction in zip(y_true, y_prediction):
        class_name = classes[label]
        total_prediction[class_name] += 1
        if label == prediction:
            correct_prediction[class_name] += 1

    metrics = {
        'Accuracy': Accuracy,
        'Precision': Precision,
        'Recall': Recall,
        'F1': F1
    }

    for metric_name, metric_cls in metrics.items():
        metric = metric_cls(num_classes=len(classes))
        metric_value = metric(y_true, y_prediction) * 100
        logger.info(f'{metric_name} of the network on test images: {metric_value:.2f} %')
        tb_writer.add_scalar(f'{metric_name.lower()}', metric_value, epoch + 1)

    logger.info(f'\n')

    for class_name in classes:
        correct_count = correct_prediction[class_name]
        total_count = total_prediction[class_name]
        accuracy = 100 * float(correct_count) / total_count
        logger.info(f'Accuracy for class: {class_name:5s} is {accuracy:.2f} %')
        tb_writer.add_scalar(f'Class Accuracy/{class_name}', accuracy, epoch + 1)

    confmat = ConfusionMatrix(num_classes=len(classes))
    logger.info(confmat(y_prediction, y_true))

    logger.info('\n')

    return metric_value

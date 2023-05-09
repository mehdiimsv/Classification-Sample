import timm

def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def model_config(model_name, is_pretrained, num_classes ,device, logger):
    """
    Function for model config
    :param model_name: model name (based on timm models name)
    :param is_pretrained: using pre_trained or not
    :param num_classes: number of classes
    :param device: model device
    :param logger: logger
    :return: model
    """
    net = timm.create_model(model_name=model_name, pretrained=is_pretrained, num_classes=num_classes).to(device)
    logger.info(f"Number of parameters {count_parameters(net)}")

    return net


if __name__ == '__main__':
    model = model_config(model_name='convnext_base', is_pretrained=True, num_classes=5, device='cuda')
    print(model)
    print(count_parameters(model))

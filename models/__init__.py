import logging

from models.img_resnet import Part_Block, ResNet50, ResNet101

__factory = {"resnet50": ResNet50, "resnet101": ResNet101}


def build_model(config, num_classes):

    logger = logging.getLogger("reid.model")
    logger.info("Initializing model: {}".format(config.MODEL.NAME))
    if config.MODEL.NAME not in __factory.keys():
        raise KeyError("Invalid model: '{}'".format(config.MODEL.NAME))
    else:
        model = __factory[config.MODEL.NAME](config, num_classes)

    return model

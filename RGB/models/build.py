from .conv_mlp import ConvMLP

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'conv_mlp':
        model = SwinTransformer(
                                num_classes=config.MODEL.NUM_CLASSES
                                )
    
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model

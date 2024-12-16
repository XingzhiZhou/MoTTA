import torch
from yacs.config import CfgNode as cdict
from torchvision import models as pt_models
from motta import normalize_model, MoTTA



# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    mu = (0.485, 0.456, 0.406)
    sigma = (0.229, 0.224, 0.225)
    backbone = normalize_model(pt_models.resnet50(pretrained=True), mu, sigma)

    cfg = cdict(new_allowed=True)
    cfg.merge_from_file('config.yml')

    model = MoTTA(model=backbone, **cfg.paras_adapt_model)
    test_images = torch.rand(8, 3, 224, 224)

    model(test_images)

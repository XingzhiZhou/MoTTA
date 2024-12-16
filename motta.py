import os
import math
from collections import OrderedDict
from copy import deepcopy
from typing import Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.jit
from torch.nn.utils import prune

from easydict import EasyDict as edict

from robustbench.model_zoo.architectures.utils_architectures import normalize_model, ImageNormalizer
from memory_bank import DropMemoryBank
from loss import NegWeightedMutualInformation_on_marginal
from optimizer import build_optimizer

pruning_methods = {
    "l1_unstructured": prune.l1_unstructured,
    "ln_structured": prune.ln_structured,
    "random_unstructured": prune.random_unstructured,
    "random_structured": prune.random_structured
}


class MoTTA(nn.Module):
    def __init__(self, model, paras_optim, capacity, num_classes, bn_alpha, temp_factor,
                 update_frequency, confidence_threshold, uncertainty_threshold, prune_ratio, arch, dataset,
                 enable_robustBN, loss_name, paras_loss, steps=1,
                 episodic=False, memory_bank_type='uhus', freeze_top=True, use_buffer=True, fix_pruning_model=True,
                 pruning_strategy='l1_unstructured', pruning_module='conv', calculate_selection_mask=False,
                 category_uniform=True, record=False, metric_name='pruning_logit_norm_change', update_counter='each'):

        super().__init__()
        self.model = model
        self.paras_optim = paras_optim

        self.bn_alpha = bn_alpha
        self.update_frequency = update_frequency
        self.update_counter = update_counter
        self.enable_robustBN = enable_robustBN

        self.arch = arch
        self.dataset = dataset
        self.prune_ratio = prune_ratio
        self.metric_name = metric_name

        self.num_instance = 0

        self.steps = steps
        self.episodic = episodic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # memory

        self.memory = DropMemoryBank(capacity, num_classes, confidence_threshold, uncertainty_threshold,
                                     category_uniform)

        self.memory_copy = deepcopy(self.memory)

        # loss function
        self.loss_fn = NegWeightedMutualInformation_on_marginal(0.)

        # optimizer
        self.configure_model()
        params, param_names = self.collect_params(freeze_top=freeze_top)
        self.optimizer = build_optimizer(params, paras_optim)
        try:
            self.optimizer.set_model(self.model)
        except:
            print("optimizer does not have set_model method")

        # state copy
        self.model_state, self.optimizer_state = deepcopy(model.state_dict()), deepcopy(self.optimizer.state_dict())
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"

        self.fix_pruning_model = fix_pruning_model
        self.pruning_strategy = pruning_strategy
        self.pruning_module = pruning_module
        self.init_pruning(model, arch, dataset, prune_ratio)

        self.initial_weights = {name: param.clone().detach() for name, param in model.named_parameters() if
                                param.requires_grad}

        self.confidence_threshold, self.uncertainty_threshold = confidence_threshold, uncertainty_threshold
        self.use_buffer = use_buffer
        self.calculate_selection_mask = calculate_selection_mask
        self.selection_mask = []

        if record:
            self.record = {}
        else:
            self.record = None

    def configure_model(self):
        if self.enable_robustBN:
            print('using the robust BN mode')
            raise NotImplemented("not implement robust BN")
        else:
            print('using the training BN mode')
            for param in self.model.parameters():  # initially turn off requires_grad for all
                param.requires_grad = False
            for module in self.model.modules():
                if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = True
                    module.momentum = self.bn_alpha

                    module.weight.requires_grad_(True)
                    module.bias.requires_grad_(True)

    def collect_params(self, freeze_top=False):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        model = self.model
        params = []
        names = []
        for nm, m in model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if freeze_top:
                if 'layer4' in nm:
                    continue
                if 'blocks.9' in nm:
                    continue
                if 'blocks.10' in nm:
                    continue
                if 'blocks.11' in nm:
                    continue
                if 'norm.' in nm:
                    continue
                if nm in ['norm']:
                    continue

            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names

    def forward(self, x):
        if isinstance(x, dict):
            x = x['img']

        if self.episodic:
            self.reset()

        # inference
        # batch data
        with torch.no_grad():
            self.model.eval()
            out = self.model(x)
            prob = torch.softmax(out, dim=1)
            pseudo_label = torch.argmax(prob, dim=1)
            pseudo_conf = torch.max(prob, dim=1)[0]
            prune_result = self.eval_pruning(x)

            metric = prune_result[self.metric_name]

            if self.record is not None:
                self.record = prune_result
                self.record['pseudo_conf'] = pseudo_conf

        # update memory
        update_model_flag = False
        filtered_data = []
        for i, data in enumerate(x):

            p_l = pseudo_label[i].item()
            conf = pseudo_conf[i].item()
            uncertainty = metric[i].item()
            current_instance = edict(data=data.cpu(), prediction=p_l, uncertainty=uncertainty,
                                     confidence=conf)

            self.memory.add_instance(current_instance)

            if (not self.use_buffer) and conf >= self.confidence_threshold and metric[
                i].item() <= self.uncertainty_threshold:
                filtered_data.append(data.cpu())

            if self.update_counter == 'each':
                self.num_instance += 1
            else:
                if conf >= self.confidence_threshold and metric[i].item() <= self.uncertainty_threshold:
                    self.num_instance += 1

            if self.num_instance != 0 and self.num_instance % self.update_frequency == 0:
                update_model_flag = True

        # update model
        if update_model_flag:
            for _ in range(self.steps):
                self.update_model(filtered_data)

        # return outputs
        return dict(logits=out)

    @torch.enable_grad()
    def update_model(self, filtered_data):
        loss_fn = self.loss_fn

        if getattr(self, 'use_buffer', False):
            sup_data, sup_uncertainty = self.memory.get_memory()
        else:
            sup_data = filtered_data

        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data)
            sup_data = sup_data.to(self.device, non_blocking=True)

            self.model.train()

            if self.paras_optim['name'] == 'GAM':
                self.optimizer.set_unsup_closure(loss_fn, sup_data)

                self.optimizer.step()
            elif self.paras_optim['name'] == 'SAM':

                preds_of_data = self.model(sup_data)
                loss_first = loss_fn(preds_of_data)

                self.optimizer.zero_grad()

                loss_first.backward()

                # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
                self.optimizer.first_step(zero_grad=True)

                preds_of_data = self.model(sup_data)

                # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
                loss_second = loss_fn(preds_of_data)

                loss_second.backward()

                self.optimizer.second_step(zero_grad=True)

            if not self.fix_pruning_model:
                update_pruned_model(self.feature_extractor, self.feature_extractor_prune)

    def check_updates(self):
        is_update = is_updated(self.feature_extractor, self.feature_extractor_init)
        print(f"Feature_extractor is updated: {is_update}")

        is_update = is_updated(self.feature_extractor_prune, self.feature_extractor_prune_init)
        print(f"Feature_extractor_prune is updated: {is_update}")

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.memory = deepcopy(self.memory_copy)
        self.num_instance = 0
        self.num_consistent = 0
        # self.model_dropout = get_backbone_dropout(self.model, self.dropout_p)
        self.uncertainty_result = dict(uncertainty=[], domain=[])
        self.init_pruning(self.model, self.arch, self.dataset, self.prune_ratio)

    def eval_dropout(self, x, logits, probs, preds):
        num_dropout = self.num_dropout
        self.model_dropout.eval()

        # with drop inference
        x_repeat = torch.repeat_interleave(x, num_dropout, dim=0)
        logits_dropout = self.model_dropout(x_repeat)  # num_dropout batch_size * num_classes
        logits_dropout = logits_dropout.view(len(x), num_dropout, -1)
        probs_dropout = torch.softmax(logits_dropout, dim=2)
        preds_dropout = torch.argmax(probs_dropout, dim=2)  # batch_size * num_dropout

        mean_probs_dropout = torch.mean(probs_dropout, dim=1)  # batch_size * num_classes
        std_probs_dropout = torch.std(probs_dropout, dim=1)  # batch_size * num_classes
        mean_based_preds = torch.argmax(mean_probs_dropout, dim=1)
        preds = mean_based_preds
        mean_probs_dropout = mean_probs_dropout[torch.arange(len(preds)), preds]  # batch_size
        std_probs_dropout = std_probs_dropout[torch.arange(len(preds)), preds]  # batch_size

        mean_logits_dropout = torch.mean(logits_dropout, dim=1)  # batch_size * num_classes
        std_logits_dropout = torch.std(logits_dropout, dim=1)  # batch_size * num_classes
        mean_logits_dropout = mean_logits_dropout[torch.arange(len(preds)), preds]
        std_logits_dropout = std_logits_dropout[torch.arange(len(preds)), preds]

        max_logits = logits[torch.arange(len(preds)), preds]
        max_probs = probs[torch.arange(len(preds)), preds]
        change_logits = (mean_logits_dropout - max_logits).abs()
        change_probs = (mean_probs_dropout - max_probs).abs()

        consistency = (preds.unsqueeze(1) == preds_dropout).mean(dim=1, dtype=float)  # batch_size * num_dropout
        result = {'mean_logits_dropout': mean_logits_dropout, 'std_logits_dropout': std_logits_dropout,
                  'mean_probs_dropout': mean_probs_dropout, 'std_probs_dropout': std_probs_dropout,
                  'change_logits': change_logits, 'change_probs': change_probs, 'consistency': consistency}

        return result

    # eval pruning
    def init_pruning(self, model, arch, dataset, prune_ratio):
        if self.fix_pruning_model:
            self.feature_extractor, self.classifier = split_up_model(deepcopy(model), arch, dataset)

        else:
            self.feature_extractor, self.classifier = split_up_model(model, arch, dataset)

        self.feature_extractor_prune = deepcopy(self.feature_extractor)
        apply_pruning(self.feature_extractor_prune, strategy=self.pruning_strategy, amount=prune_ratio,
                      target_layers=self.pruning_module)

        # self.feature_extractor_prune_init = store_initial_state(self.feature_extractor_prune)
        # self.feature_extractor_init = store_initial_state(self.feature_extractor)

    @torch.no_grad()
    def eval_pruning(self, x):
        short_version = True
        feature_extractor = self.feature_extractor
        feature_extractor_prune = self.feature_extractor_prune
        classifier = self.classifier

        feature_extractor.eval()
        feature_extractor_prune.eval()

        feature = feature_extractor(x)
        feature_prune = feature_extractor_prune(x)

        if short_version:
            fc_weight = self.classifier.weight
            cos_sim = F.cosine_similarity(feature.unsqueeze(1), fc_weight.unsqueeze(0), dim=2)
            cos_sim_prune = F.cosine_similarity(feature_prune.unsqueeze(1), fc_weight.unsqueeze(0), dim=2)
            max_angle_change_feature_fc = torch.max(torch.abs(torch.acos(cos_sim) - torch.acos(cos_sim_prune)), dim=1)[
                0]
            max_angle_change_feature_fc = max_angle_change_feature_fc / math.pi * 180  # change to angle unit
            return dict(max_angle_change_feature_fc=max_angle_change_feature_fc.detach().cpu())

        logits = classifier(feature)
        logits_prune = classifier(feature_prune)

        score_3 = torch.norm(logits - logits_prune, dim=1)

        score_relative_change = torch.norm(logits - logits_prune, dim=1) / torch.norm(logits, dim=1)

        angle_change_logits = torch.acos(
            (logits * logits_prune).sum(dim=1) / (torch.norm(logits, dim=1) * torch.norm(logits_prune, dim=1)))

        max_logit_change = torch.max(torch.abs(logits - logits_prune), dim=1)[0]
        max_logit_index = torch.argmax(logits, dim=1)

        fc_weight = self.classifier.weight
        try:
            cos_sim = F.cosine_similarity(feature.unsqueeze(1), fc_weight, dim=2)
            cos_sim_prune = F.cosine_similarity(feature_prune.unsqueeze(1), fc_weight, dim=2)
        except Exception as e:
            print(e)
            print(feature.shape, fc_weight.shape)
            raise e

        angle_change_feature_fc = torch.abs(torch.acos(cos_sim) - torch.acos(cos_sim_prune)).sum(dim=1)

        max_angle_change_feature_fc = torch.max(torch.abs(torch.acos(cos_sim) - torch.acos(cos_sim_prune)), dim=1)[0]

        max_cos_index = torch.argmax(cos_sim, dim=1)
        angle_change_feature_fc_at_max_index = torch.abs(torch.acos(cos_sim) - torch.acos(cos_sim_prune))[
            torch.arange(len(max_logit_index)), max_cos_index]

        return dict(pruning_logit_norm_change=score_3.detach().cpu(),
                    pruning_logit_relative_change=score_relative_change.detach().cpu(),
                    angle_change_logits=angle_change_logits.detach().cpu(),
                    angle_change_feature_fc=angle_change_feature_fc.detach().cpu(),
                    max_angle_change_feature_fc=max_angle_change_feature_fc.detach().cpu(),
                    angle_change_feature_fc_at_max_index=angle_change_feature_fc_at_max_index.detach().cpu())


# Method to update the pruned model whenever the original model is updated
def update_pruned_model(original_model, pruned_model, ):
    for (name_orig, module_orig), (name_pruned, module_pruned) in zip(original_model.named_modules(),
                                                                      pruned_model.named_modules()):
        if isinstance(module_orig, (nn.BatchNorm2d)):
            # Copy parameters and buffers from the original module to the pruned module
            module_pruned.load_state_dict(module_orig.state_dict())


def split_up_model(model, arch_name: str, dataset_name: str):
    """
    Split up the model into an encoder and a classifier.
    This is required for methods like RMT and AdaContrast
    Input:
        model: Model to be split up
        arch_name: Name of the network
        dataset_name: Name of the dataset
    Returns:
        encoder: The encoder of the model
        classifier The classifier of the model
    """
    if hasattr(model, "model") and hasattr(model.model, "pretrained_cfg") and hasattr(model.model,
                                                                                      model.model.pretrained_cfg[
                                                                                          "classifier"]):
        # split up models loaded from timm
        classifier = deepcopy(getattr(model.model, model.model.pretrained_cfg["classifier"]))
        encoder = model
        encoder.model.reset_classifier(0)
        if isinstance(model, ImageNetXWrapper):
            encoder = nn.Sequential(encoder.normalize, encoder.model)

    elif arch_name == "Standard" and dataset_name in {"cifar10", "cifar10_c"}:
        encoder = nn.Sequential(*list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_WRN":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8),
                                nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_ResNeXt":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:2], nn.ReLU(), *list(model.children())[2:-1],
                                nn.Flatten())
        classifier = model.classifier
    elif dataset_name == "domainnet126":
        encoder = model.encoder
        classifier = model.fc
    elif "resnet" in arch_name or "resnext" in arch_name or "wide_resnet" in arch_name or arch_name in {"Standard_R50",
                                                                                                        "Hendrycks2020AugMix",
                                                                                                        "Hendrycks2020Many",
                                                                                                        "Geirhos2018_SIN"}:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.Flatten())
        classifier = model.model.fc
    elif "densenet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten())
        classifier = model.model.classifier
    elif "efficientnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool, nn.Flatten())
        classifier = model.model.classifier
    elif "mnasnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.layers, nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                nn.Flatten())
        classifier = model.model.classifier
    elif "shufflenet" in arch_name:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1],
                                nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.fc
    elif "vit_" in arch_name and not "maxvit_" in arch_name:
        encoder = TransformerWrapper(model)
        classifier = model.model.heads.head
    elif "swin_" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.norm, model.model.permute,
                                model.model.avgpool, model.model.flatten)
        classifier = model.model.head
    elif "convnext" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool)
        classifier = model.model.classifier
    elif arch_name == "mobilenet_v2":
        encoder = nn.Sequential(model.normalize, model.model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    else:
        raise ValueError(f"The model architecture '{arch_name}' is not supported for dataset '{dataset_name}'.")

    # add a masking layer to the classifier
    if dataset_name in ["imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_d109"]:
        from imagenet_subsets import ImageNetXMaskingLayer, IMAGENET_R_MASK, IMAGENET_V2_MASK, IMAGENET_A_MASK

        mask = eval(f"{dataset_name.upper()}_MASK")
        classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(mask))

    return encoder, classifier


# Function to check if the model or a specific module has been updated
def is_updated(model, initial_state):
    for name, param in model.named_parameters():
        if not torch.equal(param, initial_state[name]):
            return True
    for name, buffer in model.named_buffers():
        if not torch.equal(buffer, initial_state[name]):
            return True
    return False


class ImageNetXMaskingLayer(torch.nn.Module):
    """ Following: https://github.com/hendrycks/imagenet-r/blob/master/eval.py
    """

    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        return x[:, self.mask]


class ImageNetXWrapper(torch.nn.Module):
    def __init__(self, model, mask):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

        self.masking_layer = ImageNetXMaskingLayer(mask)

    def forward(self, x):
        logits = self.model(self.normalize(x))
        return self.masking_layer(logits)


class TransformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.normalize(x)
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x


# Function to apply pruning with different strategies
def apply_pruning(model, strategy="l1_unstructured", amount=0.2, target_layers="conv", n=1, dim=0):
    if strategy not in pruning_methods:
        raise ValueError(
            f"Pruning strategy {strategy} not recognized. Valid strategies: {list(pruning_methods.keys())}")

    pruning_method = pruning_methods[strategy]

    for name, module in model.named_modules():
        if (target_layers == "conv" and isinstance(module, nn.Conv2d)) or \
                (target_layers == "linear" and isinstance(module, nn.Linear)) or \
                (target_layers == "conv&linear" and isinstance(module, (nn.Conv2d, nn.Linear))):
            if "unstructured" in strategy:
                pruning_method(module, name='weight', amount=amount)  # Unstructured pruning
            else:
                pruning_method(module, name='weight', amount=amount, n=n, dim=dim)  # Structured pruning

            if module.bias is not None:
                if "structured" in strategy:
                    pruning_method(module, name='bias', amount=amount, n=n, dim=dim)  # Structured pruning
                else:
                    pruning_method(module, name='bias', amount=amount)  # Unstructured pruning


# This is a sample Python script.
class ImageNormalizer(nn.Module):

    def __init__(self, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std

    def __repr__(self):
        return f'ImageNormalizer(mean={self.mean.squeeze()}, std={self.std.squeeze()})'  # type: ignore


def normalize_model(model: nn.Module, mean: Tuple[float, float, float],
                    std: Tuple[float, float, float]) -> nn.Module:
    layers = OrderedDict([('normalize', ImageNormalizer(mean, std)),
                          ('model', model)])
    return nn.Sequential(layers)

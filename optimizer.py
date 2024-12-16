import torch
import torch.optim as optim
from torch.nn.modules.batchnorm import _BatchNorm

from easydict import EasyDict as edict
import torch
from torch.distributed import ReduceOp
import contextlib


# new level of building optimizer
def build_optimizer(params, paras_optim):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if paras_optim.name == 'Adam':
        return optim.Adam(params,
                          lr=paras_optim.lr,
                          betas=(paras_optim.beta, 0.999),
                          weight_decay=paras_optim.wd)
    elif paras_optim.name == 'SGD':
        return optim.SGD(params,
                         lr=paras_optim.lr,
                         momentum=paras_optim.momentum,
                         dampening=paras_optim.dampening,
                         weight_decay=paras_optim.weight_decay,
                         nesterov=paras_optim.nesterov)
    elif paras_optim.name == 'AdamW':
        return optim.AdamW(params,
                           lr=paras_optim.lr,
                           weight_decay=paras_optim.weight_decay, )
    elif paras_optim.name == "SAM":
        base_optimizer = None
        if paras_optim.base_optimizer == "Adam":
            base_optimizer = optim.Adam
        elif paras_optim.base_optimizer == "SGD":
            base_optimizer = optim.SGD
        return SAM(params, base_optimizer, lr=paras_optim.lr, rho=paras_optim.rho)
    elif paras_optim.name == 'GAM':
        base_optimizer = build_optimizer(params, paras_optim.base_optimizer)
        args = edict()
        args.grad_beta_0 = paras_optim.lambda_0_flat
        args.grad_beta_1 = paras_optim.lambda_0_flat
        args.grad_beta_2 = 1 - paras_optim.lambda_0_flat
        args.grad_beta_3 = 1 - paras_optim.lambda_0_flat
        args.grad_gamma = paras_optim.grad_gamma
        args.grad_rho = paras_optim.grad_rho
        args.grad_norm_rho = paras_optim.grad_norm_rho
        use_projection = paras_optim.get('use_projection', True)
        return GAM(params, base_optimizer, args=args, reg_strength=paras_optim.reg_strength,
                   use_projection=use_projection)
    else:
        raise NotImplementedError


class GAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, grad_rho_scheduler=None,
                 grad_norm_rho_scheduler=None, adaptive=False, perturb_eps=1e-12, args=None,
                 grad_reduce='mean', reg_strength=0, use_projection=True, **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(GAM, self).__init__(params, defaults)
        self.grad_rho_scheduler = grad_rho_scheduler
        self.grad_norm_rho_scheduler = grad_norm_rho_scheduler
        self.perturb_eps = perturb_eps
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.args = args
        self.get_grad_reduce(grad_reduce)

        self.grad_rho = args.grad_rho
        self.grad_norm_rho = args.grad_norm_rho
        self.reg_strength = reg_strength
        self.use_projection = use_projection

        self.update_rho_t()
        self.set_regularization_params()

    def set_model(self, model):
        self.model = model

    def set_regularization_params(self):
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]['original'] = p.data.clone()

    def get_grad_reduce(self, grad_reduce: str):
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def update_rho_t(self):

        if self.grad_rho_scheduler is not None:
            self.grad_rho = self.grad_rho_scheduler.step()
        if self.grad_norm_rho_scheduler is not None:
            self.grad_norm_rho = self.grad_norm_rho_scheduler.step()

    @torch.no_grad()
    def perturb_weights(self, perturb_idx: int):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        scale = self.grad_rho / (grad_norm + self.perturb_eps)

        if perturb_idx == 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["g_0"] = p.grad.data.clone()
                    e_w = p.grad * scale.to(p)
                    if self.adaptive:
                        e_w *= torch.pow(p, 2)
                    p.add_(e_w)
                    self.state[p]['e_w_0'] = e_w

        elif perturb_idx == 1:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["g_2"] = p.grad.data.clone()
                    e_w = p.grad * scale.to(p)
                    if self.adaptive:
                        e_w *= torch.pow(p, 2)
                    p.add_(e_w)
                    self.state[p]['e_w_1_2'] += e_w

        else:
            raise ValueError('"perturb_idx" should be one of [0, 1].')

    @torch.no_grad()
    def grad_norm_ascent(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["g_1"] = p.grad.data.clone()
                p.grad.data -= self.state[p]["g_0"]

        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        scale = self.grad_norm_rho / (grad_norm + self.perturb_eps)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)
                self.state[p]['e_w_1_2'] = e_w

    @torch.no_grad()
    def unperturb(self, perturb_key: str):

        for group in self.param_groups:
            for p in group['params']:
                if perturb_key in self.state[p].keys():
                    p.data.sub_(self.state[p][perturb_key])

    @torch.no_grad()
    def gradient_decompose(self, args=None):
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                # update the weighted sum of grads
                self.state[p]['pro_m'] = self.state[p]['g_0'] + abs(args.grad_beta_2) * self.state[p]['g_2']
                p.grad.data = args.grad_beta_1 * self.state[p][
                    "g_1"] + args.grad_beta_3 * p.grad.data.detach().clone()
                inner_prod += torch.sum(
                    self.state[p]['pro_m'] * p.grad.data
                )

        # get norm
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by='pro_m')

        # get cosine
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                vertical = self.state[p]['pro_m'] - cosine * old_grad_norm * p.grad.data / (
                        new_grad_norm + self.perturb_eps)

                if self.use_projection:
                    p.grad.data.add_(vertical, alpha=-args.grad_gamma)

                # regularization
                p.grad.data.add_(2 * self.reg_strength * (p.data - self.state[p]['original']))

    @torch.no_grad()
    def gradient_decompose_0th_flatness(self, args=None):
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                # update the weighted sum of grads
                self.state[p]['pro_m'] = self.state[p]['g_0']
                p.grad.data *= args.grad_beta_1  # g_1
                inner_prod += torch.sum(
                    self.state[p]['pro_m'] * p.grad.data
                )

        # get norm
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by='pro_m')

        # get cosine
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                vertical = self.state[p]['pro_m'] - cosine * old_grad_norm * p.grad.data / (
                        new_grad_norm + self.perturb_eps)

                if self.use_projection:
                    p.grad.data.add_(vertical, alpha=-args.grad_gamma)

                # regularization
                p.grad.data.add_(2 * self.reg_strength * (p.data - self.state[p]['original']))

    @torch.no_grad()
    def _grad_norm(self, weight_adaptive: bool = False, by: str = 'grad'):
        norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                if by == 'grad':
                    g = p.grad.data
                elif by == 'pro_m':
                    g = self.state[p]['pro_m']
                # elif by == 'e_w':
                #     g = self.state[p]['e_w_0'] + self.state[p]['e_w_1_2'] + self.state[p]['e_w_2']
                elif by == 'p':
                    g = p.data
                else:
                    raise ValueError("Invalid 'by' argument in _grad_norm")

                if weight_adaptive:
                    norm += torch.sum((g * torch.abs(p.data)) ** 2)
                else:
                    norm += torch.sum(g ** 2)

        return torch.sqrt(norm)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.

        def get_grad():
            self.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def set_unsup_closure(self, loss_fn, inputs, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.

        def get_grad():
            self.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    def step(self, closure=None):

        if self.args.grad_beta_2 == 0 or self.args.grad_beta_3 == 0:
            return self.step_0th_flatness(closure=closure)

        # if self.args.grad_beta_1 == 0 or self.args.grad_beta_1 < 1e-10:
        if self.args.grad_beta_1 < 1e-10:
            return self.step_directly(closure=closure)

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            outputs, loss_value = get_grad()

            # perturb weights
            self.perturb_weights(perturb_idx=0)

            # disable running stats for second pass,
            disable_running_stats(self.model)
            # model 1
            get_grad()
            # grad 1

            self.unperturb(perturb_key="e_w_0")
            # model 0
            self.grad_norm_ascent()
            # model 2
            get_grad()
            # grad 2

            self.perturb_weights(perturb_idx=1)
            # model 3
            get_grad()
            # grad 3

            # decompose and get new update direction
            self.gradient_decompose(args=self.args)

            # unperturb
            self.unperturb(perturb_key="e_w_1_2")

        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)

        return outputs, loss_value

    def step_0th_flatness(self, closure=None):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            outputs, loss_value = get_grad()

            # perturb weights
            self.perturb_weights(perturb_idx=0)

            # disable running stats for second pass,
            disable_running_stats(self.model)
            # model 1
            get_grad()
            # grad 1

            # decompose and get new update direction
            self.gradient_decompose_0th_flatness(args=self.args)

            self.unperturb(perturb_key="e_w_0")
            # model 0

        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)

        return outputs, loss_value

    def step_directly(self, closure=None):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            outputs, loss_value = get_grad()

            # gradient decomposition
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    # regularization
                    p.grad.data.add_(2 * self.reg_strength * (p.data - self.state[p]['original']))

        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.base_optimizer.step()

        return outputs, loss_value

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    # def add_param_group(self, param_group):
    #     self.base_optimizer.add_param_group(param_group)

    def __repr__(self):
        return f'GAM({self.base_optimizer.__class__.__name__})'


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class SAM(torch.optim.Optimizer):
    # from https://github.com/davda54/sam
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
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
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

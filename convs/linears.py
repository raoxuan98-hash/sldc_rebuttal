import torch
import torch.nn as nn
from copy import deepcopy
from torch.nn.init import trunc_normal_

class SimpleScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return x * self.weight

class SimpleContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes, feat_expand=False, with_norm=False, scale_mu=-1):
        super().__init__()

        self.embed_dim = embed_dim
        self.feat_expand = feat_expand
        self.with_norm = with_norm
        self.scale_mu = scale_mu

        if self.scale_mu > 0:
            scales = []
            scales.append(SimpleScaler())
            self.scales = nn.ModuleList(scales)
        else:
            self.scales = None

        heads = []
        single_head = []
        if with_norm:
            single_head.append(nn.LayerNorm(embed_dim))

        single_head.append(nn.Linear(embed_dim, nb_classes, bias=True))
        head = nn.Sequential(*single_head)

        heads.append(head)
        self.heads = nn.ModuleList(heads)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def update_scale(self):
        if self.scales is None:
            return
        num_old_tasks = len(self.heads) - 1
        for t_id in range(num_old_tasks):  # update scale for old tasks
            new_scale = 1 + self.scale_mu * (num_old_tasks - t_id)
            self.scales[t_id].weight.data = torch.tensor([new_scale]).to(self.scales[t_id].weight)

    def backup(self):
        self.old_state_dict = deepcopy(self.state_dict())

    def recall(self):
        self.load_state_dict(self.old_state_dict)

    def update(self, nb_classes, freeze_old=True):
        single_head = []
        if self.with_norm:
            single_head.append(nn.LayerNorm(self.embed_dim))

        if self.scale_mu > 0:
            self.scales.append(SimpleScaler())

        _fc = nn.Linear(self.embed_dim, nb_classes, bias=True)
        trunc_normal_(_fc.weight, std=.02)
        nn.init.constant_(_fc.bias, 0)
        single_head.append(_fc)
        new_head = nn.Sequential(*single_head)

        if freeze_old:
            for p in self.heads.parameters():
                p.requires_grad = False

        self.heads.append(new_head)

    def reset_heads(self):
        """
        Reset the parameters of all classification heads to their initial values.
        """
        for head in self.heads:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    trunc_normal_(module.weight, std=.02)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

    def forward(self, x):
        out = []
        for ti in range(len(self.heads)):
            fc_inp = x[ti] if self.feat_expand else x
            if self.scale_mu > 0:
                fc_inp = self.scales[ti](fc_inp)
            out.append(self.heads[ti](fc_inp))
        out = {'logits': torch.cat(out, dim=1)}
        return out
    

class BayesianLinearLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(BayesianLinearLayer, self).__init__()
        self.normal = torch.distributions.Normal(0, 1)
        self.input_size = input_size
        self.output_size = output_size
        
        self.weight_loc = torch.nn.Parameter(
            (torch.randn(input_size, output_size) * (2.0 / input_size)**0.5))
        
        self.weight_var_log = torch.nn.Parameter(
            (torch.log(torch.ones(input_size, output_size) * (2.0 / input_size))))
           
        self.register_buffer('weight_loc_prior', torch.zeros_like(self.weight_loc))
        self.register_buffer('weight_var_prior', deepcopy(torch.exp(self.weight_var_log).data))

    def forward(self, x, sample=False):
        mu = torch.matmul(x, self.weight_loc)
        if not sample:
            return mu
        else:
            std = torch.sqrt(torch.mm(x.pow(2), torch.exp(self.weight_var_log)))
            x = mu + torch.randn_like(std) * std
            return x
    
    def sample(self, loc, var_log):
        epsilon = self.normal.sample(loc.size()).to(self.device)
        std = torch.exp(var_log/2)
        return loc + std*epsilon
    
    def set_posterior_as_prior(self):
        self.weight_loc_prior = deepcopy(self.weight_loc.data).to(self.weight_loc.device)
        self.weight_var_prior = deepcopy(torch.exp(self.weight_var_log.data)).to(self.weight_loc.device)
            
    def cal_KL_divergence(self, loc_prior, var_prior, loc_posterior, var_posterior_log):
        var_posterior = torch.exp(var_posterior_log)
        var_prior_log = torch.log(var_prior)
        
        mean_regs = (torch.pow(loc_posterior - loc_prior, 2) / var_prior).sum()
        var_tr_regs = (var_posterior / var_prior).sum()
        var_log_regs = -(var_posterior_log - var_prior_log).sum()
        return mean_regs + var_tr_regs + var_log_regs
        
    def KL_divergence(self):
        kl = self.cal_KL_divergence(self.weight_loc_prior, self.weight_var_prior, self.weight_loc, self.weight_var_log)
        kl = 0.5*(kl - (self.input_size*self.output_size))
        return kl
    
    def cal_para_num(self):
        self.para_num = self.weight_loc.numel() + self.weight_var_log.numel()
        return self.para_num
    
class BayesContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes):
        super().__init__()

        self.embed_dim = embed_dim

        heads = []
        head = BayesianLinearLayer(embed_dim, nb_classes)
        heads.append(head)
        self.heads = nn.ModuleList(heads)
    def backup(self):
        self.old_state_dict = deepcopy(self.state_dict())

    def recall(self):
        self.load_state_dict(self.old_state_dict)

    def update(self, nb_classes, freeze_old=False):
        head = BayesianLinearLayer(self.embed_dim, nb_classes)
        if freeze_old:
            for p in self.heads.parameters():
                p.requires_grad = False
        self.heads.append(head)

    def forward(self, x, sample=False):
        out = []
        for ti in range(len(self.heads)):
            out.append(self.heads[ti](x, sample=sample))
        out = {'logits': torch.cat(out, dim=1)}
        return out

    def kl_divergence(self):
        total_kl = 0
        for head in self.heads:
            for module in head.modules():
                if isinstance(module, BayesianLinearLayer):
                    total_kl += module.KL_divergence()
        return total_kl

    def set_posterior_as_prior(self):
        for head in self.heads:
            for module in head.modules():
                if isinstance(module, BayesianLinearLayer):
                    module.set_posterior_as_prior()
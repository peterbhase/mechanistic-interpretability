import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import transformers
import higher
import logging
from higher.patch import monkeypatch as make_functional
from collections import defaultdict
import models.nn as local_nn

LOG = logging.getLogger(__name__)

class MENDNet(torch.nn.Module):
    def __init__(
        self,
        model,
        args=None,
        config=None,
        editable_params={},
        edit_lr=1e-3,
        embedding_init=None,
    ):
        super().__init__()
        self.model = model
        self.args = args
        self.config = config
        config.update(args.__dict__) # put console args into config
        self.editable_params = editable_params
        self.editable_named_params = [(n, p) for n, p in model.named_parameters() if n in editable_params]

        self.edit_lrs = nn.Parameter(torch.tensor(            
            [edit_lr] * len(self.editable_named_params)))

        # register hooks for keeping track of 1-rank gradient info -- THIS OCCURS IN learned_optimizer.forward now
        # if not hasattr(self.model, "handles"):
        #     hook_model(self.model, editable_params)
        #     print(f"Hooked {len(self.model.handles)//2} modules")
        
        # self-contained hook test
        # x = torch.tensor([10]).cuda().reshape(1,1)
        # y = torch.tensor([1]).cuda().reshape(1)
        # mask = (x>0).float()
        # self.model = model.to(self.args.device)
        # input = {'input_ids' : x, 'labels' : y, 'attention_mask' : mask}
        # import pdb; pdb.set_trace()
        # output = self.model(**input)
        # output.loss.backward()
        # print(self.model.model.encoder.layer[0].attention.self.query._forward_hooks)
        # print(self.model.model.encoder.layer[0].attention.self.query._backward_hooks)
        # print(self.model.model.encoder.layer[0].attention.self.query.weight.grad)
        # print(self.model.model.encoder.layer[0].attention.self.query.weight.__x__)
        # import pdb; pdb.set_trace()

        # make dict from unique shapes to param names, if parameter sharing across same weight sizes
        # NOTE THIS SEEMS LIKE A BUGGY IMPLEMENTATION WHEN COMBINED WITH get_param_idx BELOW
        if self.args.mend_weight_sharing:
            self.shape_dict = defaultdict(list)
            for n, p in self.editable_named_params:
                self.shape_dict[p.shape].append(n)

        if not self.args.mend_weight_sharing:
            self.mend = nn.ModuleDict({
                n.replace(".", "#"): GradientTransform(*p.shape, args, config['mend'])
                for n, p in self.editable_named_params
            })
        else:
            self.mend = nn.ModuleDict({
                str(tuple(s)): GradientTransform(*s, args, config['mend'], len(self.shape_dict[s])) # len(self.shape_dict) is number of per-weight scaling vectors of shape out-dim
                for s in self.shape_dict.keys()
            })

    def get_editable_named_params(self):
        return [(n, p) for n, p in self.model.named_parameters() if n in self.editable_params]

    def add_hooks_to_fmodel(self, fmodel):
        hook_model(fmodel, self.editable_params)

    def forward(self, loss):
        '''
        forward takes no input because we already set self.editable_named_params, and each param will store the __x__ and __delta__ components of its gradient
        following each call of loss.backward() on that model's loss
        '''

        loss.backward()

        if self.args.mend_weight_sharing:
            def get_param_idx(n, p): return self.shape_dict[p.shape].index(n)
            transformed_factors = {
                n: self.mend[str(tuple(p.shape))](p.__x__, p.__delta__, get_param_idx(n, p))
                for n, p in self.get_editable_named_params()
            }
            # for n, p in self.get_editable_named_params():
            #     print(n)
            #     print(p.shape)
            #     print(p.__x__.shape)
            #     print(p.__delta__.shape)
            #     import pdb; pdb.set_trace()
        else:
            transformed_factors = {
                n: self.mend[n.replace(".", "#")](p.__x__, p.__delta__)
                for n, p in self.get_editable_named_params()
            }

        # compute statistics
        mean_grads = {
            n: torch.einsum(f"bi,bj->ji", x, delta)
            for n, (x, delta) in transformed_factors.items()
        }
        info_dict = {}
        idx = 0
        for n, p in self.editable_named_params:
            info_dict[f"grad/true_mag{idx}"] = p.grad.norm(2).item()
            info_dict[f"grad/pseudo_mag{idx}"] = mean_grads[n].norm(2).item()
            info_dict[f"grad/true_std{idx}"] = p.grad.std().item()
            info_dict[f"grad/pseudo_std{idx}"] = mean_grads[n].std().item()
            info_dict[f"grad/diff{idx}"] = (p.grad - mean_grads[n]).norm(2).item()
            info_dict[f"grad/cos{idx}"] = F.cosine_similarity(p.grad.reshape(-1), mean_grads[n].reshape(-1), dim=0).item()
            idx += 1

        assert len(self.edit_lrs) == len(list(mean_grads.items()))
        updates = {n: lr * g for lr,
                   (n, g) in zip(self.edit_lrs, mean_grads.items())}

        self.model.zero_grad()

        return updates

def parent_module(model, pname):
    # returns 
    comps = pname.split('.')
    parent = model
    if hasattr(parent, 'model'):
        parent = model.model
    for comp in comps[:-1]:
        if hasattr(parent, comp):
            parent = getattr(parent, comp)
        elif comp.isdigit():
            parent = parent[int(comp)]
        else:
            import pdb; pdb.set_trace()
            raise RuntimeError(f"Couldn't find child module {comp}")
    assert hasattr(parent, comps[-1])
    return parent

def linear_backward_hook(mod, grad_in, grad_out):
    if not hasattr(mod, "weight"):
        print(f"{mod} has no weight!")
        return

    if hasattr(mod.weight, "__x__"):
        assert len(grad_out) == 1
        # mod.weight.__bgrad__ = grad_out[0].unsqueeze(-1) * mod.__x__[0].unsqueeze(-2)
        mod.weight.__delta__ = grad_out[0].detach()
    else:
        print(f"{mod} has no __x__")

def linear_forward_hook(mod, activations, output):
    assert len(activations) == 1
    mod.weight.__x__ = activations[0].detach()

def hook_model(model, pnames):
    handles = [] 
    for m, pname in [(parent_module(model, pname), pname) for pname in pnames]:
        handles.append(m.register_full_backward_hook(linear_backward_hook))
        handles.append(m.register_forward_hook(linear_forward_hook))
    model.handles = handles

def update_counter(x, m, s, k):
    new_m = m + (x - m) / k
    new_s = s + (x - m) * (x - new_m)
    return new_m, new_s

def _inner_params(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [(n, param_dict[n]) for n in inner_names]

class GradientTransform(nn.Module):
    def __init__(self, x_dim: int, delta_dim: int, args, cfg, n_modes=None):
        super().__init__()

        self.x_dim = x_dim
        self.delta_dim = delta_dim
        self.args = args
        self.cfg = cfg
        if cfg['combine'] and (cfg['one_sided'] or cfg['x_only'] or cfg['delta_only']):
            raise ValueError(
                "cfg.combine cannot be used with one-sided MEND variants")

        # had to switch delta_dim and x_dim relative to original implementation...
        self.register_buffer("u_mean", torch.full((delta_dim,), float("0")))
        self.register_buffer("v_mean", torch.full((x_dim,), float("0")))
        self.register_buffer("u_stddev",  torch.full((delta_dim,), float("0")))
        self.register_buffer("v_stddev",  torch.full((x_dim,), float("0")))    
        self.register_buffer("u_sumsquares",    torch.full((delta_dim,), float("0")))
        self.register_buffer("v_sumsquares",    torch.full((x_dim,), float("0")))

        self.register_buffer("k",      torch.full((1,), float("0")))

        MlpClass = getattr(local_nn, cfg['mlp_class'])

        '''
        class args for below are:
	        indim: int,
	        outdim: int,
	        hidden_dim: int,
	        n_hidden: int,
	        init: str = "xavier_uniform",
	        act: str = "relu",
	        rank: int = None,
        '''

        def delta_net():
            return MlpClass(delta_dim, delta_dim, delta_dim * 2, cfg['n_hidden'], init=cfg['init'], act=cfg['act'], rank=cfg['rank'], n_modes=n_modes)

        def x_net():
            return MlpClass(x_dim, x_dim, x_dim * 2, cfg['n_hidden'], init=cfg['init'], act=cfg['act'], rank=cfg['rank'], n_modes=n_modes)

        def combined_net():
            return MlpClass(delta_dim + x_dim, delta_dim + x_dim, (delta_dim + x_dim) * 2,
                            cfg['n_hidden'], init=cfg['init'], act=cfg['act'], rank=cfg['rank'], n_modes=n_modes)

        def ID():
            return lambda x, mode=None: x

        if cfg['combine']:
            self.mlp = combined_net()
        elif cfg['one_sided']:
            if x_dim > delta_dim:
                self.mlp1, self.mlp2 = ID(), delta_net()
            else:
                self.mlp1, self.mlp2 = x_net(), ID()
        elif cfg['x_only']:
            self.mlp1, self.mlp2 = x_net(), ID()
        elif cfg['delta_only']:
            self.mlp1, self.mlp2 = ID(), delta_net()
        else:
            self.mlp1, self.mlp2 = x_net(), delta_net()

    def forward(self, u, v, param_idx=None):
        u, v = u.to(torch.float32), v.to(torch.float32)

        u_ = u.reshape(-1, u.shape[-1])
        v_ = v.reshape(-1, v.shape[-1])

        # Skip batch elements with zero grad
        nz_mask = (u_ != 0).any(-1) * (v_ != 0).any(-1)
        u_ = u_[nz_mask]
        v_ = v_[nz_mask]
        
        if self.training:

            k_ = u_.shape[0]
            # in first step, no moving average
            use_k = self.k if self.k > 0 else k_
            self.u_mean = (k_ * torch.mean(u_, dim=0).detach() + (use_k - k_) * self.u_mean) / (use_k + k_)
            self.v_mean = (k_ * torch.mean(v_, dim=0).detach() + (use_k - k_) * self.v_mean) / (use_k + k_)
            self.u_sumsquares = torch.sum(u_ - self.u_mean, dim=0).detach() ** 2 + self.u_sumsquares
            self.v_sumsquares = torch.sum(v_ - self.v_mean, dim=0).detach() ** 2 + self.v_sumsquares
            self.k += k_

            # for idx in range(u_.shape[0]):
            #     if not self.cfg['input_norm']:
            #         self.u_mean = u_[idx].clone().detach()
            #         self.v_mean = v_[idx].clone().detach()
            #         self.u_sumsquares.zero_()
            #         self.v_sumsquares.zero_()
            #         self.k[:] = 1
            #         self.cfg['input_norm'] = self.cfg['input_norm']
            #     else:
            #         self.k += 1
            #         try:
            #             self.u_mean, self.u_sumsquares = update_counter(
            #                 u_[idx], self.u_mean, self.u_sumsquares, self.k)
            #         except:
            #             print("issue")
            #             print(u_[idx].shape)
            #             print(self.u_mean.shape)
            #             print(self.u_sumsquares.shape)
            #             print(self.k.shape)
            #             import pdb; pdb.set_trace()
            #         try:
            #             self.v_mean, self.v_sumsquares = update_counter(
            #                 v_[idx], self.v_mean, self.v_sumsquares, self.k)
            #         except:
            #             print("issue")
            #             print(v_[idx].shape)
            #             print(self.v_mean.shape)
            #             print(self.v_sumsquares.shape)
            #             print(self.k.shape)
            #             import pdb; pdb.set_trace()

            if self.args.input_norm and self.k > 1:
                self.u_stddev = (self.u_sumsquares / (self.k - 1)) ** 0.5
                self.v_stddev = (self.v_sumsquares / (self.k - 1)) ** 0.5

        if self.args.input_norm:
            u_input = (u_ - self.u_mean) / (self.u_stddev + 1e-7)
            v_input = (v_ - self.v_mean) / (self.v_stddev + 1e-7)
        else:
            u_input = u_
            v_input = v_

        if self.cfg['combine']:
            output = self.mlp(
                torch.cat((u_input, v_input), -1), mode=param_idx)
            out1, out2 = output.split([u.shape[-1], v.shape[-1]], -1)
            return out1, out2
        else:
            return self.mlp1(u_input, mode=param_idx), self.mlp2(v_input, mode=param_idx)


# class MEND(torch.nn.Module):
#     def get_shape(self, p):
#         # We need to flip the shapes since OpenAI gpt2 uses convs instead of linear
#         return p.shape if isinstance(self.model, transformers.GPT2LMHeadModel) else (p.shape[1], p.shape[0])

#     def __init__(self, model, config, model_constructor, mend=None, edit_lrs=None):
#         super().__init__(model, config, model_constructor)

#         if edit_lrs is None:
#             edit_lrs = nn.Parameter(torch.tensor(
#                 [config.edit_lr] * len(self.config.model.inner_params)))
#         self.edit_lrs = edit_lrs

#         if not hasattr(self.model, "handles"):
#             hook_model(self.model, self.config.model.inner_params)
#             LOG.info(f"Hooked {len(self.model.handles)//2} modules")

#         if config.mend.shared:
#             shape_dict = defaultdict(list)
#             for n, p in _inner_params(model.named_parameters(), self.config.model.inner_params):
#                 shape_dict[self.get_shape(p)].append(n)
#             self.shape_dict = shape_dict

#         if mend is None:
#             if not config.mend.shared:
#                 self.mend = nn.ModuleDict({
#                     n.replace(".", "#"): GradientTransform(*self.get_shape(p), config.mend)
#                     for (n, p) in _inner_params(model.named_parameters(), self.config.model.inner_params)
#                 })
#             else:
#                 self.mend = nn.ModuleDict({
#                     str(tuple(s)): GradientTransform(*s, config.mend, len(shape_dict[s]))
#                     for s in shape_dict.keys()
#                 })
#         else:
#             self.mend = mend

    # def state_dict(self, destination=None, prefix="", keep_vars=False):
    #     state_dict = super().state_dict(prefix=prefix, keep_vars=keep_vars)  # Get default state dict
    #     model_keys = self.model.state_dict(prefix=prefix, keep_vars=keep_vars).keys()  # Remove model params
    #     for k in model_keys:
    #         del state_dict[f"model.{k}"]
    #     state_dict["model_config"] = self.model.config  # Include model config
    #     return state_dict

    # def load_state_dict(self, state_dict, strict: bool = True):
    #     config = state_dict["model_config"]
    #     del state_dict["model_config"]
    #     if config != self.model.config:
    #         LOG.info("Loaded model config doesn't match current model config.")
    #         LOG.info(f"Loaded: {config}")
    #         LOG.info(f"Current: {self.model.config}")

    #     res = super().load_state_dict(state_dict, False)
    #     # We should only have missing keys for the model, and no unexpected keys
    #     assert len([k for k in res.missing_keys if not k.startswith("model.")]) == 0, "Should only have missing keys for model."
    #     assert len(res.unexpected_keys) == 0, "Shouldn't have any unexpected keys"
    #     return res

    # def outer_parameters(self):
    #     return list(self.mend.parameters()) + [self.edit_lrs]

    # def edit(self, batch, condition=None, detach_history=False):
    #     outputs = self.model(**batch)
    #     outputs = outputs.logits
    #     loss = self.edit_loss_fn(outputs, batch["labels"])["nll"]

    #     names = set([n for n, p in self.model.named_parameters()])
    #     pset = set(self.config.model.inner_params)
    #     for p in pset:
    #         assert p in names, f"inner param {p} not in model"

    #     loss.backward()

    #     if self.config.mend.shared:
    #         def param_idx(n, p): return self.shape_dict[self.get_shape(p)].index(n) if self.config.mend.shared else None  # noqa: E731
    #         transformed_factors = {
    #             n: self.mend[str(tuple(self.get_shape(p)))](p.__x__, p.__delta__, param_idx(n, p))
    #             for n, p in _inner_params(self.model.named_parameters(), self.config.model.inner_params)
    #         }
    #     else:
    #         transformed_factors = {
    #             n: self.mend[n.replace(".", "#")](p.__x__, p.__delta__)
    #             for n, p in _inner_params(self.model.named_parameters(), self.config.model.inner_params)
    #         }

    #     # Should be bi,bj->ji for nn.Linear, but GPT2 uses Conv1d instead...
    #     if isinstance(self.model, transformers.GPT2LMHeadModel):
    #         targ = "ij"
    #     else:
    #         targ = "ji"
    #     mean_grads = {
    #         n: torch.einsum(f"bi,bj->{targ}", x, delta)
    #         for n, (x, delta) in transformed_factors.items()
    #     }

    #     info_dict = {}
    #     idx = 0
    #     for n, p in _inner_params(self.model.named_parameters(), self.config.model.inner_params):
    #         info_dict[f"grad/true_mag{idx}"] = p.grad.norm(2).item()
    #         info_dict[f"grad/pseudo_mag{idx}"] = mean_grads[n].norm(2).item()
    #         info_dict[f"grad/true_std{idx}"] = p.grad.std().item()
    #         info_dict[f"grad/pseudo_std{idx}"] = mean_grads[n].std().item()
    #         info_dict[f"grad/diff{idx}"] = (p.grad - mean_grads[n]).norm(2).item()
    #         info_dict[f"grad/cos{idx}"] = F.cosine_similarity(p.grad.reshape(-1), mean_grads[n].reshape(-1), dim=0).item()
    #         idx += 1

    #     self.model.zero_grad()

    #     assert len(self.edit_lrs) == len(list(mean_grads.items()))
    #     updates = {n: lr * g for lr,
    #                (n, g) in zip(self.edit_lrs, mean_grads.items())}

    #     edited_model = self.model
    #     if not isinstance(edited_model, higher.patch._MonkeyPatchBase):
    #         edited_model = make_functional(edited_model, in_place=True)

    #     new_params = []
    #     for n, p in edited_model.named_parameters():
    #         if n in pset:
    #             new_params.append(p + updates[n])
    #         else:
    #             new_params.append(p)

    #     edited_model.update_params(new_params)

    #     if detach_history:
    #         new_model = self.model_constructor()
    #         new_model.load_state_dict(edited_model.state_dict())
    #         edited_model = new_model

    #     return MEND(edited_model, self.config, self.model_constructor, self.mend, edit_lrs=self.edit_lrs), info_dict


if __name__ == '__main__':
    import types

    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")

    config = types.SimpleNamespace()
    config.model.inner_params = [
        "transformer.h.9.mlp.c_fc.weight",
        "transformer.h.9.mlp.c_proj.weight",
        "transformer.h.10.mlp.c_fc.weight",
        "transformer.h.10.mlp.c_proj.weight",
        "transformer.h.11.mlp.c_fc.weight",
        "transformer.h.11.mlp.c_proj.weight",
    ]
    config.edit_lr = 0.0001

    config.mend = types.SimpleNamespace()
    config.mend.n_hidden = 1
    config.mend = config.mend.__dict__

    mend = MEND(model, config, lambda: copy.deepcopy(model)).cuda()
    import pdb; pdb.set_trace()
    mend.load_state_dict(torch.load("test_state.pt"))
    x = torch.arange(20).view(1, 20).cuda() + 1000
    orig_logits = mend(x)
    edited = mend.edit(x, masks=torch.ones_like(x), labels=x)
    post_logits = mend(x)

    assert torch.allclose(orig_logits, post_logits)

    orig_param = [p for (n, p) in mend.model.named_parameters()
                  if n == config.model.inner_params[-1]][0]
    edited_param = [p for (n, p) in edited.model.named_parameters(
    ) if n == config.model.inner_params[-1]][0]

    LOG.info((orig_param - edited_param).abs().max())
    edited.eval()
    LOG.info(mend(x, labels=x).loss, edited(x, labels=x).loss,
             edited.edit_loss_fn(edited(x).logits, x)["nll"])
    edited2 = edited.edit(x, masks=torch.ones_like(x), labels=x)
    LOG.info(mend(x, labels=x).loss, edited(
        x, labels=x).loss, edited2(x, labels=x).loss)

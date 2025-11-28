import torch as th
import numpy as np
import logging
import torch.distributed as dist

import enum

from . import path
from .utils import EasyDict, log_state, mean_flat
from .integrators import ode, sde

# SIGReg support from LeJEPA
try:
    import lejepa
    LEJEPA_AVAILABLE = True
except ImportError:
    LEJEPA_AVAILABLE = False
    logging.warning("lejepa package not installed. SIGReg loss will be unavailable.")

class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)

class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()

class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
        # SIGReg parameters
        use_sigreg=False,
        sigreg_lambda=0.05,
        sigreg_num_slices=1024,
        # Auxiliary losses for attention improvement (2025 research)
        use_aux_losses=False,
        aux_entropy_floor_threshold=0.3,
        aux_entropy_floor_weight=0.02,
        aux_entropy_ceiling_threshold=0.85,
        aux_entropy_ceiling_weight=0.02,
        aux_gate_entropy_weight=0.01,
        aux_gate_sparsity_weight=0.005,
        aux_hsic_weight=0.01,
        aux_position_disagreement_weight=0.005,
        aux_lambda_smoothness_weight=0.001,
        aux_lambda_entropy_weight=0.01,
        aux_warmup_steps=1000,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

        # SIGReg initialization
        self.use_sigreg = use_sigreg and LEJEPA_AVAILABLE
        self.sigreg_lambda = sigreg_lambda
        if self.use_sigreg:
            univariate_test = lejepa.univariate.EppsPulley()
            self.sigreg_loss_fn = lejepa.multivariate.SlicingUnivariateTest(
                univariate_test=univariate_test,
                num_slices=sigreg_num_slices
            )
            logging.info(f"SIGReg initialized with lambda={sigreg_lambda}, slices={sigreg_num_slices}")

        # Auxiliary losses initialization (2025 research: σReparam, GateRA, HSIC, D-Gating)
        self.use_aux_losses = use_aux_losses
        if use_aux_losses:
            from pokemon_eqm.losses import AuxiliaryLossComputer
            self.aux_loss_computer = AuxiliaryLossComputer(
                entropy_floor_threshold=aux_entropy_floor_threshold,
                entropy_floor_weight=aux_entropy_floor_weight,
                entropy_ceiling_threshold=aux_entropy_ceiling_threshold,
                entropy_ceiling_weight=aux_entropy_ceiling_weight,
                gate_entropy_weight=aux_gate_entropy_weight,
                gate_sparsity_weight=aux_gate_sparsity_weight,
                hsic_weight=aux_hsic_weight,
                position_disagreement_weight=aux_position_disagreement_weight,
                lambda_smoothness_weight=aux_lambda_smoothness_weight,
                lambda_entropy_weight=aux_lambda_entropy_weight,
                warmup_steps=aux_warmup_steps,
            )
            logging.info(f"Auxiliary losses initialized (entropy: {aux_entropy_floor_weight}/{aux_entropy_ceiling_weight}, "
                        f"gate: {aux_gate_entropy_weight}/{aux_gate_sparsity_weight}, hsic: {aux_hsic_weight})")

    def prior_logp(self, z):
        '''
            Standard multivariate normal prior
            Assume z is batched
        '''
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.
        return th.vmap(_fn)(z)
    

    def check_interval(
        self, 
        train_eps, 
        sample_eps, 
        *, 
        diffusion_form="SBDM",
        sde=False, 
        reverse=False, 
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
            and (self.model_type != ModelType.VELOCITY or sde): # avoid numerical issue by taking a first semi-implicit step

            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        
        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1


    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
        
        x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        t = t.to(x1)
        return t, x0, x1

    def disp_loss(self, z): # Dispersive Loss implementation (InfoNCE-L2 variant)
        z = z.reshape((z.shape[0],-1)) # flatten
        diff = th.nn.functional.pdist(z).pow(2)/z.shape[1] # normalize by dimension
        diff = th.concat((diff, diff, th.zeros(z.shape[0]).cuda()))  # match JAX implementation of full BxB matrix
        return th.log(th.exp(-diff).mean())

    def sigreg_loss(self, registers):
        """
        Compute SIGReg loss on register tokens (LeJEPA-style).

        Following LeJEPA, we treat each register position as a separate "view"
        and apply SIGReg independently to each, then average. This encourages
        each register to independently follow an isotropic Gaussian distribution.

        Input shape: [N, num_registers, D]
        Reshaped to: [num_registers, N, D] (registers as views)
        Each view [N, D] is regularized independently.

        Args:
            registers: Tensor of shape [N, num_registers, D] from the model

        Returns:
            Scalar loss value (averaged across register positions)
        """
        if not self.use_sigreg or registers is None:
            return 0.0

        # Reshape: [N, num_reg, D] -> [num_reg, N, D] (treat registers as views)
        registers = registers.permute(1, 0, 2)  # [num_reg, N, D]

        # Apply SIGReg to each register position and average
        total_loss = 0.0
        num_registers = registers.shape[0]
        for i in range(num_registers):
            total_loss = total_loss + self.sigreg_loss_fn(registers[i])  # [N, D]

        return total_loss / num_registers

    def get_ct(self, t): #ct implementation
        interp = 0.8
        start = 1.0
        ct = th.minimum(start-(start-1)/(interp)*t, 1/(1-interp)-1/(1-interp)*t)*4
        return ct

    def training_losses(
        self,
        model,
        x1,
        model_kwargs=None,
        train_step=0,  # For auxiliary loss warmup
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        - train_step: current training step (for auxiliary loss warmup)
        """
        if model_kwargs == None:
            model_kwargs = {}

        # Request registers if SIGReg is enabled
        if self.use_sigreg:
            model_kwargs['return_registers'] = True

        # Request auxiliary info if aux losses are enabled
        if self.use_aux_losses:
            model_kwargs['return_aux_info'] = True
            model_kwargs['attention_layer_idx'] = -1  # Last layer (or could use middle layer)

        t, x0, x1 = self.sample(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        ut = ut * self.get_ct(t)[:,None,None,None] # use energy-compatible target
        model_output = model(xt, t, **model_kwargs)
        disp_loss = 0
        sigreg_loss_value = 0
        aux_loss_dict = {}
        aux_loss_total = 0
        registers = None
        aux_info = None

        # get intermediate activation and apply Dispersive Loss
        if "return_act" in model_kwargs and model_kwargs['return_act']:
            if self.use_sigreg or self.use_aux_losses:
                if self.use_sigreg and self.use_aux_losses:
                    # Model returns 4 values: (output, act, registers, aux_info)
                    model_output, act, registers, aux_info = model_output
                elif self.use_sigreg:
                    model_output, act, registers = model_output
                else:  # use_aux_losses only
                    model_output, act, aux_info = model_output
            else:
                model_output, act = model_output
            disp_loss = self.disp_loss(act[len(act)-1])
        elif self.use_sigreg or self.use_aux_losses:
            if self.use_sigreg and not self.use_aux_losses:
                model_output, registers = model_output
            elif self.use_aux_losses and not self.use_sigreg:
                model_output, aux_info = model_output
            else:  # Both enabled - model returns (output, registers, aux_info)
                model_output, registers, aux_info = model_output

        # Compute SIGReg loss on register tokens
        if self.use_sigreg and registers is not None:
            sigreg_loss_value = self.sigreg_loss(registers)

        # Compute auxiliary losses (entropy, gate, diversity, lambda regularization)
        if self.use_aux_losses and aux_info is not None:
            # Extract components from aux_info dict
            attn_weights = None
            head_outputs = None
            gate_values = None
            lambda_matrix = None
            entropy_normalized = None

            if isinstance(aux_info, dict):
                # For DifferentialAttention: aux_info has 'attn1', 'attn2', 'head_outputs', 'gate_values', 'lambda_matrix'
                attn_weights = aux_info.get('attn1', aux_info.get('attn', None))
                head_outputs = aux_info.get('head_outputs', None)
                gate_values = aux_info.get('gate_values', None)
                lambda_matrix = aux_info.get('lambda_matrix', None)

                # Compute entropy from attention weights if available
                if attn_weights is not None:
                    # Compute normalized entropy: H / log(N)
                    eps = 1e-7
                    attn_for_entropy = attn_weights.clamp(eps, 1 - eps)
                    entropy_per_head = -th.sum(attn_for_entropy * th.log(attn_for_entropy), dim=-1)  # [B, H, N]
                    max_entropy = th.log(th.tensor(attn_weights.shape[-1], dtype=attn_weights.dtype, device=attn_weights.device))
                    entropy_normalized = (entropy_per_head / max_entropy).mean()

            # Compute all aux losses
            aux_loss_dict = self.aux_loss_computer(
                train_step=train_step,
                attn_weights=attn_weights,
                head_outputs=head_outputs,
                gate_values=gate_values,
                lambda_matrix=lambda_matrix,
                entropy_normalized=entropy_normalized,
            )
            aux_loss_total = aux_loss_dict.get('total', 0)

        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        terms = {}
        terms['pred'] = model_output
        if self.model_type == ModelType.VELOCITY:
            terms['loss'] = mean_flat(((model_output - ut) ** 2))
        else:
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()

            if self.model_type == ModelType.NOISE:
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))

        # Add auxiliary losses (dispersive + SIGReg + attention aux losses)
        terms['loss'] += 0.5 * disp_loss + self.sigreg_lambda * sigreg_loss_value + aux_loss_total
        terms['disp_loss'] = disp_loss
        terms['sigreg_loss'] = sigreg_loss_value

        # Add individual aux loss components for logging
        for key, value in aux_loss_dict.items():
            if key != 'total':
                terms[f'aux_{key}'] = value if th.is_tensor(value) else th.tensor(value)
        terms['aux_loss_total'] = aux_loss_total if th.is_tensor(aux_loss_total) else th.tensor(aux_loss_total)

        return terms
    

    def get_drift(
        self
    ):
        """member function for obtaining the drift of the probability flow ODE"""
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output) # by change of variable
        
        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)
        
        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode
        
        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn
    

    def get_score(
        self,
    ):
        """member function for obtaining score of 
            x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        else:
            raise NotImplementedError()
        
        return score_fn


class Sampler:
    """Sampler class for the transport model"""
    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """
        
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()
    
    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):

        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion
        
        sde_drift = \
            lambda x, t, model, **kwargs: \
                self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
    
        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion
    
    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""
    
        if last_step is None:
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x
        elif last_step == "Mean":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)
            

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample
    
    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )
        
        return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):
        
        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """
        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)
        
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn
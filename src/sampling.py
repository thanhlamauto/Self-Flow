"""
Self-Flow Sampling Utilities (JAX version).

This module contains the sampling logic for Self-Flow diffusion models,
including the SDE integrators and transport path definitions, converted to JAX.
"""

import enum
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import jax
import jax.numpy as jnp


Choice_PathType = Literal["Linear", "GVP", "VP"]
Choice_Prediction = Literal["velocity", "score", "noise"]
Choice_LossWeight = Optional[Literal["velocity", "likelihood"]]
Choice_SamplingODE = Literal["heun2"]
Choice_SamplingSDE = Literal["Euler", "Heun"]
Choice_Diffusion = Literal[
    "constant", "SBDM", "sigma",
    "linear", "decreasing", "increasing-decreasing"
]
Choice_LastStep = Optional[Literal["Mean", "Tweedie", "Euler"]]


@dataclass
class TransportConfig:
    path_type: Choice_PathType = "Linear"
    prediction: Choice_Prediction = "velocity"
    loss_weight: Choice_LossWeight = None
    sample_eps: Optional[float] = None
    train_eps: Optional[float] = None


@dataclass
class ODEConfig:
    sampling_method: Choice_SamplingODE = "heun2"
    atol: float = 1e-6
    rtol: float = 1e-3
    reverse: bool = False
    likelihood: bool = False


@dataclass
class SDEConfig:
    sampling_method: Choice_SamplingSDE = "Euler"
    diffusion_form: Choice_Diffusion = "sigma"
    diffusion_norm: float = 1.0
    last_step: Choice_LastStep = "Mean"
    last_step_size: float = 0.04


@dataclass
class Config:
    transport: TransportConfig = field(default_factory=TransportConfig)
    ode: ODEConfig = field(default_factory=ODEConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    num_steps: int = 64
    cfg_scale: float = 1


def expand_t_like_x(t, x):
    dims = [1] * (len(x.shape) - 1)
    t = t.reshape(t.shape[0], *dims)
    return t


class ICPlan:
    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def compute_alpha_t(self, t):
        return t, 1.0

    def compute_sigma_t(self, t):
        return 1.0 - t, -1.0

    def compute_d_alpha_alpha_ratio_t(self, t):
        return 1.0 / t

    def compute_drift(self, x, t):
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t
        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        t = expand_t_like_x(t, x)
        if form == "constant":
            return jnp.ones_like(t) * norm
        elif form == "SBDM":
            return norm * self.compute_drift(x, t)[1]
        elif form == "sigma":
            return norm * self.compute_sigma_t(t)[0]
        elif form == "linear":
            return norm * (1.0 - t)
        elif form == "decreasing":
            return 0.25 * (norm * jnp.cos(jnp.pi * t) + 1.0) ** 2
        elif form == "increasing-decreasing":
            return norm * jnp.sin(jnp.pi * t) ** 2
        else:
            raise NotImplementedError()

    def get_score_from_velocity(self, velocity, x, t):
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score


class ModelType(enum.Enum):
    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class PathType(enum.Enum):
    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:
    def __init__(self, *, model_type, path_type, loss_type, train_eps, sample_eps):
        path_options = {
            PathType.LINEAR: ICPlan,
        }
        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def check_interval(self, train_eps, sample_eps, *, diffusion_form="SBDM", sde=False, reverse=False, eval=False, last_step_size=0.0):
        t0 = 0.0
        t1 = 1.0
        eps = train_eps if not eval else sample_eps

        if self.model_type != ModelType.VELOCITY or sde:
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0.0
            t1 = 1.0 - eps if (not sde or last_step_size == 0) else 1.0 - last_step_size

        if reverse:
            t0, t1 = 1.0 - t0, 1.0 - t1

        return t0, t1

    def get_drift_from_model_output(self):
        def velocity_ode(x, t, model_output):
            return model_output
        return velocity_ode

    def get_score_from_model_output(self):
        return lambda x, t, model_output: self.path_sampler.get_score_from_velocity(model_output, x, t)


def create_transport(path_type='Linear', prediction="velocity", loss_weight=None, train_eps=None, sample_eps=None):
    model_type = ModelType.VELOCITY
    loss_type = WeightType.NONE
    path_choice = {"Linear": PathType.LINEAR}
    path_type = path_choice[path_type]
    train_eps = 0.0
    sample_eps = 0.0
    return Transport(model_type=model_type, path_type=path_type, loss_type=loss_type, train_eps=train_eps, sample_eps=sample_eps)


class sde:
    def __init__(self, drift, diffusion, *, t0, t1, num_steps, sampler_type):
        assert t0 < t1, "SDE sampler has to be in forward time"
        self.num_timesteps = num_steps
        self.t = jnp.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def sample(self, init, rng, model_fn):
        def apply_drift(x, t):
            model_out = model_fn(x, t)
            return self.drift(x, t, model_out)

        def Euler_Maruyama_step(carry, t):
            x, mean_x, rng = carry
            rng, step_rng = jax.random.split(rng)
            w_cur = jax.random.normal(step_rng, x.shape)
            t_batch = jnp.ones(x.shape[0]) * t
            dw = w_cur * jnp.sqrt(self.dt)
            
            drift = apply_drift(x, t_batch)
            diffusion = self.diffusion(x, t_batch)
            mean_x = x + drift * self.dt
            x_next = mean_x + jnp.sqrt(2 * diffusion) * dw
            
            return (x_next, mean_x, rng), x_next

        def Heun_step(carry, t):
            x, mean_x, rng = carry
            rng, step_rng = jax.random.split(rng)
            w_cur = jax.random.normal(step_rng, x.shape)
            dw = w_cur * jnp.sqrt(self.dt)
            t_batch = jnp.ones(x.shape[0]) * t
            
            diffusion = self.diffusion(x, t_batch)
            xhat = x + jnp.sqrt(2 * diffusion) * dw
            
            K1 = apply_drift(xhat, t_batch)
            xp = xhat + self.dt * K1
            K2 = apply_drift(xp, t_batch + self.dt)
            
            x_next = xhat + 0.5 * self.dt * (K1 + K2)
            return (x_next, xhat, rng), x_next

        sampler_fn = Euler_Maruyama_step if self.sampler_type == "Euler" else Heun_step
        
        carry = (init, init, rng)
        (x_final, mean_x_final, rng_final), history = jax.lax.scan(sampler_fn, carry, self.t[:-1])
        return history


class FixedSampler:
    def __init__(self, transport):
        self.transport = transport
        self.drift = self.transport.get_drift_from_model_output()
        self.score = self.transport.get_score_from_model_output()

    def __get_sde_diffusion_and_drift(self, *, diffusion_form="SBDM", diffusion_norm=1.0):
        def diffusion_fn(x, t):
            return self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)

        def sde_drift(x, t, model_output):
            return self.drift(x, t, model_output) + diffusion_fn(x, t) * self.score(x, t, model_output)

        return sde_drift, diffusion_fn

    def __get_last_step(self, sde_drift, *, last_step, last_step_size):
        if last_step is None:
            last_step_fn = lambda x, t, model_output: x
        elif last_step == "Mean":
            last_step_fn = lambda x, t, model_output: x + sde_drift(x, t, model_output) * last_step_size
        elif last_step == "Euler":
            last_step_fn = lambda x, t, model_output: x + self.drift(x, t, model_output) * last_step_size
        else:
            raise NotImplementedError()
        return last_step_fn

    def sample_sde(self, *, sampling_method="Euler", diffusion_form="SBDM", diffusion_norm=1.0, last_step="Mean", last_step_size=0.04, num_steps=250):
        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(diffusion_form=diffusion_form, diffusion_norm=diffusion_norm)

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
            sampler_type=sampling_method,
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(init, rng, model_fn):
            xs = _sde.sample(init, rng, model_fn)
            t_last = jnp.ones(init.shape[0]) * t1
            x_last = xs[-1]
            model_out = model_fn(x_last, t_last)
            x_final = last_step_fn(x_last, t_last, model_out)
            return jax.numpy.concatenate([xs, x_final[None]])

        return _sample


def vanilla_guidance(x: jax.Array, cfg_val: float):
    x_u, x_c = jnp.split(x, 2, axis=0)
    return x_u + cfg_val * (x_c - x_u)


def denoise_loop(
    *,
    model_fn,
    x,
    rng,
    num_steps,
    cfg_scale=None,
    guidance_low=0.0,
    guidance_high=1.0,
    mode="SDE",
    sampling_method="euler",
    reverse: bool = True,
):
    args = Config()
    args.num_steps = num_steps
    transport = create_transport(
        args.transport.path_type,
        args.transport.prediction,
        args.transport.loss_weight,
        args.transport.train_eps,
        args.transport.sample_eps,
    )

    if mode == "SDE":
        sampler = FixedSampler(transport)
        sample_fn = sampler.sample_sde(
            sampling_method=args.sde.sampling_method,
            diffusion_form=args.sde.diffusion_form,
            diffusion_norm=args.sde.diffusion_norm,
            last_step=args.sde.last_step,
            last_step_size=args.sde.last_step_size,
            num_steps=args.num_steps,
        )
    else:
        raise NotImplementedError("Only SDE mode is currently supported")

    def wrapped_model_fn(z, t):
        t_orig = t
        t = 1.0 - t if reverse else t

        if cfg_scale is not None and cfg_scale > 1.0:
            apply_cfg = jnp.all((guidance_low <= t) & (t <= guidance_high))
            
            def true_fn(z_true):
                bs = z_true.shape[0]
                z_half = z_true[bs // 2:]
                z_in = jnp.concatenate((z_half, z_half), axis=0)
                pred = model_fn(z_in, t)
                pred_cfg = vanilla_guidance(pred, cfg_scale)
                return jnp.concatenate((pred_cfg, pred_cfg), axis=0)

            def false_fn(z_false):
                return model_fn(z_false, t)

            pred = jax.lax.cond(apply_cfg, true_fn, false_fn, z)
        else:
            pred = model_fn(z, t)

        return -pred if reverse else pred

    samples = sample_fn(x, rng, wrapped_model_fn)
    return samples[-1]

# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Adaptive RF Izhikevich neuron."""

import numpy as np
import torch

from . import base
from .dynamics import resonator, adaptive_phase_th
from ..spike import complex
from ..utils import quantize

# These are tuned heuristically so that scale_grad=1 and tau_grad=1 serves as
# a good starting point

# SCALE_RHO_MULT = 10
# TAU_RHO_MULT = 0.2
# SCALE_RHO_MULT = 10
# TAU_RHO_MULT = 0.5
SCALE_RHO_MULT = 0.1
TAU_RHO_MULT = 100


class Neuron(base.Neuron):
    """This is the implementation of RF neuron.

    .. math::
        \\mathfrak{Re}(z[t]) &= (1-\\alpha)(\\cos\\phi\\ \\mathfrak{Re}(z[t-1])
            - \\sin\\phi\\ \\mathfrak{Im}(z[t-1]))
            + \\mathfrak{Re}(x[t]) + \\text{real bias}\\

        \\mathfrak{Im}(z[t]) &= (1-\\alpha)(\\sin\\phi\\ \\mathfrak{Re}(z[t-1])
            + \\cos\\phi\\ \\mathfrak{Im}(z[t-1]))
            + \\mathfrak{Im}(x[t]) + \\text{imag bias} \\

        \\vartheta[t] &= (1-\\alpha_{\\vartheta})\\,
            (\\vartheta[t-1] - \\vartheta_0) + \\vartheta_0 \\

        r[t] &= (1-\\alpha_r)\\,r[t-1] \\

        s[t] &= |z[t]| \\geq (\\vartheta[t] + r[t]) \\text{ and } \\arg(z[t])=0

    The internal state representations are scaled down compared to
    the actual hardware implementation. This allows for a natural range of
    synaptic weight values as well as the gradient parameters.

    The neuron parameters like threshold, decays are represented as real
    values. They internally get converted to fixed precision representation of
    the hardware. It also provides properties to access the neuron
    parameters in fixed precision states. The parameters are internally clamped
    to the valid range.

    Parameters
    ----------
    threshold : float
        neuron threshold.
    threshold_step : float
        the increase in threshold after spike.
    period : float or tuple
        period of the neuron. If ``shared_param`` is False, then it can be
        specified as a tuple (min_period, max_period).
    decay : float or tuple
        decay factor of the neuron. If ``shared_param`` is False, then it can
        be specified as a tuple (min_decay, max_decay).
    threshold_decay : float or tuple
        the fraction of threshold decay per time step. If ``shared_param`` is
        False, then it can be specified as a tuple : min_decay, max_decay).
    refractory_decay : float or tuple
        the fraction of refractory decay per time step. If ``shared_param`` is
        False, then it can be specified as a tuple : min_decay, max_decay).
    tau_grad : float, optional
        time constant of spike function derivative. Defaults to 1.
    scale_grad : float, optional
        scale of spike function derivative. Defaults to 1.
    scale : int, optional
        scale of the internal state. ``scale=1`` will result in values in the
        range expected from the of Loihi hardware. Defaults to 1  <<  6.
    norm : fx-ptr or lambda, optional
        normalization function on the dendrite output. None means no
        normalization. Defaults to None.
    dropout : fx-ptr or lambda, optional
        neuron dropout method. None means no normalization. Defaults to None.
    shared_param : bool, optional
        flag to enable/disable shared parameter neuron group. If it is
        False, individual parameters are assigned on a per-channel basis.
        Defaults to True.
    persistent_state : bool, optional
        flag to enable/disable persistent state between iterations.
        Defaults to False.
    requires_grad : bool, optional
        flag to enable/disable learning on neuron parameter. Defaults to False.
    graded_spike : bool, optional
        flag to enable/disable graded spike output. Defaults to False.
    log_init : bool, optional
        if True, initialized the natural frequency in log spaced range.
        Default is True.
    """
    def __init__(
        self, threshold, threshold_step,
        period, decay, threshold_decay, refractory_decay,
        tau_grad=1, scale_grad=1, scale=1 << 6,
        norm=None, dropout=None,
        shared_param=True, persistent_state=False, requires_grad=False,
        graded_spike=False,
        log_init=True,
    ):
        super(Neuron, self).__init__(
            threshold=threshold,
            tau_grad=tau_grad,
            scale_grad=scale_grad,
            p_scale=1 << 12,
            w_scale=scale,
            s_scale=scale * (1 << 6),
            norm=norm,
            dropout=dropout,
            persistent_state=persistent_state,
            shared_param=shared_param,
            requires_grad=requires_grad,
            complex=True
        )

        self.threshold_step = int(threshold_step * self.w_scale) / self.w_scale
        self.graded_spike = graded_spike
        self.log_init = log_init

        # sin_decay and cos_decay are restricted to be inside the unit circle
        # in first quadrant. This means that the oscillation step can only
        # between 0 and 90 degrees, i.e. period >= 4
        # It makes sense to have period >= 4 because we would not get proper
        # oscillation at all in a discrete in time system for period < 4.
        # It is possible to set period < 4, but it will get quantized.
        # So the result might not be as desired for period = 2 and 3.
        if self.shared_param is True:
            assert np.isscalar(decay) is True,\
                f'Expected decay to be a scalar when shared_param is True. '\
                f'Found {decay=}.'
            assert np.isscalar(period) is True,\
                f'Expected period to be a scalar when shared_param is True. '\
                f'Found {period=}.'
            assert period >= 4, \
                f'Neuron period less than 4 does not make sense. '\
                f'Found {period=}.'
            sin_decay = np.sin(2 * np.pi / period) * (1 - decay)
            cos_decay = np.cos(2 * np.pi / period) * (1 - decay)
            self.register_parameter(
                'sin_decay',
                torch.nn.Parameter(
                    torch.FloatTensor([self.p_scale * sin_decay]),
                    requires_grad=self.requires_grad,
                )
            )
            self.register_parameter(
                'cos_decay',
                torch.nn.Parameter(
                    torch.FloatTensor([self.p_scale * cos_decay]),
                    requires_grad=self.requires_grad,
                )
            )
            self.register_parameter(
                'threshold_decay',
                torch.nn.Parameter(
                    torch.FloatTensor([self.p_scale * threshold_decay]),
                    requires_grad=self.requires_grad,
                )
            )
            self.register_parameter(
                'refractory_decay',
                torch.nn.Parameter(
                    torch.FloatTensor([self.p_scale * refractory_decay]),
                    requires_grad=self.requires_grad,
                )
            )
        else:
            if np.isscalar(period) is True:  # 1% jitter for now
                assert period >= 4, \
                    f'Neuron period less than 4 does not make sense. '\
                    f'Found {period=}.'
                self.period_min = period * 0.99
                self.period_max = period * 1.01
            else:
                assert len(period) == 2,\
                    f'Expected period to be of length 2 i.e. [min, max]. '\
                    f'Found {period=}.'
                assert min(period) >= 4, \
                    f'Neuron period less than 4 does not make sense. '\
                    f'Found {period=}.'
                self.period_min = period[0]
                self.period_max = period[1]

            if np.isscalar(decay) is True:
                self.decay_min = decay * 0.99
                self.decay_max = decay * 1.01
            else:
                assert len(decay) == 2,\
                    f'Expected decay to be of length 2 i.e. [min, max]. '\
                    f'Found {decay=}.'
                self.decay_min = decay[0]
                self.decay_max = decay[1]

            if np.isscalar(threshold_decay) is True:
                self.threshold_decay_min = threshold_decay
                self.threshold_decay_max = threshold_decay
            else:
                assert len(threshold_decay) == 2,\
                    f'Expected threshold decay to be of length 2 i.e. '\
                    f'[min, max]. Found {threshold_decay=}.'
                self.threshold_decay_min = threshold_decay[0]
                self.threshold_decay_max = threshold_decay[1]

            if np.isscalar(refractory_decay) is True:
                self.refractory_decay_min = refractory_decay
                self.refractory_decay_max = refractory_decay
            else:
                assert len(refractory_decay) == 2,\
                    f'Expected refractory decay to be of length 2 i.e. '\
                    f'[min, max]. Found {refractory_decay=}.'
                self.refractory_decay_min = refractory_decay[0]
                self.refractory_decay_max = refractory_decay[1]

            self.register_parameter(
                'sin_decay',
                torch.nn.Parameter(
                    torch.FloatTensor([0]),
                    requires_grad=self.requires_grad,
                )
            )
            self.register_parameter(
                'cos_decay',
                torch.nn.Parameter(
                    torch.FloatTensor([0]),
                    requires_grad=self.requires_grad,
                )
            )
            self.register_parameter(
                'threshold_decay',
                torch.nn.Parameter(
                    torch.FloatTensor([
                        self.p_scale * self.threshold_decay_min
                    ]), requires_grad=self.requires_grad,
                )
            )
            self.register_parameter(
                'refractory_decay',
                torch.nn.Parameter(
                    torch.FloatTensor([
                        self.p_scale * self.refractory_decay_min
                    ]), requires_grad=self.requires_grad,
                )
            )

        self.register_buffer(
            'real_state',
            torch.zeros(1, dtype=torch.float),
            persistent=False
        )
        self.register_buffer(
            'imag_state',
            torch.zeros(1, dtype=torch.float),
            persistent=False
        )
        self.register_buffer(
            'threshold_state',
            torch.tensor([self.threshold], dtype=torch.float),
            persistent=False
        )
        self.register_buffer(
            'refractory_state',
            torch.zeros(1, dtype=torch.float),
            persistent=False
        )

        self.clamp()

    def clamp(self):
        """A function to clamp the sin decay and cosine decay parameters to be
        within valid range. The user will generally not need to call this
        function.
        """
        # the dynamics parameters must be positive within the allowable range
        # they must be inside the unit circle
        with torch.no_grad():
            gain = torch.sqrt(self.sin_decay**2 + self.cos_decay**2)
            clamped_gain = gain.clamp(0, self.p_scale - 1)
            self.sin_decay.data *= clamped_gain / gain
            self.cos_decay.data *= clamped_gain / gain
            # this should not be clamped to 0 because
            # 0 would mean no oscillation
            self.sin_decay.data.clamp_(1)
            self.cos_decay.data.clamp_(0)

    @property
    def decay(self):
        """The decay parameter of the neuron."""
        self.clamp()
        return 1 - np.sqrt(
            (quantize(self.sin_decay).cpu().data.numpy() / self.p_scale)**2
            + (quantize(self.cos_decay).cpu().data.numpy() / self.p_scale)**2
        )

    @property
    def lam(self):
        """The lambda parameter of the neuron."""
        return -np.log(1 - self.decay)

    @property
    def period(self):
        """The period of the neuron oscillation."""
        return 1 / self.frequency

    @property
    def frequency(self):
        """The frequency of neuron oscillation."""
        self.clamp()
        return np.arctan2(
            quantize(self.sin_decay).cpu().data.numpy(),
            quantize(self.cos_decay).cpu().data.numpy()
        ) / 2 / np.pi

    @property
    def device(self):
        """The device memory (cpu/cuda) where the object lives."""
        return self.sin_decay.device

    @property
    def cx_sin_decay(self):
        """The compartment sin decay parameter to be used for configuration."""
        self.clamp()
        val = quantize(self.sin_decay).cpu().data.numpy().astype(int)
        if len(val) == 1:
            return val[0]
        return val

    @property
    def cx_cos_decay(self):
        """The compartment cos decay parameter to be used for configuration."""
        self.clamp()
        val = quantize(self.cos_decay).cpu().data.numpy().astype(int)
        if len(val) == 1:
            return val[0]
        return val

    @property
    def cx_threshold_decay(self):
        """The compartment threshold decay parameter to be used for configuring
        Loihi hardware."""
        self.clamp()
        val = quantize(self.threshold_decay).cpu().data.numpy().astype(int)
        if len(val) == 1:
            return val[0]
        return val

    @property
    def cx_refractory_decay(self):
        """The compartment refractory decay parameter to be used for
        configuring Loihi hardware."""
        self.clamp()
        val = quantize(self.refractory_decay).cpu().data.numpy().astype(int)
        if len(val) == 1:
            return val[0]
        return val

    @property
    def scale(self):
        """Scale difference between slayer representation and hardware
        representation of the variable states."""
        return self.w_scale

    def dynamics(self, input):
        """Computes the dynamics (without spiking behavior) of the neuron
        instance to a complex input tuple. The input shape must match with the
        neuron shape. For the first time, the neuron shape is determined from
        the input automatically. It is essentially a resonator dynamics with
        adaptive threshold and refractory response.

        Parameters
        ----------
        input : tuple of torch tensors
            Complex input tuple of tensor, i.e. (real_input, imag_input).

        Returns
        -------
        torch tensor
            real response of the neuron.
        torch tensor
            imaginary response of the neuron.
        torch tensor
            adaptive threshold of the neuron.
        torch tensor
            refractory response of the neuorn.
        """
        real_input, imag_input = input
        if self.shape is None:
            self.shape = real_input.shape[1:-1]
            assert len(self.shape) > 0, \
                f"Expected input to have at least 3 dimensions: "\
                f"[Batch, Spatial dims ..., Time]. "\
                f"It's shape is {real_input.shape}."
            self.num_neurons = np.prod(self.shape)
            if self.shared_param is False:
                if self.log_init is False:
                    frequency = (
                        (1 / self.period_max)
                        + (1 / self.period_min - 1 / self.period_max)
                        * torch.rand(
                            self.shape[0], dtype=torch.float
                        ).to(self.device)
                    )
                else:
                    frequency = torch.logspace(
                        -np.log10(self.period_max),
                        -np.log10(self.period_min),
                        steps=self.shape[0]
                    ).to(self.device)

                decay = self.decay_min \
                    + (self.decay_max - self.decay_min) * torch.rand(
                        self.shape[0], dtype=torch.float
                    ).to(self.device)
                sin_decay = torch.sin(2 * np.pi * frequency) * (1 - decay)
                cos_decay = torch.cos(2 * np.pi * frequency) * (1 - decay)
                threshold_decay = self.threshold_decay_min \
                    + (self.threshold_decay_max - self.threshold_decay_min)\
                    * torch.rand(
                        self.shape[0], dtype=torch.float
                    ).to(self.device)
                refractory_decay = self.refractory_decay_min \
                    + (self.refractory_decay_max - self.refractory_decay_min)\
                    * torch.rand(
                        self.shape[0], dtype=torch.float
                    ).to(self.device)
                self.sin_decay.data = self.p_scale * sin_decay
                self.cos_decay.data = self.p_scale * cos_decay
                self.threshold_decay.data = self.p_scale * threshold_decay
                self.refractory_decay.data = self.p_scale * refractory_decay

                del self.period_min
                del self.period_max
                del self.decay_min
                del self.decay_max
                del self.threshold_decay_min
                del self.threshold_decay_max
                del self.refractory_decay_min
                del self.refractory_decay_max
        else:
            assert real_input.shape[1:-1] == self.shape, \
                f'Real input tensor shape ({real_input.shape}) '\
                f'does not match with Neuron shape ({self.shape}).'
        assert real_input.shape == imag_input.shape, \
            f'Real input tensor shape ({imag_input.shape}) does not match '\
            f'with imaginary input shape ({imag_input.shape}).'

        dtype = self.real_state.dtype
        device = self.real_state.device
        if self.real_state.shape[0] != real_input.shape[0]:
            # persistent state cannot proceed due to change in batch dimension.
            # this likely indicates change from training to testing set
            self.real_state = torch.zeros(
                real_input.shape[:-1]
            ).to(dtype).to(device)
            self.imag_state = torch.zeros(
                real_input.shape[:-1]
            ).to(dtype).to(device)
            self.threshold_state = self.threshold * torch.ones(
                real_input.shape[:-1]
            ).to(dtype).to(device)
            self.refractory_state = torch.zeros(
                real_input.shape[:-1]
            ).to(dtype).to(device)

        if self.real_state.shape[1:] != self.shape:
            # this means real_state and imag_state are not initialized properly
            self.real_state = self.real_state * torch.ones(
                real_input.shape[:-1]
            ).to(dtype).to(device)
            self.imag_state = self.imag_state * torch.ones(
                real_input.shape[:-1]
            ).to(dtype).to(device)
            self.threshold_state = self.threshold_state * torch.ones(
                real_input.shape[:-1]
            ).to(dtype).to(device)
            self.refractory_state = self.refractory_state * torch.ones(
                real_input.shape[:-1]
            ).to(dtype).to(device)

        # clamp the values only when learning is enabled
        # This means we don't need to clamp the values after gradient update.
        # It is done in runtime now. Might be slow, but overhead is negligible.
        if self.requires_grad is True:
            self.clamp()

        if self.real_norm is not None:
            real_input = self.real_norm(real_input)
        if self.imag_norm is not None:
            imag_input = self.imag_norm(imag_input)

        real, imag = resonator.dynamics(
            real_input, imag_input,
            quantize(self.sin_decay),
            quantize(self.cos_decay),
            self.real_state, self.imag_state,
            self.s_scale,
            debug=self.debug
        )

        threshold, refractory = adaptive_phase_th.dynamics(
            real, imag,
            self.imag_state,
            self.refractory_state,
            quantize(self.refractory_decay),
            self.threshold_state,
            quantize(self.threshold_decay),
            self.threshold_step,
            self.threshold,
            self.s_scale,
            debug=self.debug
        )

        if self.persistent_state is True:
            with torch.no_grad():
                self.real_state = real[..., -1].clone()
                # self.imag_state = imag[..., -1].clone()
                # this should be done post spike
                self.threshold_state = threshold[..., -1].clone()
                self.refractory_state = refractory[..., -1].clone()

        return real, imag, threshold, refractory

    def spike(self, real, imag, threshold, refractory):
        """Extracts spike points from the real and imaginary states.

        Parameters
        ----------
        real : torch tensor
            real dynamics of the neuron.
        imag : torch tensor
            imaginary dynamics of the neuron.
        threshold :torch tensor
            threshold dynamics of the neuron.
        refractory :torch tensor
            refractory dynamics of the neuron.

        Returns
        -------
        torch tensor
            spike output

        """

        # print(
        #     real[..., -1].item() * self.s_scale,
        #     imag[..., -1].item() * self.s_scale,
        #     self.imag_state.item() * self.s_scale
        # )
        spike = complex.Spike.apply(
            real, imag, threshold + refractory,
            self.tau_rho * TAU_RHO_MULT,
            self.scale_rho * SCALE_RHO_MULT,
            self.graded_spike,
            self.imag_state,
            # self.s_scale,
            1,
        )

        if self.persistent_state is True:
            with torch.no_grad():
                # self.real_state *= (1 - spike[..., -1]).detach().clone()
                self.imag_state = imag[..., -1].clone()
                self.refractory_state = adaptive_phase_th.persistent_ref_state(
                    self.refractory_state, spike[..., -1],
                    self.threshold_state
                ).detach().clone()
                self.threshold_state = adaptive_phase_th.persistent_th_state(
                    self.threshold_state, spike[..., -1],
                    self.threshold_step
                ).detach().clone()

        if self.drop is not None:
            spike = self.drop(spike)

        return spike

    def forward(self, input):
        """omputes the full response of the neuron instance to a complex
        input tuple. The input shape must match with the neuron shape. For the
        first time, the neuron shape is determined from the input
        automatically.

        Parameters
        ----------
        input :
            Complex input tuple of tensor, i.e. (real_input, imag_input).

        Returns
        -------
        torch tensor
            spike response of the neuron.

        """
        real, imag, threshold, refractory = self.dynamics(input)
        return self.spike(real, imag, threshold + refractory)
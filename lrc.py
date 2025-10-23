import numpy as np
import torch
import torch.nn as nn

class LRC(nn.Module):
    def __init__(
            self,
            input_size: int,
            units,
            return_sequences: bool = True,
    ):
        super(LRC, self).__init__()
        self.input_size = input_size
        self.units = units
        self.rnn_cell = LRCCell(in_features=self.input_size, units=self.units)
        self.batch_first = True
        self.return_sequences = return_sequences
        self.fc = nn.Linear(32, 1)

    def forward(self, input, hx=None, timespans=None):
        """

        :param input: Input tensor of shape (L,C) in batchless mode, or (B,L,C) if batch_first was set to True and (L,B,C) if batch_first is False
        :param hx: Initial hidden state of the RNN of shape (B,H) if mixed_memory is False and a tuple ((B,H),(B,H)) if mixed_memory is True. If None, the hidden states are initialized with all zeros.
        :param timespans:
        :return: A pair (output, hx), where output and hx the final hidden state of the RNN
        """
        device = input.device
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if timespans is not None:
                timespans = timespans.unsqueeze(batch_dim)

        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)

        if hx is None:
            h_state = torch.zeros((batch_size, self.state_size), device=device)
        else:
            h_state = hx
            if is_batched:
                if h_state.dim() != 2:
                    msg = (
                        "For batched 2-D input, hx and cx should "
                        f"also be 2-D but got ({h_state.dim()}-D) tensor"
                    )
                    raise RuntimeError(msg)
            else:
                # batchless  mode
                if h_state.dim() != 1:
                    msg = (
                        "For unbatched 1-D input, hx and cx should "
                        f"also be 1-D but got ({h_state.dim()}-D) tensor"
                    )
                    raise RuntimeError(msg)
                h_state = h_state.unsqueeze(0)

        output_sequence = []
        for t in range(seq_len):
            if self.batch_first:
                inputs = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            else:
                inputs = input[t]
                ts = 1.0 if timespans is None else timespans[t].squeeze()

            h_out, h_state = self.rnn_cell.forward(inputs, h_state, ts)
            h_out = self.fc(h_out)
            if self.return_sequences:
                output_sequence.append(h_out)

        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = h_out
        hx = h_state

        if not is_batched:
            # batchless  mode
            readout = readout.squeeze(batch_dim)
            hx = h_state[0]

        return readout, hx

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return 1

class LRCCell(nn.Module):
    def __init__(
            self,
            in_features,
            units,
            ode_unfolds=1,
            epsilon=1e-8,
            elastance_var="simple_mult"
    ):
        super(LRCCell, self).__init__()
        self.in_features = in_features
        self.units = units
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self.elastance_var = elastance_var
        self.softplus = nn.Softplus()
        # self.softplus = nn.Identity()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        self._allocate_parameters()

    @property
    def state_size(self):
        return self.units

    @property
    def sensory_size(self):
        return self.in_features

    def add_weight(self, name, init_value):
        param = torch.nn.Parameter(init_value)
        self.register_parameter(name, param)
        return param

    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    def _allocate_parameters(self):
        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak", init_value=self._get_init_value((self.state_size,), "gleak")
        )
        self._params["vleak"] = self.add_weight(
            name="vleak", init_value=self._get_init_value((self.state_size,), "vleak")
        )
        self._params["cm"] = self.add_weight(
            name="cm", init_value=self._get_init_value((self.state_size,), "cm")
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            init_value=self._get_init_value(
                (self.state_size, self.state_size), "sigma"
            ),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            init_value=self._get_init_value((self.state_size, self.state_size), "mu"),
        )
        self._params["w"] = self.add_weight(
            name="w",
            init_value=self._get_init_value((self.state_size, self.state_size), "w"),
        )
        self._params["h"] = self.add_weight(
            name="h",
            init_value=self._get_init_value((self.state_size, self.state_size), "w"),
        )
        torch.nn.init.orthogonal_(self._params["h"])

        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_sigma"
            ),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_mu"
            ),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_w"
            ),
        )

        self._params["sensory_h"] = self.add_weight(
            name="sensory_h",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_w"
            ),
        )
        torch.nn.init.orthogonal_(self._params["sensory_h"])

        self._params["input_w"] = self.add_weight(
            name="input_w",
            init_value=torch.ones((self.sensory_size,)),
        )
        self._params["input_b"] = self.add_weight(
            name="input_b",
            init_value=torch.zeros((self.sensory_size,)),
        )

        self._params["output_w"] = self.add_weight(
            name="output_w",
            init_value=torch.ones((self.state_size,)),
        )
        self._params["output_b"] = self.add_weight(
            name="output_b",
            init_value=torch.zeros((self.state_size,)),
        )

        self.elastance_dense = nn.Linear(self.state_size + self.sensory_size, self.state_size)
        if self.elastance_var == "normal_distr":
            self._params["distr_shift"] = self.add_weight(
                name="distr_shift",
                init_value=torch.ones((self.state_size,)),
            )

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state.to(inputs.device)

        # We can pre-compute the effects of the sensory neurons here
        sensory_syn = self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )

        sensory_h_activation = self._params["sensory_h"] * sensory_syn

        sensory_w_activation = self.softplus(self._params["sensory_w"]) * sensory_syn

        # Reduce over dimension 1 (=source sensory neurons)
        sensory_h_activation_reduced = torch.sum(sensory_h_activation, dim=1)
        sensory_w_activation_reduced = torch.sum(sensory_w_activation, dim=1)

        dt = elapsed_time / self._ode_unfolds

        # Unfold the multiply ODE multiple times into one RNN step
        for t in range(self._ode_unfolds):
            if self.elastance_var == "simple_mult":
                # Concatenate inputs and state
                x = torch.cat([v_pre, inputs], dim=1)
                t_a = self.elastance_dense(x)
                t_interp = self.sigm(t_a) * dt
            elif self.elastance_var == "normal_distr":
                x = torch.cat([v_pre, inputs], dim=1)
                t_a = self.elastance_dense(x)
                t_interp = (self.sigm(t_a + self._params["distr_shift"]) - self.sigm(
                    t_a - self._params["distr_shift"])) * dt
            else:
                t_interp = dt

            syn = self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            h_activation = self._params["h"] * syn

            w_activation = self.softplus(self._params["w"]) * syn

            g = self.softplus(self._params["gleak"]) + torch.sum(h_activation, dim=1) + sensory_h_activation_reduced

            f = self.softplus(self._params["gleak"]) + torch.sum(w_activation, dim=1) + sensory_w_activation_reduced

            v_prime = - v_pre * self.sigm(f) + self._params["vleak"] * self.tanh(g)

            v_pre = v_pre + t_interp * v_prime

        return v_pre

    def _map_inputs(self, inputs):
        inputs = inputs * self._params["input_w"]
        inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        output = state
        output = output[:, 0: 1]  # slice
        output = output * self._params["output_w"]
        output = output + self._params["output_b"]
        return output

    def forward(self, input, hx, elapsed_time=1.0):
        # Regularly sampled mode (elapsed time = 1 second)
        inputs = self._map_inputs(input)

        next_state = self._ode_solver(inputs, hx, elapsed_time)

        # outputs = self._map_outputs(next_state)

        return next_state, next_state


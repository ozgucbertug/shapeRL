"""Network components for Shape RL."""

from __future__ import annotations

import math
import tensorflow as tf
from keras import layers
from tf_agents.networks import network
import tensorflow_probability as tfp

def bilinear_sample_nhwc(feat: tf.Tensor, xy01: tf.Tensor) -> tf.Tensor:
    """Samples per-location features from an NHWC map using normalized coords.

    Args:
        feat: Feature map tensor of shape [B, H, W, C].
        xy01: Continuous coordinates in [0, 1] with shape [B, 2] ordered as (x, y).

    Returns:
        Tensor of shape [B, C] with bilinearly interpolated features.
    """
    feat = tf.convert_to_tensor(feat)
    xy01 = tf.convert_to_tensor(xy01)

    batch_size = tf.shape(feat)[0]
    height = tf.shape(feat)[1]
    width = tf.shape(feat)[2]

    x = tf.clip_by_value(xy01[..., 0], 0.0, 1.0) * (tf.cast(width, tf.float32) - 1.0)
    y = tf.clip_by_value(xy01[..., 1], 0.0, 1.0) * (tf.cast(height, tf.float32) - 1.0)

    x0 = tf.cast(tf.floor(x), tf.int32)
    y0 = tf.cast(tf.floor(y), tf.int32)
    x1 = tf.minimum(x0 + 1, width - 1)
    y1 = tf.minimum(y0 + 1, height - 1)

    def _gather(h: tf.Tensor, w: tf.Tensor) -> tf.Tensor:
        idx = tf.stack([tf.range(batch_size), h, w], axis=-1)
        return tf.gather_nd(feat, idx)

    Ia = _gather(y0, x0)
    Ib = _gather(y0, x1)
    Ic = _gather(y1, x0)
    Id = _gather(y1, x1)

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    x0f = tf.cast(x0, tf.float32)
    y0f = tf.cast(y0, tf.float32)
    x1f = tf.cast(x1, tf.float32)
    y1f = tf.cast(y1, tf.float32)

    wa = (x1f - x) * (y1f - y)
    wb = (x - x0f) * (y1f - y)
    wc = (x1f - x) * (y - y0f)
    wd = (x - x0f) * (y - y0f)

    wa = tf.expand_dims(wa, axis=-1)
    wb = tf.expand_dims(wb, axis=-1)
    wc = tf.expand_dims(wc, axis=-1)
    wd = tf.expand_dims(wd, axis=-1)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


class SpatialSoftmax(layers.Layer):
    """Computes spatial expectation for heatmap logits with bounded temperature.

    Lower temperatures sharpen the softmax distribution, while higher temperatures
    produce smoother, more diffuse expectations. Temperature is constrained to a
    configurable range via a sigmoid, to avoid degenerate over-sharp or over-diffuse
    regimes while remaining fully differentiable.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        learnable: bool = True,
        min_temperature: float = 0.2,
        max_temperature: float = 3.0,
        reg_coeff: float = 0.0,
        target_temperature: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if min_temperature <= 0.0:
            raise ValueError("min_temperature must be positive.")
        if not (min_temperature < max_temperature):
            raise ValueError("min_temperature must be < max_temperature.")
        # Clamp the requested initial temperature to bounds
        t0 = float(temperature)
        tmin = float(min_temperature)
        tmax = float(max_temperature)
        t0 = min(max(t0, tmin), tmax)
        # Inverse of: T = Tmin + (Tmax-Tmin) * sigmoid(alpha)
        ratio = (t0 - tmin) / (tmax - tmin + 1e-12)
        ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
        initial_alpha = math.log(ratio / (1.0 - ratio))

        # Unconstrained parameter that we squash into [Tmin, Tmax]
        self._temp_alpha = self.add_weight(
            name='temp_alpha',
            shape=(),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(initial_alpha),
            trainable=bool(learnable),
        )
        self._tmin = tmin
        self._tmax = tmax
        self._initial_temperature = t0
        self._learnable = bool(learnable)
        self._reg_coeff = float(reg_coeff)
        # If None, regularize toward mid-range temperature
        self._target_temperature = float(target_temperature) if target_temperature is not None else None

    def _current_temperature(self) -> tf.Tensor:
        # Differentiable mapping into [Tmin, Tmax]
        p = tf.nn.sigmoid(self._temp_alpha)
        return self._tmin + (self._tmax - self._tmin) * p

    def current_temperature(self) -> tf.Tensor:
        """Public accessor for the current (bounded) temperature tensor."""
        return tf.identity(self._current_temperature(), name="spatialsoftmax_temperature")

    def call(self, logits: tf.Tensor) -> tf.Tensor:
        logits = tf.cast(logits, tf.float32)
        shape = tf.shape(logits)
        batch = shape[0]
        height = shape[1]
        width = shape[2]

        flat = tf.reshape(logits, [batch, height * width])
        temperature = self._current_temperature()
        scaled = flat / temperature
        # Stabilize softmax by subtracting per-batch max
        max_per_batch = tf.reduce_max(scaled, axis=-1, keepdims=True)
        normalized = scaled - max_per_batch
        weights = tf.nn.softmax(normalized)

        # Optional gentle regularization to discourage extreme temperatures
        if self._reg_coeff > 0.0:
            target_T = (
                tf.constant(0.5 * (self._tmin + self._tmax), dtype=tf.float32)
                if self._target_temperature is None
                else tf.constant(float(self._target_temperature), dtype=tf.float32)
            )
            # Normalize by range so the magnitude is consistent across settings
            rng = tf.constant(self._tmax - self._tmin, dtype=tf.float32)
            reg = ((temperature - target_T) / (rng + 1e-12)) ** 2
            # Add as a per-batch mean to layer losses
            self.add_loss(self._reg_coeff * tf.reduce_mean(reg))

        xs = tf.linspace(0.0, 1.0, width)
        ys = tf.linspace(0.0, 1.0, height)
        grid_x, grid_y = tf.meshgrid(xs, ys)
        coords = tf.stack([tf.reshape(grid_x, [-1]), tf.reshape(grid_y, [-1])], axis=-1)
        return tf.matmul(weights, coords)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'temperature': self._initial_temperature,
                'learnable': self._learnable,
                'min_temperature': self._tmin,
                'max_temperature': self._tmax,
                'reg_coeff': self._reg_coeff,
                'target_temperature': (
                    float(self._target_temperature) if self._target_temperature is not None else None
                ),
            }
        )
        return config


# ==========================================================================
#  Spatial-softmax encoder family
# ==========================================================================


class _ConvBlock(layers.Layer):
    """Conv-GN-ReLU block with configurable stride."""

    def __init__(self, filters: int, stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters, 3, strides=stride, padding='same')
        # Use GN with groups chosen by channel count
        groups = 16 if filters >= 64 else (8 if filters >= 32 else 4)
        self.norm = layers.GroupNormalization(groups=groups, axis=-1, epsilon=1e-5)
        self.act = layers.Activation('relu')

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = self.conv(inputs)
        x = self.norm(x, training=training)
        return self.act(x)


class SpatialSoftmaxEncoder(layers.Layer):
    def current_temperature(self) -> tf.Tensor | None:
        if self._use_heatmap and self.spatial_softmax is not None:
            return self.spatial_softmax.current_temperature()
        return None
    """CNN backbone that predicts a spatial expectation alongside a latent vector."""

    def __init__(
        self,
        filters: tuple[int, ...] = (32, 64, 128),
        latent_dim: int = 128,
        return_feature_maps: bool = False,
        use_heatmap: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.blocks = []
        for idx, f in enumerate(filters):
            stride = 1 if idx == 0 else 2
            self.blocks.append(_ConvBlock(f, stride=stride))
        self._use_heatmap = use_heatmap
        if self._use_heatmap:
            self.heatmap_head = layers.Conv2D(1, 1, padding='same')
            self.spatial_softmax = SpatialSoftmax()
        else:
            self.heatmap_head = None
            self.spatial_softmax = None
        self.global_pool = layers.GlobalAveragePooling2D()
        self.latent = layers.Dense(latent_dim, activation='elu')
        self._return_feature_maps = return_feature_maps

    def call(
        self, inputs: tf.Tensor, training: bool | None = None
    ) -> tuple[tf.Tensor, tf.Tensor] | tuple[tf.Tensor, tf.Tensor, tuple[tf.Tensor, ...]]:
        x = tf.cast(inputs, tf.float32)
        feature_maps: list[tf.Tensor] = []
        for block in self.blocks:
            x = block(x, training=training)
            feature_maps.append(x)
        xy: tf.Tensor | None = None
        if self._use_heatmap:
            heatmap = self.heatmap_head(x)
            xy = self.spatial_softmax(heatmap)
        latent = self.latent(self.global_pool(x))
        if self._return_feature_maps:
            if len(feature_maps) >= 2:
                maps = (feature_maps[-2], feature_maps[-1])
            else:
                maps = (feature_maps[-1],)
            return latent, xy, maps
        return latent, xy


class SpatialSoftmaxActorNetwork(network.Network):
    def current_temperature(self) -> tf.Tensor | None:
        return self.encoder.current_temperature()
    """Actor that anchors xy means using a spatial softmax heatmap."""

    def __init__(self, observation_spec, action_spec, name: str = 'SpatialSoftmaxActorNetwork'):
        super().__init__(input_tensor_spec=observation_spec, state_spec=(), name=name)
        self._action_spec = action_spec
        self.encoder = SpatialSoftmaxEncoder(latent_dim=128, return_feature_maps=True)
        self.fc1 = layers.Dense(128, activation='elu')
        self.fc2 = layers.Dense(64, activation='elu')
        self.depth_fc = layers.Dense(64, activation='elu')
        self.dz_head = layers.Dense(1)
        action_dims = action_spec.shape[0]
        self.mean_residual = layers.Dense(action_dims)
        self.logstd_head = layers.Dense(action_dims)

    def call(self, observations, step_type=None, network_state=(), training: bool = False):
        obs = tf.cast(observations, tf.float32)
        latent, xy, feature_maps = self.encoder(obs, training=training)
        local_feats = [bilinear_sample_nhwc(feature_map, xy) for feature_map in feature_maps]
        local_feat = local_feats[0] if len(local_feats) == 1 else tf.concat(local_feats, axis=-1)
        diff_xy = bilinear_sample_nhwc(obs[..., 0:1], xy)
        grad_xy = bilinear_sample_nhwc(obs[..., 3:4], xy)
        lap_xy = bilinear_sample_nhwc(obs[..., 4:5], xy)
        local_feat = tf.concat([local_feat, diff_xy, grad_xy, lap_xy], axis=-1)
        local_feat = tf.stop_gradient(local_feat)
        fused = tf.concat([latent, xy], axis=-1)
        fused = self.fc1(fused)
        fused = self.fc2(fused)

        xy_clipped = tf.clip_by_value(xy, 1e-3, 1.0 - 1e-3)
        xy_pre = tf.clip_by_value(2.0 * xy_clipped - 1.0, -0.999, 0.999)
        xy_base = tf.math.atanh(xy_pre)
        depth_in = tf.concat([fused, local_feat, diff_xy, grad_xy, lap_xy], axis=-1)
        depth_h = self.depth_fc(depth_in)
        dz_base = self.dz_head(depth_h)
        base_mean = tf.concat([xy_base, dz_base], axis=-1)
        mean = base_mean + self.mean_residual(fused)

        logstd = self.logstd_head(fused)
        logstd = tf.clip_by_value(logstd, -3.0, 1.0)
        scale = tf.nn.softplus(logstd) + 1e-3

        base = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=scale)
        bijector = tfp.bijectors.Chain(
            [tfp.bijectors.Shift(0.5), tfp.bijectors.Scale(0.5), tfp.bijectors.Tanh()]
        )
        dist = tfp.distributions.TransformedDistribution(base, bijector)
        return dist, network_state


class SpatialSoftmaxCriticNetwork(network.Network):
    """Critic that evaluates supplied action xy coordinates via bilinear sampling."""

    def __init__(self, observation_spec, action_spec, name: str = 'SpatialSoftmaxCriticNetwork'):
        super().__init__(input_tensor_spec=(observation_spec, action_spec), state_spec=(), name=name)
        self.encoder = SpatialSoftmaxEncoder(latent_dim=128, return_feature_maps=True, use_heatmap=False)
        self.action_fc = layers.Dense(64, activation='relu')
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.q_head = layers.Dense(1)

    def call(self, inputs, step_type=None, network_state=(), training: bool = False):
        observations, actions = inputs
        obs = tf.cast(observations, tf.float32)
        acts = tf.cast(actions, tf.float32)

        latent, _, feature_maps = self.encoder(obs, training=training)
        xy = acts[..., :2]
        action_latent = self.action_fc(acts)

        local_feats = [bilinear_sample_nhwc(feature_map, xy) for feature_map in feature_maps]
        local_feat = local_feats[0] if len(local_feats) == 1 else tf.concat(local_feats, axis=-1)

        diff = bilinear_sample_nhwc(obs[..., 0:1], xy)
        grad = bilinear_sample_nhwc(obs[..., 3:4], xy)
        lap = bilinear_sample_nhwc(obs[..., 4:5], xy)
        sampled_scalars = tf.concat([diff, grad, lap], axis=-1)

        fused = tf.concat([latent, action_latent, local_feat, sampled_scalars], axis=-1)
        fused = self.fc1(fused)
        fused = self.fc2(fused)
        q = self.q_head(fused)
        return tf.squeeze(q, axis=-1), network_state


__all__ = [
    'SpatialSoftmax',
    'SpatialSoftmaxEncoder',
    'SpatialSoftmaxActorNetwork',
    'SpatialSoftmaxCriticNetwork',
]

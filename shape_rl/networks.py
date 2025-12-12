"""Network components for Shape RL."""

from __future__ import annotations

import math
import tensorflow as tf
from keras import layers
from tf_agents.networks import network
import tensorflow_probability as tfp


# ==========================================================================
# Helpers
# ==========================================================================


def make_tanh_squash_bijector() -> tfp.bijectors.Bijector:
    """Shared bijector used to squash Gaussian actions into [0, 1]^3."""
    return tfp.bijectors.Chain(
        [tfp.bijectors.Shift(0.5), tfp.bijectors.Scale(0.5), tfp.bijectors.Tanh()]
    )


class SquashedMultivariateNormalDiag(tfp.distributions.TransformedDistribution):
    """MultivariateNormalDiag with tanh+affine squashing and an analytic mode."""

    def __init__(self, loc: tf.Tensor, scale_diag: tf.Tensor,
                 bijector: tfp.bijectors.Bijector | None = None,
                 name: str = 'SquashedMultivariateNormalDiag'):
        base = tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
        parameters = dict(locals())
        super().__init__(distribution=base,
                         bijector=bijector or make_tanh_squash_bijector(),
                         name=name)
        # Store subclass parameters so TFP shape utilities see the correct signature.
        self._parameters = parameters

    def mode(self) -> tf.Tensor:
        # Chain bijector is monotonic, so transforming the base mode is valid.
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            loc=tfp.util.ParameterProperties(event_ndims=1),
            scale_diag=tfp.util.ParameterProperties(event_ndims=1,
                                                    default_constraining_bijector_fn=tfp.bijectors.Softplus),
            bijector=tfp.util.ParameterProperties(),
            name=tfp.util.ParameterProperties(default_constraining_bijector_fn=None),
        )


class FiLMConditioner(layers.Layer):
    """Produces per-block FiLM scales/shifts from a conditioning vector."""

    def __init__(self, num_blocks: int, hidden_dim: int = 64,
                 gamma_scale: float = 0.25, beta_scale: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.num_blocks = int(num_blocks)
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.fc2 = layers.Dense(2 * self.num_blocks)

    def call(self, cond: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = self.fc1(cond)
        params = self.fc2(x)
        params = tf.reshape(params, [-1, self.num_blocks, 2])
        gamma_raw = params[..., 0]
        beta_raw = params[..., 1]
        gamma = 1.0 + self.gamma_scale * tf.tanh(gamma_raw)
        beta = self.beta_scale * tf.tanh(beta_raw)
        return gamma, beta


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


def sample_feature_maps(feature_maps: tuple[tf.Tensor, ...] | list[tf.Tensor], xy: tf.Tensor) -> tf.Tensor:
    """Bilinear-sample the provided feature maps at xy and concatenate."""
    locals_list = [bilinear_sample_nhwc(fmap, xy) for fmap in feature_maps]
    if len(locals_list) == 1:
        return locals_list[0]
    return tf.concat(locals_list, axis=-1)


def sample_obs_scalars(obs: tf.Tensor, xy: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Sample diff/grad/lap at xy and return with global scale token."""
    diff_xy = bilinear_sample_nhwc(obs[..., 0:1], xy)
    grad_xy = bilinear_sample_nhwc(obs[..., 3:4], xy)
    lap_xy = bilinear_sample_nhwc(obs[..., 4:5], xy)
    scale_token = tf.reduce_mean(obs[..., 5:], axis=[1, 2])  # [B,1]
    return diff_xy, grad_xy, lap_xy, scale_token


# ==========================================================================
# Blocks
# ==========================================================================


class SpatialSoftmax(layers.Layer):
    """Computes spatial expectation for heatmap logits with bounded temperature."""

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
        t0 = float(temperature)
        tmin = float(min_temperature)
        tmax = float(max_temperature)
        t0 = min(max(t0, tmin), tmax)
        ratio = (t0 - tmin) / (tmax - tmin + 1e-12)
        ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
        initial_alpha = math.log(ratio / (1.0 - ratio))

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
        self._target_temperature = float(target_temperature) if target_temperature is not None else None

    def _current_temperature(self) -> tf.Tensor:
        p = tf.nn.sigmoid(self._temp_alpha)
        return self._tmin + (self._tmax - self._tmin) * p

    def current_temperature(self) -> tf.Tensor:
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
        max_per_batch = tf.reduce_max(scaled, axis=-1, keepdims=True)
        normalized = scaled - max_per_batch
        weights = tf.nn.softmax(normalized)

        if self._reg_coeff > 0.0:
            target_T = (
                tf.constant(0.5 * (self._tmin + self._tmax), dtype=tf.float32)
                if self._target_temperature is None
                else tf.constant(float(self._target_temperature), dtype=tf.float32)
            )
            rng = tf.constant(self._tmax - self._tmin, dtype=tf.float32)
            reg = ((temperature - target_T) / (rng + 1e-12)) ** 2
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


class _ConvBlock(layers.Layer):
    """Conv-GN-ReLU block with configurable stride."""

    def __init__(self, filters: int, stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters, 3, strides=stride, padding='same')
        groups = 16 if filters >= 64 else (8 if filters >= 32 else 4)
        self.norm = layers.GroupNormalization(groups=groups, axis=-1, epsilon=1e-5)
        self.act = layers.Activation('relu')

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = self.conv(inputs)
        x = self.norm(x, training=training)
        return self.act(x)


class FiLMConvBlock(layers.Layer):
    """Conv-GN block with FiLM modulation and ReLU."""

    def __init__(self, filters: int, stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters, 3, strides=stride, padding='same')
        groups = 16 if filters >= 64 else (8 if filters >= 32 else 4)
        self.norm = layers.GroupNormalization(groups=groups, axis=-1, epsilon=1e-5)
        self.act = layers.Activation('relu')

    def call(self, inputs: tf.Tensor, gamma: tf.Tensor | None = None,
             beta: tf.Tensor | None = None, training: bool | None = None) -> tf.Tensor:
        x = self.conv(inputs)
        x = self.norm(x, training=training)
        if gamma is not None:
            # Broadcast (B,) or (B,1) to spatial maps.
            gamma = tf.reshape(gamma, [-1, 1, 1, 1])
            x = gamma * x
        if beta is not None:
            beta = tf.reshape(beta, [-1, 1, 1, 1])
            x = x + beta
        return self.act(x)

# ==========================================================================
# Encoders
# ==========================================================================

class SpatialSoftmaxEncoder(layers.Layer):
    """CNN backbone that predicts a spatial expectation alongside a latent vector."""

    def current_temperature(self) -> tf.Tensor | None:
        if self._use_heatmap and self.spatial_softmax is not None:
            return self.spatial_softmax.current_temperature()
        return None

    def __init__(
        self,
        filters: tuple[int, ...] = (32, 64, 128),
        latent_dim: int = 128,
        return_feature_maps: bool = False,
        use_heatmap: bool = True,
        num_obs_channels: int | None = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num_obs_channels = int(num_obs_channels) if num_obs_channels is not None else None
        self.blocks = []
        for idx, f in enumerate(filters):
            # Downsample only once (2x) to keep a ~64x64 heatmap for 128x128 inputs.
            stride = 2 if idx == 1 else 1
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
        if self._num_obs_channels is not None:
            x = x[..., : self._num_obs_channels]
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


class SpatialFiLMEncoder(layers.Layer):
    """CNN backbone with FiLM modulation using global scale/diff statistics."""

    def current_temperature(self) -> tf.Tensor | None:
        if self._use_heatmap and self.spatial_softmax is not None:
            return self.spatial_softmax.current_temperature()
        return None

    def __init__(
        self,
        filters: tuple[int, ...] = (32, 64, 128),
        latent_dim: int = 128,
        return_feature_maps: bool = False,
        use_heatmap: bool = True,
        include_diff_stat: bool = True,
        film_hidden: int = 64,
        num_obs_channels: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num_obs_channels = int(num_obs_channels) if num_obs_channels is not None else None
        self.blocks = []
        for idx, f in enumerate(filters):
            # Downsample only once (2x) to keep a ~64x64 heatmap for 128x128 inputs.
            stride = 2 if idx == 1 else 1
            self.blocks.append(FiLMConvBlock(f, stride=stride))
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
        self._include_diff_stat = bool(include_diff_stat)
        self.conditioner = FiLMConditioner(num_blocks=len(filters), hidden_dim=film_hidden)

    def _conditioning_vector(self, inputs: tf.Tensor) -> tf.Tensor:
        # Scale token is constant plane; reduce to scalar per batch.
        scale_token = tf.reduce_mean(inputs[..., 5:], axis=[1, 2])  # [B,1]
        parts = [scale_token]
        if self._include_diff_stat:
            diff_abs = tf.reduce_mean(tf.abs(inputs[..., 0:1]), axis=[1, 2])
            parts.append(diff_abs)
        return tf.concat(parts, axis=-1)

    def call(
        self, inputs: tf.Tensor, training: bool | None = None
    ) -> tuple[tf.Tensor, tf.Tensor] | tuple[tf.Tensor, tf.Tensor, tuple[tf.Tensor, ...]]:
        x = tf.cast(inputs, tf.float32)
        if self._num_obs_channels is not None:
            x = x[..., : self._num_obs_channels]
        cond_vec = self._conditioning_vector(x)
        gamma, beta = self.conditioner(cond_vec)

        feature_maps: list[tf.Tensor] = []
        for idx, block in enumerate(self.blocks):
            x = block(x, gamma=gamma[:, idx], beta=beta[:, idx], training=training)
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


# ==========================================================================
# Critics
# ==========================================================================


class SpatialSoftmaxCriticNetwork(network.Network):
    """Critic that evaluates supplied action xy coordinates via bilinear sampling."""

    def __init__(
        self,
        observation_spec,
        action_spec,
        name: str = 'SpatialSoftmaxCriticNetwork',
        encoder_cls=SpatialSoftmaxEncoder,
        use_heatmap: bool = False,
    ):
        super().__init__(input_tensor_spec=(observation_spec, action_spec), state_spec=(), name=name)
        self.encoder = encoder_cls(
            return_feature_maps=True,
            use_heatmap=use_heatmap,
        )
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

        local_feat = sample_feature_maps(feature_maps, xy)
        diff, grad, lap, scale_token = sample_obs_scalars(obs, xy)
        sampled_scalars = tf.concat([diff, grad, lap, scale_token], axis=-1)

        fused = tf.concat([latent, action_latent, local_feat, sampled_scalars], axis=-1)
        fused = self.fc1(fused)
        fused = self.fc2(fused)
        q = self.q_head(fused)
        return tf.squeeze(q, axis=-1), network_state


class SpatialValueMapCriticNetwork(network.Network):
    """Critic that predicts a spatial value map and samples it at the commanded (x, y)."""

    def __init__(
        self,
        observation_spec,
        action_spec,
        name: str = 'SpatialValueMapCriticNetwork',
        encoder_cls=SpatialSoftmaxEncoder,
        use_heatmap: bool = False,
    ):
        super().__init__(input_tensor_spec=(observation_spec, action_spec), state_spec=(), name=name)
        self.encoder = encoder_cls(
            return_feature_maps=True,
            use_heatmap=use_heatmap,
        )
        self.action_fc = layers.Dense(64, activation='relu')
        self.latent_fc = layers.Dense(64, activation='relu')
        self.fuse_fc = layers.Dense(64, activation='relu')
        self.q_bias = layers.Dense(1)
        self.conv1 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(1, 1, padding='same')

    def call(self, inputs, step_type=None, network_state=(), training: bool = False):
        observations, actions = inputs
        obs = tf.cast(observations, tf.float32)
        acts = tf.cast(actions, tf.float32)

        latent, _, feature_maps = self.encoder(obs, training=training)
        feat_map = feature_maps[-1]
        xy = acts[..., :2]
        dz = acts[..., 2:3]

        height = tf.shape(feat_map)[1]
        width = tf.shape(feat_map)[2]

        depth_plane = tf.tile(dz[:, None, None, :], [1, height, width, 1])
        scale_token = tf.reduce_mean(obs[..., 5:], axis=[1, 2], keepdims=True)
        scale_plane = tf.tile(scale_token, [1, height, width, 1])

        fused_map = tf.concat([feat_map, depth_plane, scale_plane], axis=-1)
        value_map = self.conv2(self.conv1(fused_map))

        q_spatial = bilinear_sample_nhwc(value_map, xy)
        q_spatial = tf.squeeze(q_spatial, axis=-1)

        act_latent = self.action_fc(acts)
        lat_latent = self.latent_fc(latent)
        fused = self.fuse_fc(tf.concat([lat_latent, act_latent], axis=-1))
        q_bias = tf.squeeze(self.q_bias(fused), axis=-1)

        q = q_spatial + q_bias
        return q, network_state


# ==========================================================================
# Actors
# ==========================================================================


class SpatialSoftmaxActorNetwork(network.Network):
    """Actor that anchors xy means using a spatial softmax heatmap."""

    def current_temperature(self) -> tf.Tensor | None:
        return self.encoder.current_temperature()

    def __init__(
        self,
        observation_spec,
        action_spec,
        name: str = 'SpatialSoftmaxActorNetwork',
        encoder_cls=SpatialSoftmaxEncoder,
        use_heatmap: bool = True,
    ):
        super().__init__(input_tensor_spec=observation_spec, state_spec=(), name=name)
        self._action_spec = action_spec
        self.encoder = encoder_cls(
            return_feature_maps=True,
            use_heatmap=use_heatmap,
        )
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
        local_feat = sample_feature_maps(feature_maps, xy)
        diff_xy, grad_xy, lap_xy, scale_token = sample_obs_scalars(obs, xy)
        local_feat = tf.concat([local_feat, diff_xy, grad_xy, lap_xy, scale_token], axis=-1)
        fused_in = tf.concat([latent, xy, scale_token], axis=-1)
        fused = self.fc1(fused_in)
        fused = self.fc2(fused)

        xy_clipped = tf.clip_by_value(xy, 1e-3, 1.0 - 1e-3)
        xy_pre = tf.clip_by_value(2.0 * xy_clipped - 1.0, -0.999, 0.999)
        xy_base = tf.math.atanh(xy_pre)
        depth_in = tf.concat([fused, local_feat], axis=-1)
        depth_h = self.depth_fc(depth_in)
        dz_base = self.dz_head(depth_h)
        base_mean = tf.concat([xy_base, dz_base], axis=-1)
        mean = base_mean + self.mean_residual(fused)

        logstd = self.logstd_head(fused)
        logstd = tf.clip_by_value(logstd, -3.0, 1.0)
        scale = tf.nn.softplus(logstd) + 1e-3

        dist = SquashedMultivariateNormalDiag(loc=mean,
                                              scale_diag=scale,
                                              bijector=make_tanh_squash_bijector())
        return dist, network_state


class SpatialKActorNetwork(network.Network):
    """Multi-modal spatial actor with straight-through Gumbel gating over K heatmaps."""

    def __init__(
        self,
        observation_spec,
        action_spec,
        K: int = 4,
        name: str = 'SpatialKActor',
        encoder_cls=SpatialSoftmaxEncoder,
    ):
        super().__init__(input_tensor_spec=observation_spec, state_spec=(), name=name)
        if K <= 0:
            raise ValueError('Number of modes K must be positive.')
        if action_spec.shape.rank != 1 or action_spec.shape[0] != 3:
            raise ValueError('SpatialKActorNetwork expects action spec with shape [3].')
        self._K = int(K)

        self.encoder = encoder_cls(
            return_feature_maps=True,
            use_heatmap=False,
        )
        self.heatmap_head = layers.Conv2D(self._K, 1, padding='same', name='mixture_heatmap_head')
        self.spatial_softmax = SpatialSoftmax()

        self.gating_gap = layers.GlobalAveragePooling2D()
        self.gating_fc1 = layers.Dense(128, activation='elu', name='gating_fc1')
        self.gating_fc2 = layers.Dense(64, activation='elu', name='gating_fc2')
        self.gating_head = layers.Dense(self._K, name='mode_logits')

        self.xy_res_fc1 = layers.Dense(128, activation='elu', name='xy_res_fc1')
        self.xy_res_fc2 = layers.Dense(64, activation='elu', name='xy_res_fc2')
        self.xy_res_head = layers.Dense(2, name='xy_residual_head')

        self.dz_hidden = layers.Dense(64, activation='elu', name='dz_hidden')
        self.dz_mean_head = layers.Dense(self._K, name='dz_mean_head')
        self.dz_logstd_head = layers.Dense(self._K, name='dz_logstd_head')

        self._xy_logstd = self.add_weight(
            name='xy_logstd',
            shape=(),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(-0.7),
            trainable=True,
        )
        self._dz_logstd_bias = self.add_weight(
            name='dz_logstd_bias',
            shape=(),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=True,
        )
        self._bijector = make_tanh_squash_bijector()

    def current_temperature(self) -> tf.Tensor | None:
        return self.spatial_softmax.current_temperature()

    def _gumbel_onehot(self, logits: tf.Tensor, tau: tf.Tensor, training_flag: tf.Tensor) -> tf.Tensor:
        logits = tf.cast(logits, tf.float32)
        tau = tf.cast(tau, tf.float32)

        training_mask = tf.cast(training_flag, tf.float32)
        uniform = tf.random.uniform(tf.shape(logits), minval=1e-6, maxval=1.0 - 1e-6, dtype=tf.float32)
        gumbel = -tf.math.log(-tf.math.log(uniform))
        gumbel *= training_mask

        y_soft = tf.nn.softmax((logits + gumbel) / tau, axis=-1)
        y_hard = tf.one_hot(tf.argmax(y_soft, axis=-1), depth=self._K, dtype=tf.float32)
        y_train = y_hard + tf.stop_gradient(y_soft - y_hard)

        return training_mask * y_train + (1.0 - training_mask) * y_hard

    def call(self, observations, step_type=None, network_state=(), training: bool = False):
        obs = tf.cast(observations, tf.float32)
        latent, _, feature_maps = self.encoder(obs, training=training)
        final_map = feature_maps[-1]

        heatmap_logits = self.heatmap_head(final_map)
        height = tf.shape(heatmap_logits)[1]
        width = tf.shape(heatmap_logits)[2]

        heatmaps = tf.transpose(heatmap_logits, [0, 3, 1, 2])
        heatmaps = tf.reshape(heatmaps, [-1, height, width, 1])
        coords = self.spatial_softmax(heatmaps)
        xy_modes = tf.reshape(coords, [-1, self._K, 2])

        gating_feat = self.gating_gap(final_map)
        gating_hidden = self.gating_fc1(gating_feat)
        gating_hidden = self.gating_fc2(gating_hidden)
        gating_logits = self.gating_head(gating_hidden)

        training_flag = tf.convert_to_tensor(training if training is not None else False, dtype=tf.bool)
        tau = tf.constant(0.1, dtype=tf.float32) + tf.cast(training_flag, tf.float32) * 0.6
        mode_weights = self._gumbel_onehot(gating_logits, tau, training_flag)

        probs = tf.nn.softmax(gating_logits, axis=-1)
        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
        entropy_loss = 0.001 * tf.reduce_mean(entropy)
        self.add_loss(tf.cast(training_flag, tf.float32) * entropy_loss)

        mode_weights_expanded = tf.expand_dims(mode_weights, axis=-1)
        xy_selected = tf.reduce_sum(mode_weights_expanded * xy_modes, axis=1)
        xy_clipped = tf.clip_by_value(xy_selected, 1e-3, 1.0 - 1e-3)
        xy_pre = tf.clip_by_value(2.0 * xy_clipped - 1.0, -0.999, 0.999)
        xy_base = tf.math.atanh(xy_pre)

        xy_residual_input = tf.concat([latent, xy_selected], axis=-1)
        xy_residual = self.xy_res_head(self.xy_res_fc2(self.xy_res_fc1(xy_residual_input)))
        xy_mean = xy_base + xy_residual

        diff_samples = []
        grad_samples = []
        lap_samples = []
        for mode_idx in range(self._K):
            coord = xy_modes[:, mode_idx, :]
            diff_samples.append(bilinear_sample_nhwc(obs[..., 0:1], coord))
            grad_samples.append(bilinear_sample_nhwc(obs[..., 3:4], coord))
            lap_samples.append(bilinear_sample_nhwc(obs[..., 4:5], coord))
        diff_stack = tf.stack(diff_samples, axis=1)
        grad_stack = tf.stack(grad_samples, axis=1)
        lap_stack = tf.stack(lap_samples, axis=1)
        scale_token = tf.reduce_mean(obs[..., 5:], axis=[1, 2])  # [B,1]
        tok_scale_stack = tf.tile(scale_token[:, None, :], [1, self._K, 1])
        sampled_scalars = tf.concat([diff_stack, grad_stack, lap_stack, tok_scale_stack], axis=-1)
        sampled_flat = tf.reshape(sampled_scalars, [-1, self._K * (3 + 1)])

        dz_hidden = self.dz_hidden(sampled_flat)
        dz_means = self.dz_mean_head(dz_hidden)
        dz_logstd = self.dz_logstd_head(dz_hidden) + self._dz_logstd_bias

        dz_mean_selected = tf.reduce_sum(dz_means * mode_weights, axis=-1, keepdims=True)
        dz_logstd_selected = tf.reduce_sum(dz_logstd * mode_weights, axis=-1, keepdims=True)

        xy_logstd = tf.clip_by_value(self._xy_logstd, -3.0, 1.0)
        xy_scale_scalar = tf.nn.softplus(xy_logstd) + 1e-3
        xy_scale = tf.ones_like(xy_mean) * xy_scale_scalar

        dz_logstd_selected = tf.clip_by_value(dz_logstd_selected, -3.0, 1.0)
        dz_scale = tf.nn.softplus(dz_logstd_selected) + 1e-3

        mean = tf.concat([xy_mean, dz_mean_selected], axis=-1)
        scale = tf.concat([xy_scale, dz_scale], axis=-1)

        dist = SquashedMultivariateNormalDiag(loc=mean,
                                              scale_diag=scale,
                                              bijector=self._bijector)
        return dist, network_state


__all__ = [
    'SpatialSoftmax',
    'SpatialSoftmaxEncoder',
    'SpatialFiLMEncoder',
    'SpatialSoftmaxCriticNetwork',
    'SpatialValueMapCriticNetwork',
    'SpatialSoftmaxActorNetwork',
    'SpatialKActorNetwork',
]

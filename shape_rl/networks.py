"""Network components for Shape RL."""

from __future__ import annotations

import tensorflow as tf
from keras import layers, models
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


# ==========================================================================
#  Feature Pyramid Encoder utilities
# ==========================================================================


class CoordConv(layers.Layer):
    """Appends normalized xy coordinates before applying a depthwise separable conv."""

    def __init__(self, filters: int = 32, kernel_size: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.SeparableConv2D(filters, kernel_size, padding='same', activation='relu')
        self._coords: tf.Tensor | None = None

    def build(self, input_shape):
        _, height, width, _ = input_shape
        xs = tf.linspace(-1.0, 1.0, width)
        ys = tf.linspace(-1.0, 1.0, height)
        xs = tf.reshape(xs, [1, 1, width])
        xs = tf.tile(xs, [1, height, 1])
        ys = tf.reshape(ys, [1, height, 1])
        ys = tf.tile(ys, [1, 1, width])
        self._coords = tf.stack([xs, ys], axis=-1)  # [1, H, W, 2]
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch = tf.shape(inputs)[0]
        coords = tf.tile(self._coords, [batch, 1, 1, 1])
        conv_input = tf.concat([inputs, coords], axis=-1)
        return self.conv(conv_input)


class FPNBlock(layers.Layer):
    """Depthwise separable residual block with optional projection."""

    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.shortcut: layers.Layer | None = None
        self.conv1 = layers.SeparableConv2D(filters, 3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.SeparableConv2D(filters, 3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

    def build(self, input_shape):
        input_channels = input_shape[-1]
        if input_channels != self.filters:
            self.shortcut = layers.SeparableConv2D(self.filters, 1, padding='same')
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        shortcut = self.shortcut(inputs) if self.shortcut is not None else inputs
        return self.relu(shortcut + x)


class FPNEncoder(layers.Layer):
    """Multi-scale encoder that fuses features across pyramid levels."""

    def __init__(self, filters_list: tuple[int, ...] = (32, 64, 128), latent_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.coordconv = CoordConv()
        self.blocks = [FPNBlock(f) for f in filters_list]
        self.pools = [layers.MaxPool2D() for _ in filters_list]
        head_channels = filters_list[-1]
        self.lateral_convs = [layers.SeparableConv2D(head_channels, 1, padding='same') for _ in filters_list]
        self.upsamples = [layers.UpSampling2D(size=2) for _ in filters_list[:-1]]
        self.merge_upsamples = [layers.UpSampling2D(size=2 ** i) for i in range(len(filters_list))]
        self.global_pool = layers.GlobalAveragePooling2D()
        self.latent = layers.Dense(latent_dim, activation='relu')

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = self.coordconv(inputs)
        pyramid: list[tf.Tensor] = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x, training=training)
            pyramid.append(x)
            x = pool(x)

        last = len(pyramid) - 1
        fused = [None] * len(pyramid)
        fused[last] = self.lateral_convs[last](pyramid[last])
        for level in range(last - 1, -1, -1):
            fused[level] = self.lateral_convs[level](pyramid[level]) + self.upsamples[level](fused[level + 1])

        merged = tf.concat(
            [self.merge_upsamples[idx](fused[idx]) for idx in range(len(fused))], axis=-1
        )
        pooled = self.global_pool(merged)
        return self.latent(pooled)


class SpatialSoftmax(layers.Layer):
    """Computes spatial expectation for heatmap logits."""

    def call(self, logits: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(logits)
        batch = shape[0]
        height = shape[1]
        width = shape[2]

        flat = tf.reshape(logits, [batch, height * width])
        weights = tf.nn.softmax(flat)

        xs = tf.linspace(0.0, 1.0, width)
        ys = tf.linspace(0.0, 1.0, height)
        grid_x, grid_y = tf.meshgrid(xs, ys)
        coords = tf.stack([tf.reshape(grid_x, [-1]), tf.reshape(grid_y, [-1])], axis=-1)
        return tf.matmul(weights, coords)


# ==========================================================================
#  FPN-based actor / critic
# ==========================================================================


class FPNActorNetwork(network.Network):
    """Actor that encodes observations with an FPN and outputs diagonal Gaussian actions."""

    def __init__(self, observation_spec, action_spec, name: str = 'FPNActorNetwork'):
        super().__init__(input_tensor_spec=observation_spec, state_spec=(), name=name)
        self._action_spec = action_spec
        self.encoder = FPNEncoder(latent_dim=128)
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        action_dims = action_spec.shape[0]
        self.mean_head = layers.Dense(action_dims)
        self.logstd_head = layers.Dense(action_dims)

    def call(self, observations, step_type=None, network_state=(), training: bool = False):
        x = tf.cast(observations, tf.float32)
        latent = self.encoder(x, training=training)
        x = self.fc1(latent)
        x = self.fc2(x)
        mean = self.mean_head(x)
        logstd = self.logstd_head(x)
        logstd = tf.clip_by_value(logstd, -3.0, 1.0)
        scale = tf.nn.softplus(logstd) + 1e-3

        base = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=scale)
        bijector = tfp.bijectors.Chain(
            [tfp.bijectors.Shift(0.5), tfp.bijectors.Scale(0.5), tfp.bijectors.Tanh()]
        )
        dist = tfp.distributions.TransformedDistribution(base, bijector)
        return dist, network_state


class FPNCriticNetwork(network.Network):
    """Critic that encodes observations and actions independently before fusion."""

    def __init__(self, observation_spec, action_spec, name: str = 'FPNCriticNetwork'):
        super().__init__(input_tensor_spec=(observation_spec, action_spec), state_spec=(), name=name)
        self.encoder = FPNEncoder(latent_dim=128)
        self.action_fc = layers.Dense(64, activation='relu')
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.q_head = layers.Dense(1)

    def call(self, inputs, step_type=None, network_state=(), training: bool = False):
        observations, actions = inputs
        obs = tf.cast(observations, tf.float32)
        acts = tf.cast(actions, tf.float32)

        latent = self.encoder(obs, training=training)
        action_latent = self.action_fc(acts)
        fused = tf.concat([latent, action_latent], axis=-1)
        fused = self.fc1(fused)
        fused = self.fc2(fused)
        q = self.q_head(fused)
        return tf.squeeze(q, axis=-1), network_state


# ==========================================================================
#  Spatial-softmax encoder family
# ==========================================================================


class _ConvBlock(layers.Layer):
    """Conv-BN-ReLU block with configurable stride."""

    def __init__(self, filters: int, stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters, 3, strides=stride, padding='same')
        self.norm = layers.BatchNormalization()
        self.act = layers.Activation('relu')

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = self.conv(inputs)
        x = self.norm(x, training=training)
        return self.act(x)


class SpatialSoftmaxEncoder(layers.Layer):
    """CNN backbone that predicts a spatial expectation alongside a latent vector."""

    def __init__(
        self,
        filters: tuple[int, ...] = (32, 64, 128),
        latent_dim: int = 128,
        return_feature_maps: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.blocks = []
        for idx, f in enumerate(filters):
            stride = 1 if idx == 0 else 2
            self.blocks.append(_ConvBlock(f, stride=stride))
        self.heatmap_head = layers.Conv2D(1, 1, padding='same')
        self.spatial_softmax = SpatialSoftmax()
        self.global_pool = layers.GlobalAveragePooling2D()
        self.latent = layers.Dense(latent_dim, activation='relu')
        self._return_feature_maps = return_feature_maps

    def call(
        self, inputs: tf.Tensor, training: bool | None = None
    ) -> tuple[tf.Tensor, tf.Tensor] | tuple[tf.Tensor, tf.Tensor, tuple[tf.Tensor, ...]]:
        x = tf.cast(inputs, tf.float32)
        feature_maps: list[tf.Tensor] = []
        for block in self.blocks:
            x = block(x, training=training)
            feature_maps.append(x)
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
    """Actor that anchors xy means using a spatial softmax heatmap."""

    def __init__(self, observation_spec, action_spec, name: str = 'SpatialSoftmaxActorNetwork'):
        super().__init__(input_tensor_spec=observation_spec, state_spec=(), name=name)
        self._action_spec = action_spec
        self.encoder = SpatialSoftmaxEncoder(latent_dim=128)
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.dz_head = layers.Dense(1)
        action_dims = action_spec.shape[0]
        self.mean_residual = layers.Dense(action_dims)
        self.logstd_head = layers.Dense(action_dims)

    def call(self, observations, step_type=None, network_state=(), training: bool = False):
        obs = tf.cast(observations, tf.float32)
        latent, xy = self.encoder(obs, training=training)
        fused = tf.concat([latent, xy], axis=-1)
        fused = self.fc1(fused)
        fused = self.fc2(fused)

        xy_clipped = tf.clip_by_value(xy, 1e-3, 1.0 - 1e-3)
        xy_pre = tf.clip_by_value(2.0 * xy_clipped - 1.0, -0.999, 0.999)
        xy_base = tf.math.atanh(xy_pre)
        dz_base = self.dz_head(fused)
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
    """Critic that shares the spatial-softmax encoder and compares xy predictions."""

    def __init__(self, observation_spec, action_spec, name: str = 'SpatialSoftmaxCriticNetwork'):
        super().__init__(input_tensor_spec=(observation_spec, action_spec), state_spec=(), name=name)
        self.encoder = SpatialSoftmaxEncoder(latent_dim=128, return_feature_maps=True)
        self.action_fc = layers.Dense(64, activation='relu')
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.q_head = layers.Dense(1)

    def call(self, inputs, step_type=None, network_state=(), training: bool = False):
        observations, actions = inputs
        obs = tf.cast(observations, tf.float32)
        acts = tf.cast(actions, tf.float32)

        latent, predicted_xy, feature_maps = self.encoder(obs, training=training)
        xy = acts[..., :2]
        action_latent = self.action_fc(acts)

        local_feats = [bilinear_sample_nhwc(feature_map, xy) for feature_map in feature_maps]
        local_feat = local_feats[0] if len(local_feats) == 1 else tf.concat(local_feats, axis=-1)

        xy_delta = xy - predicted_xy
        fused = tf.concat([latent, action_latent, xy_delta, local_feat], axis=-1)
        fused = self.fc1(fused)
        fused = self.fc2(fused)
        q = self.q_head(fused)
        return tf.squeeze(q, axis=-1), network_state


# ==========================================================================
#  Auxiliary encoders
# ==========================================================================


def build_unet_encoder(input_shape, latent_dim: int = 256):
    """Returns a lightweight UNet-style encoder producing a latent vector."""

    inputs = layers.Input(shape=input_shape)
    c1 = layers.SeparableConv2D(32, 3, padding='same', activation='relu')(inputs)
    c1 = layers.SeparableConv2D(32, 3, padding='same', activation='relu')(c1)
    p1 = layers.MaxPool2D()(c1)

    c2 = layers.SeparableConv2D(64, 3, padding='same', activation='relu')(p1)
    c2 = layers.SeparableConv2D(64, 3, padding='same', activation='relu')(c2)
    p2 = layers.MaxPool2D()(c2)

    c3 = layers.SeparableConv2D(128, 3, padding='same', activation='relu')(p2)
    c3 = layers.SeparableConv2D(128, 3, padding='same', activation='relu')(c3)
    p3 = layers.MaxPool2D()(c3)

    bottleneck = layers.SeparableConv2D(128, 3, padding='same', activation='relu')(p3)
    bottleneck = layers.SeparableConv2D(128, 3, padding='same', activation='relu')(bottleneck)

    u1 = layers.UpSampling2D()(bottleneck)
    u1 = layers.Concatenate()([u1, c3])
    c4 = layers.SeparableConv2D(128, 3, padding='same', activation='relu')(u1)
    c4 = layers.SeparableConv2D(128, 3, padding='same', activation='relu')(c4)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = layers.SeparableConv2D(64, 3, padding='same', activation='relu')(u2)
    c5 = layers.SeparableConv2D(64, 3, padding='same', activation='relu')(c5)

    u3 = layers.UpSampling2D()(c5)
    u3 = layers.Concatenate()([u3, c1])
    c6 = layers.SeparableConv2D(32, 3, padding='same', activation='relu')(u3)
    c6 = layers.SeparableConv2D(32, 3, padding='same', activation='relu')(c6)

    pooled = layers.GlobalAveragePooling2D()(c6)
    latent = layers.Dense(latent_dim, activation='relu')(pooled)
    return models.Model(inputs, latent, name='unet_encoder')


class GatedConvBlock(layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.SeparableConv2D(2 * filters, 3, padding='same')
        self.norm = layers.BatchNormalization()

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = self.conv(inputs)
        x = self.norm(x, training=training)
        a, b = tf.split(x, num_or_size_splits=2, axis=-1)
        return tf.nn.tanh(a) * tf.nn.sigmoid(b)


def build_gated_encoder(input_shape, latent_dim: int = 128):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for filters in (32, 64, 128):
        x = GatedConvBlock(filters)(x)
        x = layers.MaxPool2D()(x)
    x = GatedConvBlock(128)(x)
    x = layers.GlobalAveragePooling2D()(x)
    latent = layers.Dense(latent_dim, activation='relu')(x)
    return models.Model(inputs, latent, name='gated_encoder')


__all__ = [
    'CoordConv',
    'FPNBlock',
    'FPNEncoder',
    'SpatialSoftmax',
    'FPNActorNetwork',
    'FPNCriticNetwork',
    'SpatialSoftmaxEncoder',
    'SpatialSoftmaxActorNetwork',
    'SpatialSoftmaxCriticNetwork',
    'build_unet_encoder',
    'build_gated_encoder',
]

"""Network components for Shape RL."""

import tensorflow as tf
from keras import layers, models
from tf_agents.networks import actor_distribution_network, network
from tf_agents.agents.ddpg.critic_network import CriticNetwork
import tensorflow_probability as tfp
import numpy as np
from shape_rl.policies.heuristic import HeuristicPressPolicy

# ==================== FPN/CoordConv/Custom Actor & Critic ====================
class CoordConv(layers.Layer):
    def __init__(self, filters=32, kernel_size=1, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.SeparableConv2D(filters, kernel_size, padding='same', activation='relu')
        self.coords = None

    def build(self, input_shape):
        # Precompute coordinate grid [1, H, W, 2]
        batch_dim, h, w, _ = input_shape
        xx = tf.linspace(-1.0, 1.0, w)
        yy = tf.linspace(-1.0, 1.0, h)
        xx = tf.reshape(xx, [1, 1, w])
        xx = tf.tile(xx, [1, h, 1])
        yy = tf.reshape(yy, [1, h, 1])
        yy = tf.tile(yy, [1, 1, w])
        self.coords = tf.stack([xx, yy], axis=-1)  # shape [1,h,w,2]
        super().build(input_shape)

    def call(self, x):
        # x is expected to be [B, H, W, C]
        batch_size = tf.shape(x)[0]
        coords = tf.tile(self.coords, [batch_size, 1, 1, 1])
        conv_input = tf.concat([x, coords], axis=-1)
        return self.conv(conv_input)

class FPNBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.shortcut = None
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

    def call(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.shortcut is not None:
            x_proj = self.shortcut(x)
        else:
            x_proj = x
        out = self.relu(x_proj + y)
        return out

class FPNEncoder(layers.Layer):
    def __init__(self, filters_list=(32, 64, 128), latent_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.coordconv = CoordConv()
        self.downs = []
        for f in filters_list:
            self.downs.append(FPNBlock(f))
        self.pools = [layers.MaxPool2D() for _ in filters_list]
        # FPN lateral and upsample layers
        fpn_channels = filters_list[-1]
        self.lateral_convs = [layers.SeparableConv2D(fpn_channels, 1, padding='same') for _ in filters_list]
        self.upsamples     = [layers.UpSampling2D(size=2) for _ in filters_list[:-1]]
        self.merge_upsamples = [layers.UpSampling2D(size=2**i) for i in range(len(filters_list))]
        self.global_pool = layers.GlobalAveragePooling2D()
        self.latent = layers.Dense(latent_dim, activation='relu')
    def call(self, x):
        # Bottom-up pass
        x = self.coordconv(x)
        c_feats = []
        for block, pool in zip(self.downs, self.pools):
            x = block(x)
            c_feats.append(x)
            x = pool(x)
        # Top-down lateral fusion
        p_levels = [None] * len(c_feats)
        last = len(c_feats) - 1
        p_levels[last] = self.lateral_convs[last](c_feats[last])
        for i in range(last - 1, -1, -1):
            p_levels[i] = self.lateral_convs[i](c_feats[i]) + self.upsamples[i](p_levels[i+1])
        # Merge multi-scale features
        merged = tf.concat([self.merge_upsamples[i](p_levels[i]) for i in range(len(p_levels))], axis=-1)
        x = self.global_pool(merged)
        return self.latent(x)
    
class SpatialSoftmax(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, logits):
        # logits: [B,H,W,1]
        # Extract batch and spatial dims without Python iteration
        shape_un = tf.unstack(tf.shape(logits))
        b = shape_un[0]
        h = shape_un[1]
        w = shape_un[2]
        flat = tf.reshape(logits, [b, h * w])
        prob = tf.nn.softmax(flat)
        coords_x, coords_y = tf.meshgrid(
            tf.linspace(0.0, 1.0, w), tf.linspace(0.0, 1.0, h)
        )
        coords = tf.stack([tf.reshape(coords_x, [-1]), tf.reshape(coords_y, [-1])], axis=1)
        exp = tf.matmul(prob, coords)
        return exp  # [B,2]

class CarveActorNetwork(network.Network):
    def __init__(self, observation_spec, action_spec, name='CarveActorNetwork'):
        super().__init__(input_tensor_spec=observation_spec, state_spec=(), name=name)
        self._action_spec = action_spec
        self.encoder = FPNEncoder(latent_dim=128)
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        # Output mean/logstd for each action dimension
        self.mean = layers.Dense(action_spec.shape[0])
        self.logstd = layers.Dense(action_spec.shape[0])
    def call(self, observations, step_type=None, network_state=(), training=False):
        x = tf.cast(observations, tf.float32)
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.fc2(x)
        mean = self.mean(x)
        logstd = self.logstd(x)
        # Prevent extreme log-std values and enforce minimum scale
        logstd = tf.clip_by_value(logstd, -3.0, 1.0)
        std = tf.nn.softplus(logstd) + 1e-3

        # Base Gaussian distribution
        base_dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)
        # Squash with tanh to [-1,1], then scale to [–0.5,0.5] and shift to [0,1]
        bijector = tfp.bijectors.Chain([
            tfp.bijectors.Shift(shift=0.5),
            tfp.bijectors.Scale(scale=0.5),
            tfp.bijectors.Tanh()
        ])
        dist = tfp.distributions.TransformedDistribution(distribution=base_dist, bijector=bijector)
        return dist, network_state

class CarveCriticNetwork(network.Network):
    def __init__(self, observation_spec, action_spec, name='CarveCriticNetwork'):
        super().__init__(input_tensor_spec=(observation_spec, action_spec), state_spec=(), name=name)
        self.encoder = FPNEncoder(latent_dim=128)
        self.action_fc = layers.Dense(64, activation='relu')
        self.concat_fc1 = layers.Dense(128, activation='relu')
        self.concat_fc2 = layers.Dense(64, activation='relu')
        self.q_out = layers.Dense(1)
        # Gaussian attention parameters
        self._gauss_sigma = 0.05  # width of Gaussian bump in normalized coords
        self._grid = None

    def build(self, input_shape):
        # input_shape: tuple(obs_shape, action_shape)
        obs_shape, _ = input_shape
        # obs_shape: (batch, H, W, C)
        _, H, W, _ = obs_shape
        # Create normalized coordinate grid [0,1] for x and y
        ys = tf.linspace(0.0, 1.0, H)
        xs = tf.linspace(0.0, 1.0, W)
        grid_y, grid_x = tf.meshgrid(ys, xs, indexing='ij')  # shape [H, W]
        grid = tf.stack([grid_x, grid_y], axis=-1)          # [H, W, 2]
        grid = tf.reshape(grid, [1, H, W, 2])                # [1, H, W, 2]
        self._grid = tf.cast(grid, tf.float32)
        super().build(input_shape)

    def call(self, inputs, step_type=None, network_state=(), training=False):
        obs, actions = inputs
        # Cast observation
        obs = tf.cast(obs, tf.float32)  # [B, H, W, C]
        # Gaussian attention map from action (x,y)
        # Extract normalized x,y from actions
        xy = actions[..., :2]  # [B, 2]
        # Reshape to [B, 1, 1, 2]
        xy = tf.reshape(xy, [-1, 1, 1, 2])
        # Compute squared distance on grid (broadcasting grid over batch)
        # self._grid: [1, H, W, 2]
        dist2 = tf.reduce_sum((self._grid - xy)**2, axis=-1, keepdims=True)  # [B, H, W, 1]
        gauss = tf.exp(-dist2 / (2 * self._gauss_sigma**2))                   # [B, H, W, 1]
        # Augment observation with Gaussian bump channel
        obs_aug = tf.concat([obs, gauss], axis=-1)  # [B, H, W, C+1]
        # Encode augmented observation
        x = self.encoder(obs_aug)
        # Process action latents
        a = self.action_fc(actions)
        x = tf.concat([x, a], axis=-1)
        x = self.concat_fc1(x)
        x = self.concat_fc2(x)
        q = self.q_out(x)
        return tf.squeeze(q, axis=-1), network_state

# ---------- UNet Encoder ----------------------------------------------------
def build_unet_encoder(input_shape, latent_dim=256):
    """
    Returns a keras.Model that maps an (H,W,C) observation to a latent vector
    of length `latent_dim`.  A lightweight 2‑down 2‑up UNet.
    """
    inputs = layers.Input(shape=input_shape)
    # Down 1
    c1 = layers.SeparableConv2D(32, 3, padding='same', activation='relu')(inputs)
    c1 = layers.SeparableConv2D(32, 3, padding='same', activation='relu')(c1)
    p1 = layers.MaxPool2D()(c1)
    # Down 2
    c2 = layers.SeparableConv2D(64, 3, padding='same', activation='relu')(p1)
    c2 = layers.SeparableConv2D(64, 3, padding='same', activation='relu')(c2)
    p2 = layers.MaxPool2D()(c2)
    # Down 3
    c3 = layers.SeparableConv2D(128, 3, padding='same', activation='relu')(p2)
    c3 = layers.SeparableConv2D(128, 3, padding='same', activation='relu')(c3)
    p3 = layers.MaxPool2D()(c3)
    # Bottleneck
    b  = layers.SeparableConv2D(128, 3, padding='same', activation='relu')(p3)
    b  = layers.SeparableConv2D(128, 3, padding='same', activation='relu')(b)
    # Up 1
    u1 = layers.UpSampling2D()(b)
    u1 = layers.Concatenate()([u1, c3])
    c4 = layers.SeparableConv2D(128, 3, padding='same', activation='relu')(u1)
    c4 = layers.SeparableConv2D(128, 3, padding='same', activation='relu')(c4)
    # Up 2
    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = layers.SeparableConv2D(64, 3, padding='same', activation='relu')(u2)
    c5 = layers.SeparableConv2D(64, 3, padding='same', activation='relu')(c5)
    # Up 3
    u3 = layers.UpSampling2D()(c5)
    u3 = layers.Concatenate()([u3, c1])
    c6 = layers.SeparableConv2D(32, 3, padding='same', activation='relu')(u3)
    c6 = layers.SeparableConv2D(32, 3, padding='same', activation='relu')(c6)
    pooled = layers.GlobalAveragePooling2D()(c6)
    latent = layers.Dense(latent_dim, activation='relu')(pooled)
    return models.Model(inputs, latent, name='unet_encoder')


class GatedConvBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv = layers.SeparableConv2D(2 * filters, 3, padding='same')
        self.norm = layers.BatchNormalization()

    def call(self, x):
        y = self.conv(x)
        y = self.norm(y)
        a, b = tf.split(y, num_or_size_splits=2, axis=-1)
        return tf.nn.tanh(a) * tf.nn.sigmoid(b)


def build_gated_encoder(input_shape, latent_dim=128):
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
    "CoordConv",
    "FPNBlock",
    "FPNEncoder",
    "SpatialSoftmax",
    "CarveActorNetwork",
    "CarveCriticNetwork",
    "build_unet_encoder",
    "build_gated_encoder",
    "HeuristicPressPolicy",
]

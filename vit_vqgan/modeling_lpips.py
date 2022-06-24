# JAX port of `lpips`, as implemented in:
# - Taming Transformers (https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/lpips.py)
# - richzhang/PerceptualSimilarity (https://github.com/richzhang/PerceptualSimilarity/blob/31bc1271ae6f13b7e281b9959ac24a5e8f2ed522/lpips/pretrained_networks.py)
# Simplified to only include the features we need.

import h5py
import jax.numpy as jnp
import flax.linen as nn

from flaxmodels import VGG16

class NetLinLayer(nn.Module):
    weights: jnp.array
    kernel_size = (1,1)
    
    def setup(self):
        w = lambda *_ : self.weights
        self.layer = nn.Conv(1, self.kernel_size, kernel_init=w, strides=None, padding=0, use_bias=False)
        
    def __call__(self, x):
        x = self.layer(x)
        return x

class LPIPS(nn.Module):
    # TODO: permanent URL
    lin_weights_filename = 'pretrained/lpips_lin.h5'
    
    def setup(self):
        # We don't add a scaling layer because `VGG16` already includes it
        self.feature_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        
        self.vgg = VGG16(output='activations', pretrained='imagenet', include_head=False)
        
        lin_weights = h5py.File(self.lin_weights_filename)
        self.lins = [NetLinLayer(jnp.array(lin_weights[f'lin{i}'])) for i in range(len(self.feature_names))]
        
    def __call__(self, x, t):
        x = self.vgg((x + 1) / 2)
        t = self.vgg((t + 1) / 2)
            
        feats_x, feats_t, diffs = {}, {}, {}
        for i, f in enumerate(self.feature_names):
            feats_x[i], feats_t[i] = normalize_tensor(x[f]), normalize_tensor(t[f])
            diffs[i] = (feats_x[i] - feats_t[i]) ** 2

        # We should maybe vectorize this better
        res = [spatial_average(self.lins[i](diffs[i]), keepdims=True) for i in range(len(self.feature_names))]
        
        val = res[0]
        for i in range(1, len(res)):
            val += res[i]
        return val

def normalize_tensor(x, eps=1e-10):
    # Use `-1` because we are channel-last
    norm_factor = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True))
    return x / (norm_factor + eps)

def spatial_average(x, keepdims=True):
    # Mean over W, H
    return jnp.mean(x, axis=[1, 2], keepdims=keepdims)

from taming.modules.losses.lpips import LPIPS as taming_LPIPS
import torch

x = torch.rand([8, 3, 256, 256])
y = torch.rand([8, 3, 256, 256])

torch_lpips = taming_LPIPS()
print(torch_lpips(x, y))

import random
import jax
import jax.numpy as jnp
from vit_vqgan.modeling_lpips import LPIPS

seed = random.randint(0, 2**32 - 1)
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)

x = jnp.array(x).transpose(0, 2, 3, 1)
y = jnp.array(y).transpose(0, 2, 3, 1)

lpips = LPIPS()
params = lpips.init(key, x, x)

result = lpips.apply(params, x, y)
print(result)
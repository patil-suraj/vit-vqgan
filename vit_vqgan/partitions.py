import re

from flax.core.frozen_dict import freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.experimental import PartitionSpec as P

# utils adapted from https://github.com/google-research/google-research/blob/master/flax_models/t5x/partitions.py
# Sentinels
_unmatched = object()

# For specifying empty leaf dict `{}`
empty_dict = object()


def _match(qs, ks):
    """Return True if regexes in qs match any window of strings in tuple ks."""
    # compile regexes and force complete match
    qts = tuple(map(lambda x: re.compile(x + "$"), qs))
    for i in range(len(ks) - len(qs) + 1):
        matches = [x.match(y) for x, y in zip(qts, ks[i:])]
        if matches and all(matches):
            return True
    return False


def _replacement_rules(rules):
    def replace(key, val):
        for rule, replacement in rules:
            if _match(rule, key):
                return replacement
        return val

    return replace


def _get_partition_rules():
    return [
        # embeddings
        (
            ("pos_embedding",),
            P(None, None, "mp"),
        ),
        (
            ("patch_embeds", "kernel"),
            P(None, None, None, "mp"),
        ),
        (
            ("codebook_embedding",),
            P("mp", None),
        ),
        # attention
        (("(q_proj|k_proj|v_proj)", "kernel"), P(None, "mp")),
        (("out_proj", "kernel"), P("mp", None)),
        # FFN
        (("fc1", "kernel"), P("mp", None)),
        (("fc2", "kernel"), P("mp", None)),
        (("fc_out", "kernel"), P(None, "mp")),
        (("(bias|scale|logit_scale)",), None),
        # Conv
        (("mid_ffn_conv", "kernel"), P(None, None, None, "mp")),
        # projection
        (("factor_in", "kernel"), P("mp", None)),
        (("factor_out", "kernel"), P(None, "mp")),
        (("quantizer", "codebook_embedding"), P("mp", None)),
        (("to_image", "kernel"), P(None, None, "mp", None)),
        # discriminator/lpips
        (("conv.*", "kernel"), P(None, None, None, "mp")),
        # discriminator
        (("classifier", "kernel"), None),
        (("dense", "kernel"), P("mp", None)),
        # lpips
        (("layer", "kernel"), P(None, None, "mp", None)),
    ]


def set_partitions(in_dict, use_scan):
    rules = _get_partition_rules()
    replace = _replacement_rules(rules)
    initd = {k: _unmatched for k in flatten_dict(in_dict)}
    result = {k: replace(k, v) for k, v in initd.items()}
    for k, v in result.items():
        if v == _unmatched:
            print(f"Unmatched -> {k}")
    if use_scan:
        # add None dimension to layers
        result = {k: (P(*(None,) + v) if v is not None else None) if "scanned" in k else v for k, v in result.items()}
    assert _unmatched not in result.values(), "Incomplete partition spec."
    return freeze(unflatten_dict(result))

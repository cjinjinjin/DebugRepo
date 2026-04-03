"""
Monkey-patch for ms-swift 4.1.0.dev0 bug:
_colocate_rollout passes prompt_id/request_id through to model.generate(),
which raises ValueError("model_kwargs not used").

This patch wraps _engine_infer to strip those keys before they reach
TransformersEngine.infer().

Loaded via --external_plugins alongside reward_grpo.py.
"""

from swift.rlhf_trainers.rollout_mixin import RolloutTrainerMixin

_orig_engine_infer = RolloutTrainerMixin._engine_infer


def _patched_engine_infer(self, infer_requests, request_config, **kwargs):
    _strip_keys = {'prompt_id', 'request_id'}
    cleaned = []
    for req in infer_requests:
        if isinstance(req, dict):
            cleaned.append({k: v for k, v in req.items() if k not in _strip_keys})
        else:
            cleaned.append(req)
    return _orig_engine_infer(self, cleaned, request_config, **kwargs)


RolloutTrainerMixin._engine_infer = _patched_engine_infer
print('[patch] _engine_infer patched: stripping prompt_id/request_id from colocate inputs')

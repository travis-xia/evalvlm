import os

from .base import ModelAdapter, register_adapter


@register_adapter('qwen3')
class Qwen3Adapter(ModelAdapter):
    """For OpenAI/vLLM paths, sets ``mm_processor_kwargs.max_pixels`` on the request.

    Override pixel cap without code changes::

        export VLMEVAL_QWEN_MAX_PIXELS=1003520
        python run.py ... --base-url http://127.0.0.1:8000/v1 --custom-prompt qwen3

    If unset and ``max_pixels`` is not passed to the constructor, no ``mm_processor_kwargs`` is added.
    """

    def __init__(self, max_pixels=None):
        if max_pixels is None:
            raw = os.environ.get('VLMEVAL_QWEN_MAX_PIXELS', '').strip()
            if raw:
                max_pixels = int(raw)
        self.max_pixels = max_pixels

    def process_payload(self, payload, dataset=None):
        if self.max_pixels is not None and self.max_pixels > 0:
            payload = payload.copy()
            payload['mm_processor_kwargs'] = {'max_pixels': self.max_pixels}
        return payload

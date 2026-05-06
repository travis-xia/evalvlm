from __future__ import annotations
import os
import shutil
import tempfile
import uuid
import warnings
from pathlib import Path

import numpy as np

from vlmeval.api.base import BaseAPI
from vlmeval.smp import get_logger, proxy_set
from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin

logger = get_logger(__name__)


def _ensure_unique_local_media_url(path: str, prefixes: list[str], media_type: str) -> str:
    if any(path.startswith(prefix) for prefix in prefixes):
        return path
    if not os.path.exists(path):
        raise ValueError(f'Invalid {media_type}: {path}')

    src = Path(path).resolve()
    tmp_dir = Path(tempfile.gettempdir()) / 'vlmeval_dashscope_uploads' / media_type
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dst = tmp_dir / f'{src.stem}-{uuid.uuid4().hex}{src.suffix}'
    shutil.copy2(src, dst)
    return f'file://{dst}'


def ensure_image_url(image: str) -> str:
    return _ensure_unique_local_media_url(
        image,
        prefixes=['http://', 'https://', 'file://', 'data:image;'],
        media_type='image',
    )


def ensure_video_url(video: str) -> str:
    return _ensure_unique_local_media_url(
        video,
        prefixes=['http://', 'https://', 'file://', 'data:video;'],
        media_type='video',
    )


def _multimodal_response_text_segments(response) -> list[str]:
    """Collect ordered 'text' fields from assistant message content."""
    if response is None:
        return []
    try:
        out = response.output
        choices = out.choices
        c0 = choices[0]
        if isinstance(c0, dict):
            msg = c0.get('message')
        else:
            msg = getattr(c0, 'message', None)
        if msg is None:
            return []
        if isinstance(msg, dict):
            content = msg.get('content')
        else:
            content = getattr(msg, 'content', None)
        if not content:
            return []
    except (AttributeError, IndexError, KeyError, TypeError):
        return []
    texts = []
    for block in content:
        if isinstance(block, dict):
            t = block.get('text')
            if t:
                texts.append(t)
    return texts


def extract_multimodal_answer_text(response) -> str:
    """Return answer text: last non-empty segment (answer after reasoning when thinking is on)."""
    segs = _multimodal_response_text_segments(response)
    return segs[-1] if segs else ''


def extract_multimodal_stream_text(chunks, incremental_output: bool) -> str:
    """Aggregate streaming chunks: deltas concatenated if incremental, else last snapshot."""
    per_chunk = []
    for resp in chunks:
        segs = _multimodal_response_text_segments(resp)
        if segs:
            per_chunk.append(''.join(segs))
    if not per_chunk:
        return ''
    if incremental_output:
        return ''.join(per_chunk)
    return per_chunk[-1]


class Qwen2VLAPI(Qwen2VLPromptMixin, BaseAPI):
    is_api: bool = True

    def __init__(
        self,
        model: str = 'qwen-vl-max-0809',
        key: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_length=1024,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        seed=3407,
        use_custom_prompt: bool = True,
        **kwargs,
    ):
        import dashscope

        self.model = model
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_length=max_length,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
        )

        key = os.environ.get('DASHSCOPE_API_KEY', None) if key is None else key
        assert key is not None, (
            'Please set the API Key (obtain it here: '
            'https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start)'
        )
        dashscope.api_key = key
        super().__init__(use_custom_prompt=use_custom_prompt, **kwargs)

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(self, inputs, **kwargs) -> str:
        import dashscope

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append(
            {'role': 'user', 'content': self._prepare_content(inputs, dataset=kwargs.get('dataset', None))}
        )
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        # generate
        generation_kwargs = self.generate_kwargs.copy()
        kwargs.pop('dataset', None)
        generation_kwargs.update(kwargs)
        try:
            response = dashscope.MultiModalConversation.call(
                model=self.model,
                messages=messages,
                **generation_kwargs,
            )
            if self.verbose:
                print(response)
            answer = response.output.choices[0]['message']['content'][0]['text']
            return 0, answer, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                logger.error(f'{type(err)}: {err}')
                logger.error(f'The input messages are {inputs}.')
            return -1, '', ''


class QwenVLDashScopeVideoAPI(BaseAPI):
    """Qwen VL API via DashScope SDK with native video support.

    Uses dashscope.MultiModalConversation.call() to send video files
    directly (via file:// URI), controlled by fps and max_frames.

    Pass enable_thinking / stream / incremental_output to match DashScope
    MultiModalConversation (e.g. qwen3.5-plus extended thinking). When
    stream is True, chunks are aggregated; incremental_output=True joins
    token deltas, False uses the last chunk snapshot.

    ``video_llm`` (default False) sets ``VIDEO_LLM`` for the dataloader: pass
    ``run.py --video-llm`` so ``build_prompt(..., video_llm=True)`` sends one
    native video; otherwise frames are expanded to many images.
    """

    is_api: bool = True

    def __init__(
        self,
        model: str = 'qwen-vl-max-latest',
        key: str | None = None,
        max_length: int = 8192,
        temperature: float = 0.01,
        top_p: float = 0.001,
        top_k: int = 1,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        seed: int = 3407,
        fps: float = 2.0,
        max_frames: int | None = None,
        video_llm: bool = False,
        enable_thinking: bool = False,
        stream: bool = False,
        incremental_output: bool | None = None,
        use_custom_prompt: bool = True,
        **kwargs,
    ):
        import dashscope

        self.VIDEO_LLM = video_llm
        self.model = model
        self.fps = fps
        self.max_frames = max_frames
        self.enable_thinking = enable_thinking
        _think = os.environ.get('DASHSCOPE_ENABLE_THINKING', '').strip()
        if _think == '1':
            self.enable_thinking = True
        elif _think == '0':
            self.enable_thinking = False
        self.stream = stream
        self.incremental_output = incremental_output
        self.generate_kwargs = dict(
            max_length=max_length,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
        )

        key = os.environ.get('DASHSCOPE_API_KEY', None) if key is None else key
        assert key is not None, (
            'Please set the environment variable DASHSCOPE_API_KEY '
            '(obtain it here: https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start)'
        )
        dashscope.api_key = key
        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
        super().__init__(use_custom_prompt=use_custom_prompt, **kwargs)

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
            elif s['type'] == 'video':
                video_path = ensure_video_url(s['value'])
                item = {'type': 'video', 'video': video_path, 'fps': self.fps}
                if self.max_frames is not None:
                    item['max_frames'] = self.max_frames
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(self, inputs, **kwargs) -> str:
        import dashscope

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append(
            {'role': 'user', 'content': self._prepare_content(inputs, dataset=kwargs.get('dataset', None))}
        )
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        generation_kwargs = self.generate_kwargs.copy()
        kwargs.pop('dataset', None)
        generation_kwargs.update(kwargs)
        generation_kwargs.setdefault('enable_thinking', self.enable_thinking)
        generation_kwargs.setdefault('stream', self.stream)
        if self.incremental_output is not None:
            generation_kwargs.setdefault('incremental_output', self.incremental_output)
        try:
            response = dashscope.MultiModalConversation.call(
                model=self.model,
                messages=messages,
                **generation_kwargs,
            )
            use_stream = generation_kwargs.get('stream', False)
            if use_stream:
                chunk_list = []
                for chunk in response:
                    chunk_list.append(chunk)
                    if self.verbose:
                        print(chunk)
                incr = generation_kwargs.get('incremental_output', True)
                answer = extract_multimodal_stream_text(chunk_list, incr)
            else:
                if self.verbose:
                    print(response)
                answer = extract_multimodal_answer_text(response)
            if not answer:
                return -1, '', 'Empty model output. '
            return 0, answer, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                logger.error(f'{type(err)}: {err}')
                logger.error(f'The input messages are {inputs}.')
            return -1, '', ''


class QwenVLWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'qwen-vl-plus',
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 temperature: float = 0.0,
                 system_prompt: str = None,
                 max_tokens: int = 2048,
                 proxy: str = None,
                 **kwargs):

        assert model in ['qwen-vl-plus', 'qwen-vl-max']
        self.model = model
        import dashscope
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        if key is None:
            key = os.environ.get('DASHSCOPE_API_KEY', None)
        assert key is not None, (
            'Please set the API Key (obtain it here: '
            'https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start)'
        )
        dashscope.api_key = key
        if proxy is not None:
            proxy_set(proxy)
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(text=msg['value']))
                elif msg['type'] == 'image':
                    content_list.append(dict(image='file://' + msg['value']))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        from dashscope import MultiModalConversation

        assert isinstance(inputs, str) or isinstance(inputs, list)

        if 'type' in inputs[0]:
            pure_text = np.all([x['type'] == 'text' for x in inputs])
        else:
            pure_text = True
            for inp in inputs:
                if not np.all([x['type'] == 'text' for x in inp['content']]):
                    pure_text = False
                    break

        assert not pure_text
        messages = self.prepare_inputs(inputs)
        gen_config = dict(max_output_tokens=self.max_tokens, temperature=self.temperature)
        gen_config.update(kwargs)
        try:
            response = MultiModalConversation.call(model=self.model, messages=messages)
            if self.verbose:
                print(response)
            answer = response.output.choices[0]['message']['content'][0]['text']
            return 0, answer, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                logger.error(f'{type(err)}: {err}')
                logger.error(f'The input messages are {inputs}.')

            return -1, '', ''


class QwenVLAPI(QwenVLWrapper):

    def generate(self, message, dataset=None):
        return super(QwenVLAPI, self).generate(message)

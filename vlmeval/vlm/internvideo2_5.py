import warnings

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from .base import BaseModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _get_num_frames_by_duration(duration):
    """Official InternVideo2.5 dynamic frame sampling based on video duration."""
    local_num_frames = 4
    num_segments = int(duration // local_num_frames)
    if num_segments == 0:
        num_frames = local_num_frames
    else:
        num_frames = local_num_frames * num_segments
    num_frames = min(512, num_frames)
    num_frames = max(128, num_frames)
    return num_frames


def _load_video(video_path, max_num=1, num_segments=None, input_size=448, fps=None):
    """Load video frames.

    Sampling strategy (by priority):
    1. If *fps* > 0: sample at that rate, capped to *num_segments* (or 512).
    2. If *num_segments* is given (> 0): uniform sampling with that many frames.
    3. Otherwise: use official duration-based dynamic sampling (128~512 frames).
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    avg_fps = float(vr.get_avg_fps())
    duration = total_frames / avg_fps

    if fps is not None and fps > 0:
        n_samples = max(1, int(duration * fps))
        cap = num_segments if (num_segments and num_segments > 0) else 512
        n_samples = min(n_samples, cap)
    elif num_segments is not None and num_segments > 0:
        n_samples = num_segments
    else:
        n_samples = _get_num_frames_by_duration(duration)

    seg_size = float(total_frames) / n_samples
    frame_indices = np.array([
        int(seg_size / 2 + seg_size * idx) for idx in range(n_samples)
    ])
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)

    transform = _build_transform(input_size)
    pixel_values_list, num_patches_list = [], []
    for idx in frame_indices:
        img = Image.fromarray(vr[idx].asnumpy()).convert('RGB')
        tiles = _dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pv = torch.stack([transform(t) for t in tiles])
        num_patches_list.append(pv.shape[0])
        pixel_values_list.append(pv)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


class InternVideo2_5(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False
    VIDEO_LLM = True

    def __init__(self, model_path, **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.nframe = 0
        self.fps = 0

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True
        ).half().cuda().to(torch.bfloat16)
        self.model.eval()

        self.device = 'cuda'

        kwargs_default = dict(do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=0.1, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'InternVideo2_5 generation config: {self.kwargs}')

    def generate_inner(self, message, dataset=None):
        video_path = None
        prompt_parts = []

        for item in message:
            if item['type'] == 'video':
                video_path = item['value']
            elif item['type'] == 'text':
                prompt_parts.append(item['value'])

        if video_path is None:
            image_paths = [item['value'] for item in message if item['type'] == 'image']
            if not image_paths:
                return 'Error: no video or image input found.'
            return self._generate_from_images(image_paths, '\n'.join(prompt_parts))

        return self._generate_from_video(video_path, '\n'.join(prompt_parts), message)

    def _generate_from_video(self, video_path, question, message=None):
        fps_to_use = None
        num_segments = None

        if message is not None:
            for item in message:
                if item['type'] == 'video':
                    if 'fps' in item and float(item['fps']) > 0:
                        fps_to_use = float(item['fps'])
                    if 'nframes' in item and int(item['nframes']) > 0:
                        num_segments = int(item['nframes'])
                    break

        pixel_values, num_patches_list = _load_video(
            video_path, max_num=1, num_segments=num_segments, fps=fps_to_use
        )
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)

        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        full_question = video_prefix + question

        with torch.no_grad():
            response, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                full_question,
                self.kwargs,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
        return response

    def _generate_from_images(self, image_paths, question):
        transform = _build_transform(input_size=448)
        pixel_values_list, num_patches_list = [], []
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            tiles = _dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=1)
            pv = torch.stack([transform(t) for t in tiles])
            num_patches_list.append(pv.shape[0])
            pixel_values_list.append(pv)

        pixel_values = torch.cat(pixel_values_list).to(torch.bfloat16).to(self.device)
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        full_question = video_prefix + question

        with torch.no_grad():
            response, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                full_question,
                self.kwargs,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
        return response

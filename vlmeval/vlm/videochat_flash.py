import warnings

import importlib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .base import BaseModel

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = '<image>'
QWEN_SYSTEM_PROMPT = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'


class VideoChatFlash(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False
    VIDEO_LLM = True

    def __init__(
        self,
        model_path,
        max_num_frames=512,
        mm_llm_compress=False,
        llm_compress_type='uniform0_attention',
        llm_compress_layer_list=None,
        llm_image_token_ratio_list=None,
        generation_config=None,
        **kwargs,
    ):
        assert model_path is not None
        self.model_path = model_path
        self.max_num_frames = max_num_frames

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
        self.model.eval()

        if mm_llm_compress:
            self.model.config.mm_llm_compress = True
            self.model.config.llm_compress_type = llm_compress_type
            self.model.config.llm_compress_layer_list = llm_compress_layer_list or [4, 18]
            self.model.config.llm_image_token_ratio_list = llm_image_token_ratio_list or [1, 0.75, 0.25]
        else:
            self.model.config.mm_llm_compress = False

        default_generation_config = dict(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            top_p=0.1,
            num_beams=1,
        )
        if generation_config is not None:
            default_generation_config.update(generation_config)
        default_generation_config.update(kwargs)
        self.generation_config = default_generation_config

        warnings.warn(f'VideoChatFlash generation config: {self.generation_config}')

    @staticmethod
    def _tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

        def insert_separator(chunks, sep):
            return [ele for sublist in zip(chunks, [sep] * len(chunks)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for chunk in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(chunk[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    @staticmethod
    def _build_chat_prompt(user_prompt):
        return (
            QWEN_SYSTEM_PROMPT
            + f'<|im_start|>user\n{user_prompt}<|im_end|>\n'
            + '<|im_start|>assistant\n'
        )

    def _sample_video_frames(self, video_path, num_frames):
        VideoReader = importlib.import_module('decord').VideoReader
        vr = VideoReader(video_path, num_threads=1)
        total_frames = len(vr)
        if total_frames == 0:
            raise ValueError(f'Video {video_path} contains no frames.')

        num_frames = max(1, min(int(num_frames), total_frames))
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        return vr.get_batch(indices).asnumpy()

    def _allocate_frame_budgets(self, video_paths):
        stats = []
        raw_targets = []

        for path in video_paths:
            VideoReader = importlib.import_module('decord').VideoReader
            vr = VideoReader(path, num_threads=1)
            total_frames = len(vr)
            if total_frames == 0:
                raise ValueError(f'Video {path} contains no frames.')

            avg_fps = float(vr.get_avg_fps())
            if avg_fps > 0:
                duration = total_frames / avg_fps
                num_segments = int(duration // 8)
                target = 8 if num_segments == 0 else 8 * num_segments
                target = max(64, min(self.max_num_frames, target))
            else:
                target = min(self.max_num_frames, total_frames)

            target = max(1, min(target, total_frames))
            stats.append((path, total_frames))
            raw_targets.append(target)

        total_target = sum(raw_targets)
        if total_target <= self.max_num_frames:
            return raw_targets

        scaled = [self.max_num_frames * target / total_target for target in raw_targets]
        budgets = [max(1, int(np.floor(x))) for x in scaled]
        remainder = self.max_num_frames - sum(budgets)

        if remainder > 0:
            order = sorted(
                range(len(scaled)),
                key=lambda idx: scaled[idx] - budgets[idx],
                reverse=True,
            )
            for idx in order[:remainder]:
                budgets[idx] += 1
        elif remainder < 0:
            order = sorted(
                range(len(scaled)),
                key=lambda idx: scaled[idx] - budgets[idx],
            )
            for idx in order:
                if remainder == 0:
                    break
                if budgets[idx] > 1:
                    budgets[idx] -= 1
                    remainder += 1

        for idx, (_, total_frames) in enumerate(stats):
            budgets[idx] = max(1, min(budgets[idx], total_frames))

        return budgets

    def _generate_from_image_sequence(self, message):
        video_paths = [item['value'] for item in message if item['type'] == 'video']
        frame_budgets = self._allocate_frame_budgets(video_paths)
        image_processor = self.model.get_vision_tower().image_processor

        images = []
        prompt_parts = []
        image_sizes = []
        video_idx = 0

        for item in message:
            if item['type'] == 'text':
                prompt_parts.append(item['value'])
                continue

            if item['type'] != 'video':
                continue

            frames = self._sample_video_frames(item['value'], frame_budgets[video_idx])
            prompt_parts.append((DEFAULT_IMAGE_TOKEN + '\n') * len(frames))
            for frame in frames:
                image_sizes.append((frame.shape[1], frame.shape[0]))
                pixel_values = image_processor.preprocess(frame, return_tensors='pt')['pixel_values'][0]
                images.append(pixel_values.to(dtype=self.model.dtype, device='cuda', non_blocking=True))
            video_idx += 1

        prompt = self._build_chat_prompt(''.join(prompt_parts).strip())
        input_ids = self._tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt',
        ).unsqueeze(0).cuda()

        if self.tokenizer.pad_token_id is None and 'qwen' in self.tokenizer.name_or_path.lower():
            self.tokenizer.pad_token_id = 151643
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()

        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=images,
                attention_mask=attention_mask,
                modalities=['image'] * len(images),
                image_sizes=image_sizes,
                use_cache=True,
                **self.generation_config,
            )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def generate_inner(self, message, dataset=None):
        video_paths = [item['value'] for item in message if item['type'] == 'video']
        prompt = '\n'.join(item['value'] for item in message if item['type'] == 'text').strip()

        if not video_paths:
            return 'Error: no video input found.'
        if len(video_paths) > 1:
            return self._generate_from_image_sequence(message)

        with torch.no_grad():
            response, _ = self.model.chat(
                video_path=video_paths[0],
                tokenizer=self.tokenizer,
                user_prompt=prompt,
                return_history=True,
                max_num_frames=self.max_num_frames,
                generation_config=self.generation_config,
            )
        return response

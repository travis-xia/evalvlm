import warnings

import torch
from transformers import AutoModel, AutoTokenizer

from .base import BaseModel


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

    def generate_inner(self, message, dataset=None):
        video_paths = [item['value'] for item in message if item['type'] == 'video']
        prompt = '\n'.join(item['value'] for item in message if item['type'] == 'text').strip()

        if not video_paths:
            return 'Error: no video input found.'
        if len(video_paths) > 1:
            return 'Error: VideoChat-Flash only supports a single input video per request.'

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

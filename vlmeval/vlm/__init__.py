import importlib

import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)

# (submodule under vlmeval.vlm, names re-exported to this package)
_SUBMODULE_EXPORTS = (
    ('aki', ('AKI',)),
    ('aria', ('Aria',)),
    ('bagel_umm', ('Bagel',)),
    ('base', ('BaseModel',)),
    ('bunnyllama3', ('BunnyLLama3',)),
    ('cambrian', ('Cambrian',)),
    ('cambrian_s', ('CambrianS',)),
    ('chameleon', ('Chameleon',)),
    ('cogvlm', ('CogVlm', 'GLM4v', 'GLMThinking')),
    ('cosmos', ('Cosmos',)),
    ('covt', ('CoVTChat',)),
    ('deepseek_ocr', ('DeepSeekOCR',)),
    ('deepseek_vl', ('DeepSeekVL',)),
    ('deepseek_vl2', ('DeepSeekVL2',)),
    ('eagle_x', ('Eagle',)),
    ('emu', ('Emu', 'Emu3_chat', 'Emu3_gen')),
    ('falcon_vlm', ('Falcon2VLM',)),
    ('flash_vl', ('FlashVL',)),
    ('videochat_flash', ('VideoChatFlash',)),
    ('gemma', ('Gemma3', 'PaliGemma')),
    ('granite_docling', ('DOCLING',)),
    ('granite_vision', ('GraniteVision3',)),
    ('h2ovl_mississippi', ('H2OVLChat',)),
    ('hawk_vl', ('HawkVL',)),
    ('idefics', ('IDEFICS', 'IDEFICS2')),
    ('insight_v', ('InsightV',)),
    ('instructblip', ('InstructBLIP',)),
    ('interns1', ('InternS1Chat',)),
    ('internvideo2_5', ('InternVideo2_5',)),
    ('internvl', ('InternVLChat',)),
    ('janus', ('Janus',)),
    ('keye_vlm', ('KeyeChat',)),
    ('kimi_vl', ('KimiVL',)),
    ('kosmos', ('Kosmos2',)),
    ('liquid', ('LFM2VL',)),
    ('llama4', ('llama4',)),
    ('llama_vision', ('llama_vision',)),
    (
        'llava',
        (
            'LLaVA', 'LLaVA_Next', 'LLaVA_Next2', 'LLaVA_OneVision', 'LLaVA_OneVision_1_5',
            'LLaVA_OneVision_HF', 'LLaVA_XTuner',
        ),
    ),
    ('logics', ('Logics_Thinking',)),
    ('long_vita', ('LongVITA',)),
    ('mantis', ('Mantis',)),
    ('mgm', ('Mini_Gemini',)),
    (
        'minicpm_v',
        (
            'MiniCPM_Llama3_V', 'MiniCPM_o_2_6', 'MiniCPM_o_4_5', 'MiniCPM_V', 'MiniCPM_V_2_6',
            'MiniCPM_V_4', 'MiniCPM_V_4_5',
        ),
    ),
    ('minigpt4', ('MiniGPT4',)),
    ('minimonkey', ('MiniMonkey',)),
    ('mixsense', ('LLama3Mixsense',)),
    ('mmalaya', ('MMAlaya', 'MMAlaya2')),
    ('molmo', ('molmo',)),
    ('monkey', ('Monkey', 'MonkeyChat')),
    ('moondream', ('Moondream1', 'Moondream2', 'Moondream3')),
    ('mplug_owl2', ('mPLUG_Owl2',)),
    ('mplug_owl3', ('mPLUG_Owl3',)),
    ('nvlm', ('NVLM',)),
    ('ola', ('Ola',)),
    ('omchat', ('OmChat',)),
    ('omnilmm', ('OmniLMM12B',)),
    ('open_flamingo', ('OpenFlamingo',)),
    ('oryx', ('Oryx',)),
    ('ovis', ('Ovis', 'Ovis1_6', 'Ovis1_6_Plus', 'Ovis2', 'Ovis2_5', 'OvisU1')),
    ('pandagpt', ('PandaGPT',)),
    ('parrot', ('Parrot',)),
    ('phi3_vision', ('Phi3_5Vision', 'Phi3Vision')),
    ('phi4_multimodal', ('Phi4Multimodal',)),
    ('pixtral', ('Pixtral',)),
    ('points', ('POINTS', 'POINTSV15')),
    ('qh_360vl', ('QH_360VL',)),
    ('qianfan_vl', ('Qianfan_VL',)),
    ('qtunevl', ('QTuneVL', 'QTuneVLChat')),
    ('qwen2_vl', ('Qwen2VLChat', 'Qwen2VLChatAguvis')),
    ('qwen3_vl', ('Qwen3VLChat',)),
    ('qwen_vl', ('QwenVL', 'QwenVLChat')),
    ('rbdash', ('RBDash',)),
    ('ristretto', ('Ristretto',)),
    ('ross', ('Ross',)),
    ('sail_vl', ('SailVL',)),
    ('slime', ('SliME',)),
    ('smolvlm', ('SmolVLM', 'SmolVLM2')),
    ('spatial_mllm', ('SpatialMLLM',)),
    ('thyme', ('Thyme',)),
    ('transcore_m', ('TransCoreM',)),
    ('treevgr', ('TreeVGR',)),
    ('ursa', ('UrsaChat',)),
    ('valley', ('Valley2Chat', 'Valley3Chat')),
    ('varco_vision', ('VarcoVision',)),
    (
        'video_llm',
        (
            'Chatunivi', 'LLaMAVID', 'PLLaVA', 'VideoChat2_HD', 'VideoChatGPT', 'VideoLLaVA',
            'VideoLLaVA_HF',
        ),
    ),
    ('vila', ('NVILA', 'VILA')),
    ('vintern_chat', ('VinternChat',)),
    ('visualglm', ('VisualGLM',)),
    ('vita', ('VITA', 'VITAQwen2')),
    ('vlaa_thinker', ('VLAAThinkerChat',)),
    ('vlm3r', ('VLM3R',)),
    ('vlm_r1', ('VLMR1Chat',)),
    ('vxverse', ('VXVERSE',)),
    ('wemm', ('WeMM',)),
    ('wethink_vl', ('WeThinkVL',)),
    ('x_vl', ('X_VL_HF',)),
    ('xcomposer', ('ShareCaptioner', 'XComposer', 'XComposer2', 'XComposer2_4KHD', 'XComposer2d5')),
    ('xgen_mm', ('XGenMM',)),
    ('yi_vl', ('Yi_VL',)),
)


def _load_submodules():
    pkg = __name__
    for submod, names in _SUBMODULE_EXPORTS:
        try:
            mod = importlib.import_module(f'.{submod}', pkg)
        except ImportError:
            for n in names:
                globals()[n] = None
            continue
        for n in names:
            globals()[n] = getattr(mod, n, None)


_load_submodules()
del _load_submodules, _SUBMODULE_EXPORTS

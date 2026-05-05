"""Aggregated API wrappers. Submodules may be absent on minimal installs; missing ones become None."""
import importlib

# (submodule stem under this package, attribute names to re-export)
_SUBMODULE_EXPORTS = (
    ('arm_thinker', ('ARM_thinker',)),
    ('bailingmm', ('bailingMMAPI',)),
    ('bedrock', ('BedrockAPI',)),
    ('bluelm_api', ('BlueLM_API', 'BlueLMWrapper')),
    ('claude', ('Claude3V', 'Claude_Wrapper')),
    ('cloudwalk', ('CWWrapper',)),
    ('doubao_vl_api', ('DoubaoVL',)),
    ('gcp_vertex', ('GCPVertexAPI',)),
    ('gemini', ('Gemini', 'GeminiWrapper')),
    ('glm_vision', ('GLMVisionAPI',)),
    ('gpt', ('GPT4V', 'OpenAIWrapper')),
    ('hf_chat_model', ('HFChatModel',)),
    ('hunyuan', ('HunyuanVision',)),
    ('jt_vl_chat', ('JTVLChatAPI',)),
    ('jt_vl_chat_mini', ('JTVLChatAPI_2B', 'JTVLChatAPI_Mini')),
    ('kimivl_api', ('KimiVLAPI', 'KimiVLAPIWrapper')),
    ('lmdeploy', ('LMDeployAPI', 'LMDeployWrapper')),
    ('minimax_api', ('MiniMaxAPI',)),
    ('mug_u', ('MUGUAPI',)),
    ('openai_sdk', ('OpenAISDKWrapper',)),
    ('qwen_api', ('QwenAPI',)),
    ('qwen_vl_api', ('Qwen2VLAPI', 'QwenVLAPI', 'QwenVLWrapper', 'QwenVLDashScopeVideoAPI')),
    ('rbdashmm_chat3_5_api', ('RBdashMMChat3_5_38B_API', 'RBdashMMChat3_78B_API')),
    ('rbdashmm_chat3_api', ('RBdashChat3_5_API', 'RBdashMMChat3_API')),
    ('reka', ('Reka',)),
    ('sensechat_vision', ('SenseChatVisionAPI', 'SenseChatVisionV2API')),
    ('siliconflow', ('SiliconFlowAPI', 'TeleMMAPI')),
    ('taichu', ('TaichuVLAPI', 'TaichuVLRAPI')),
    ('taiyi', ('TaiyiAPI',)),
    ('telemm', ('TeleMM2_API',)),
    ('telemm_thinking', ('TeleMM2Thinking_API',)),
    ('together', ('TogetherAPI',)),
    ('video_chat_online_v2', ('VideoChatOnlineV2API',)),
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

__all__ = [
    'OpenAIWrapper', 'HFChatModel', 'GeminiWrapper', 'GPT4V', 'Gemini', 'QwenVLWrapper',
    'QwenVLAPI', 'QwenAPI', 'Claude3V', 'Claude_Wrapper', 'Reka', 'GLMVisionAPI', 'CWWrapper',
    'SenseChatVisionAPI', 'HunyuanVision', 'Qwen2VLAPI', 'BlueLMWrapper', 'BlueLM_API',
    'JTVLChatAPI', 'JTVLChatAPI_Mini', 'JTVLChatAPI_2B', 'bailingMMAPI', 'TaiyiAPI', 'TeleMMAPI',
    'SiliconFlowAPI', 'LMDeployAPI', 'ARM_thinker', 'OpenAISDKWrapper', 'LMDeployWrapper',
    'TaichuVLAPI', 'TaichuVLRAPI', 'DoubaoVL', 'MUGUAPI', 'KimiVLAPIWrapper', 'KimiVLAPI',
    'RBdashMMChat3_API', 'RBdashChat3_5_API', 'RBdashMMChat3_78B_API', 'RBdashMMChat3_5_38B_API',
    'VideoChatOnlineV2API', 'TeleMM2_API', 'TeleMM2Thinking_API', 'TogetherAPI', 'GCPVertexAPI',
    'BedrockAPI', 'SenseChatVisionV2API', 'MiniMaxAPI', 'QwenVLDashScopeVideoAPI',
]

import ast
import json
import os
import os.path as osp
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load
from .video_base import VideoBaseDataset


IOU_THRESHOLDS = [0.3, 0.5, 0.7]

# TransVideoBench local data config.
# You can edit these defaults directly, or override them with env vars:
#   TRANSVIDEOBENCH_ROOT
#   TRANSVIDEOBENCH_MCQ_FILE
#   TRANSVIDEOBENCH_TG_FILE
#   TRANSVIDEOBENCH_MV_FILE
#   TRANSVIDEOBENCH_MCQ_VIDEO_DIR
#   TRANSVIDEOBENCH_TG_VIDEO_DIR
#   TRANSVIDEOBENCH_MV_VIDEO_DIR
#   TRANSVIDEOBENCH_ENABLE_FRAME_TIMESTAMPS
TRANSVIDEOBENCH_ROOT = '/inspire/ssd/project/traffic-congestion-management/public/bench-v1'
TRANSVIDEOBENCH_MCQ_FILE = 'results-en_en.merged.json'
TRANSVIDEOBENCH_TG_FILE = 'results-tg-en_en.merged.json'
TRANSVIDEOBENCH_MV_FILE = 'results-mv-en_en.json'
TRANSVIDEOBENCH_MCQ_VIDEO_DIR = 'videos-mask'
TRANSVIDEOBENCH_TG_VIDEO_DIR = 'videos-tg-mask'
TRANSVIDEOBENCH_MV_VIDEO_DIR = 'videos-mv-mask'
TRANSVIDEOBENCH_ENABLE_FRAME_TIMESTAMPS = False

MCQ_PROMPT = """These are the frames of a traffic surveillance video.
Select the best answer to the following multiple-choice question based on the video.
Respond with only the letter (A, B, C, or D) of the correct option.

Question: {question}
{options}
Answer: """

TG_PROMPT = """These are the frames of a traffic surveillance video.
Answer the temporal grounding question based on the video.
Return the time interval in seconds using the format <time>start - end seconds</time>.

Question: {question}
Answer: """

MV_PROMPT = """You are given two traffic surveillance videos from different camera views of related road sections.
The first video is the upstream view, and the second video is the downstream view.
Answer the following multiple-choice question based on both videos.
Respond with only the letter (A, B, C, or D) of the correct option.

Question: {question}
{options}
Answer: """


def _safe_json_load(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get('test'), list):
        return data['test']
    if isinstance(data, list):
        return data
    raise ValueError(f'Unsupported TransVideoBench annotation format: {path}')


def _literal_eval(value, default=None):
    if isinstance(value, (dict, list)):
        return value
    if pd.isna(value):
        return default
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return default


def _strip_video_suffix(video):
    video = str(video)
    return video[:-4] if video.endswith('.mp4') else video


def _duration_bucket(duration):
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        return 'unknown'
    if np.isnan(duration):
        return 'unknown'
    return '0-30s' if duration <= 30 else '30s以上'


def _video_duration(path):
    if not osp.exists(path):
        return np.nan
    try:
        import decord
        vr = decord.VideoReader(path)
        fps = vr.get_avg_fps()
        return len(vr) / fps if fps else np.nan
    except Exception:
        return np.nan


def _format_options(options):
    if not isinstance(options, dict):
        return ''
    return '\n'.join(f'{k}. {v}' for k, v in sorted(options.items()))


def _format_tg_answer(label):
    if isinstance(label, dict):
        start = label.get('start')
        end = label.get('end')
        return f'<time>{start} - {end} seconds</time>'
    return str(label)


def _extract_time(paragraph):
    paragraph = str(paragraph).lower().replace('to', '-')
    timestamps = []

    tagged_spans = re.findall(r'<time>(.*?)</time>', paragraph, flags=re.DOTALL)
    search_space = tagged_spans if tagged_spans else [paragraph]

    time_regex = re.compile(r'\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b')
    for text in search_space:
        time_matches = re.findall(time_regex, text)
        time_matches = time_matches[:len(time_matches) // 2 * 2]
        if time_matches:
            converted = []
            for time_str in time_matches:
                parts = time_str.split(':')
                if len(parts) == 3:
                    h, m = map(int, parts[:2])
                    seconds = h * 3600 + m * 60 + float(parts[2])
                else:
                    m = int(parts[0])
                    seconds = m * 60 + float(parts[1])
                converted.append(float(seconds))
            timestamps = [(converted[i], converted[i + 1]) for i in range(0, len(converted), 2)]
            break

    if not timestamps:
        patterns = [
            r'(\d+\.?\d*)\s*(?:seconds?|secs?|s)?\s*-\s*(\d+\.?\d*)\s*(?:seconds?|secs?|s)?',
            r'from\s*(\d+\.?\d*)\s*(?:seconds?|secs?|s)?\s*-\s*(\d+\.?\d*)\s*(?:seconds?|secs?|s)?',
            r'from\s*(\d+\.?\d*)\s*(?:seconds?|secs?|s)?\s*until\s*(\d+\.?\d*)\s*(?:seconds?|secs?|s)?',
        ]
        for text in search_space:
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    timestamps = [(float(start), float(end)) for start, end in matches]
                    break
            if timestamps:
                break

    return timestamps


def _temporal_iou(gt_span, pred_span):
    max_start = max(gt_span[0], pred_span[0])
    min_start = min(gt_span[0], pred_span[0])
    max_end = max(gt_span[1], pred_span[1])
    min_end = min(gt_span[1], pred_span[1])
    denom = max_end - min_start
    if denom <= 0:
        return 0.0
    return max(min_end - max_start, 0) / denom


def _score_timelens_tg(prediction, answer):
    pred_segments = _extract_time(prediction)
    gt_segments = _extract_time(answer)
    if not pred_segments or not gt_segments:
        iou = 0.0
    else:
        iou = _temporal_iou(gt_segments[0], pred_segments[0])
    metrics = {'mIoU': iou}
    for th in IOU_THRESHOLDS:
        metrics[f'R@{th}'] = float(iou >= th)
    return metrics


def _extract_mcq_answer(prediction, options):
    def normalize_text(text):
        text = str(text).strip()
        return text.replace('（', '(').replace('）', ')').replace('．', '.')

    def extract_letter(text, prefer_last=False):
        matches = re.findall(r'(?<![A-Za-z])[ABCD](?![A-Za-z])', text)
        if matches:
            return matches[-1] if prefer_last else matches[0]
        return ''

    def extract_option_text(text, prefer_last=False):
        if not isinstance(options, dict):
            return ''
        matches = []
        for key, option in options.items():
            option = str(option).strip()
            if option and option in text:
                matches.append(key)
        if matches:
            return matches[-1] if prefer_last else matches[0]
        return ''

    def extract_from_candidate(text, prefer_last=False):
        text = normalize_text(text)
        if not text:
            return ''
        answer = extract_letter(text, prefer_last=prefer_last)
        if answer:
            return answer
        return extract_option_text(text, prefer_last=prefer_last)

    text = normalize_text(prediction)

    answer_blocks = re.findall(r'<answer>(.*?)</answer>', text, flags=re.IGNORECASE | re.DOTALL)
    for block in reversed(answer_blocks):
        answer = extract_from_candidate(block)
        if answer:
            return answer

    if '</think>' in text:
        answer = extract_from_candidate(text.split('</think>')[-1])
        if answer:
            return answer

    direct_patterns = [
        r'(?:the final answer is|the correct answer is|the answer is|the correct option is|'
        r'the best answer is|the best option is)\s*(?:option\s*)?[\(\[]?\s*([ABCD])(?:[\)\].:：\s]|$)',
        r'(?:final answer|correct answer|answer|correct option|best answer|best option)\s*[:：]\s*'
        r'(?:option\s*)?[\(\[]?\s*([ABCD])(?:[\)\].:：\s]|$)',
        r'(?:最终答案是|正确答案是|答案是|答案为|选项为|选择)\s*[:：]?\s*[\(\[]?\s*([ABCD])(?:[\)\].:：\s]|$)',
    ]
    for pattern in direct_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-5:]):
        answer = extract_from_candidate(line)
        if answer:
            return answer

    text_wo_think = re.sub(r'<think>.*?</think>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text_wo_think = re.sub(r'<thinking>.*?</thinking>', ' ', text_wo_think, flags=re.IGNORECASE | re.DOTALL)
    answer = extract_from_candidate(text_wo_think, prefer_last=True)
    if answer:
        return answer

    return extract_from_candidate(text, prefer_last=True)


def _get_transvideobench_config():
    config = dict(
        root=os.environ.get('TRANSVIDEOBENCH_ROOT', TRANSVIDEOBENCH_ROOT),
        mcq_file=os.environ.get('TRANSVIDEOBENCH_MCQ_FILE', TRANSVIDEOBENCH_MCQ_FILE),
        tg_file=os.environ.get('TRANSVIDEOBENCH_TG_FILE', TRANSVIDEOBENCH_TG_FILE),
        mv_file=os.environ.get('TRANSVIDEOBENCH_MV_FILE', TRANSVIDEOBENCH_MV_FILE),
        mcq_video_dir=os.environ.get('TRANSVIDEOBENCH_MCQ_VIDEO_DIR', TRANSVIDEOBENCH_MCQ_VIDEO_DIR),
        tg_video_dir=os.environ.get('TRANSVIDEOBENCH_TG_VIDEO_DIR', TRANSVIDEOBENCH_TG_VIDEO_DIR),
        mv_video_dir=os.environ.get('TRANSVIDEOBENCH_MV_VIDEO_DIR', TRANSVIDEOBENCH_MV_VIDEO_DIR),
        enable_frame_timestamps=os.environ.get(
            'TRANSVIDEOBENCH_ENABLE_FRAME_TIMESTAMPS',
            str(TRANSVIDEOBENCH_ENABLE_FRAME_TIMESTAMPS),
        ).lower() in {'1', 'true', 'yes', 'on'},
    )
    return config


def _video_full_path(data_root, video):
    return osp.join(data_root, video + '.mp4')


def _safe_frame_key(video):
    key = _strip_video_suffix(video)
    key = key.replace('\\', '__').replace('/', '__').replace(':', '_')
    return key


def _format_frame_timestamp_text(timestamps):
    parts = [f'Frame-{idx + 1}={timestamp:.1f}s' for idx, timestamp in enumerate(timestamps)]
    return (
        "The frames are sampled in chronological order. "
        "Use these timestamps to map each frame to the original video time:\n" +
        ', '.join(parts) + '\n'
    )


class TransVideoBench(VideoBaseDataset):
    TYPE = 'Video-Mixed'
    IOUTHS = IOU_THRESHOLDS
    DATASET_NAMES = ['TransVideoBench', 'TransVideoBench_MCQ', 'TransVideoBench_TG', 'TransVideoBench_MV']

    def __init__(self, dataset='TransVideoBench', nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.transvideobench_config = _get_transvideobench_config()

    @classmethod
    def supported_datasets(cls):
        return cls.DATASET_NAMES

    def prepare_dataset(self, dataset_name='TransVideoBench'):
        data_root, config = self._find_data_root()
        data_file = osp.join(data_root, f'{dataset_name}.tsv')
        data = self._build_dataframe(data_root, dataset_name, config)
        data.to_csv(data_file, sep='\t', index=False)
        return dict(root=data_root, data_file=data_file)

    @staticmethod
    def _find_data_root():
        config = _get_transvideobench_config()
        candidates = [config['root'], osp.join(LMUDataRoot(), 'TransVideoBench')]
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(str(repo_root / 'trans_data_example'))

        for root in candidates:
            if not root:
                continue
            mcq_name = config['mcq_file'] if root == config['root'] else 'results_en.json'
            tg_name = config['tg_file'] if root == config['root'] else 'results-tg_en.json'
            mv_name = config['mv_file'] if root == config['root'] else 'results-mv_en.json'
            mcq_file = osp.join(root, mcq_name)
            tg_file = osp.join(root, tg_name)
            mv_file = osp.join(root, mv_name)
            if osp.exists(mcq_file) or osp.exists(tg_file) or osp.exists(mv_file):
                cfg = dict(config)
                if root != config['root']:
                    cfg.update(
                        mcq_file='results_en.json',
                        tg_file='results-tg_en.json',
                        mv_file='results-mv_en.json',
                        mcq_video_dir='videos',
                        tg_video_dir='videos-tg',
                        mv_video_dir='videos-mv',
                    )
                return root, cfg
        raise FileNotFoundError(
            'Cannot find TransVideoBench annotations. Please check TRANSVIDEOBENCH_ROOT / '
            'TRANSVIDEOBENCH_MCQ_FILE / TRANSVIDEOBENCH_TG_FILE / TRANSVIDEOBENCH_MV_FILE.'
        )

    @classmethod
    def _build_dataframe(cls, data_root, dataset_name, config):
        records = []
        if dataset_name not in ['TransVideoBench_TG', 'TransVideoBench_MV']:
            records.extend(cls._load_mcq_records(data_root, config))
        if dataset_name not in ['TransVideoBench_MCQ', 'TransVideoBench_MV']:
            records.extend(cls._load_tg_records(data_root, config))
        if dataset_name not in ['TransVideoBench_MCQ', 'TransVideoBench_TG']:
            records.extend(cls._load_mv_records(data_root, config))

        data = pd.DataFrame(records)
        if data.empty:
            raise ValueError(f'No samples found for {dataset_name} in {data_root}')
        data['index'] = np.arange(len(data))
        return data[[
            'index', 'id', 'video', 'video_path', 'question', 'answer', 'options',
            'task_type', 'static_dynamic', 'category', 'specific_interaction', 'video_paths',
            'duration', 'duration_bucket'
        ]]

    @classmethod
    def _load_mcq_records(cls, data_root, config):
        anno_path = osp.join(data_root, config['mcq_file'])
        if not osp.exists(anno_path):
            return []
        records = []
        for item in _safe_json_load(anno_path):
            video = cls._resolve_video(data_root, item, 'mcq', config)
            if video is None:
                warnings.warn(
                    f"Skip invalid MCQ sample id={item.get('id')} because no matching video file was found."
                )
                continue
            duration = cls._resolve_duration(data_root, video, item)
            records.append({
                'id': item['id'],
                'video': video,
                'video_path': video + '.mp4',
                'question': item['question'],
                'answer': item['label'],
                'options': json.dumps(item.get('options', {}), ensure_ascii=False),
                'task_type': 'mcq',
                'static_dynamic': item.get('static_dynamic', ''),
                'category': item.get('category', ''),
                'specific_interaction': item.get('specific_interaction', ''),
                'video_paths': json.dumps([video], ensure_ascii=False),
                'duration': duration,
                'duration_bucket': _duration_bucket(duration),
            })
        return records

    @classmethod
    def _load_tg_records(cls, data_root, config):
        anno_path = osp.join(data_root, config['tg_file'])
        if not osp.exists(anno_path):
            return []
        records = []
        for item in _safe_json_load(anno_path):
            video = cls._resolve_video(data_root, item, 'tg', config)
            if video is None:
                warnings.warn(
                    f"Skip invalid TG sample id={item.get('id')} because no matching video file was found."
                )
                continue
            duration = cls._resolve_duration(data_root, video, item)
            records.append({
                'id': item['id'],
                'video': video,
                'video_path': video + '.mp4',
                'question': item['question'],
                'answer': _format_tg_answer(item['label']),
                'options': '{}',
                'task_type': 'tg',
                'static_dynamic': item.get('static_dynamic', ''),
                'category': item.get('category', ''),
                'specific_interaction': item.get('specific_interaction', ''),
                'video_paths': json.dumps([video], ensure_ascii=False),
                'duration': duration,
                'duration_bucket': _duration_bucket(duration),
            })
        return records

    @classmethod
    def _load_mv_records(cls, data_root, config):
        anno_path = osp.join(data_root, config['mv_file'])
        if not osp.exists(anno_path):
            return []
        records = []
        for item in _safe_json_load(anno_path):
            video_pair = cls._resolve_mv_videos(data_root, item, config)
            if video_pair is None:
                warnings.warn(
                    f"Skip invalid MV sample id={item.get('id')} because one or more matching video files were not found."
                )
                continue
            upstream_video, downstream_video = video_pair
            duration = max(
                cls._resolve_duration(data_root, upstream_video, item),
                cls._resolve_duration(data_root, downstream_video, item),
            )
            records.append({
                'id': item['id'],
                'video': upstream_video,
                'video_path': upstream_video + '.mp4',
                'question': item['question'],
                'answer': item['label'],
                'options': json.dumps(item.get('options', {}), ensure_ascii=False),
                'task_type': 'mv',
                'static_dynamic': item.get('static_dynamic', ''),
                'category': item.get('category', ''),
                'specific_interaction': item.get('specific_interaction', ''),
                'video_paths': json.dumps([upstream_video, downstream_video], ensure_ascii=False),
                'duration': duration,
                'duration_bucket': _duration_bucket(duration),
            })
        return records

    @staticmethod
    def _resolve_video(data_root, item, task_type, config):
        if item.get('video_path'):
            video = _strip_video_suffix(item['video_path'])
            if osp.exists(_video_full_path(data_root, video)):
                return video
        if item.get('video'):
            video = _strip_video_suffix(item['video'])
            if osp.exists(_video_full_path(data_root, video)):
                return video

        sample_id = item['id']
        preferred_dir = config['mcq_video_dir'] if task_type == 'mcq' else config['tg_video_dir']
        task_dirs = [preferred_dir]
        if task_type == 'mcq':
            task_dirs.extend(['videos', 'videos-mcq', 'videos_mcq'])
        else:
            task_dirs.extend(['videos-tg', 'videos_tg', 'tg'])
        candidates = [f'{sample_id}.mp4']
        candidates.extend(f'{folder}/{sample_id}.mp4' for folder in task_dirs)
        candidates.extend(f'videos/{sample_id}.mp4' for _ in [0])

        for rel_path in candidates:
            if osp.exists(osp.join(data_root, rel_path)):
                return _strip_video_suffix(rel_path)
        return None

    @staticmethod
    def _resolve_mv_videos(data_root, item, config):
        upstream_video = item.get('upstream_video')
        downstream_video = item.get('downstream_video')
        if upstream_video is None or downstream_video is None:
            return None

        upstream_video = _strip_video_suffix(upstream_video)
        downstream_video = _strip_video_suffix(downstream_video)
        if osp.exists(_video_full_path(data_root, upstream_video)) and osp.exists(
            _video_full_path(data_root, downstream_video)
        ):
            return upstream_video, downstream_video

        mv_dir = config['mv_video_dir']
        upstream_candidates = [upstream_video + '.mp4', f'{mv_dir}/{item["id"]}-upstream.mp4']
        downstream_candidates = [downstream_video + '.mp4', f'{mv_dir}/{item["id"]}-downstream.mp4']

        resolved = []
        for candidates in [upstream_candidates, downstream_candidates]:
            current = None
            for rel_path in candidates:
                if osp.exists(osp.join(data_root, rel_path)):
                    current = _strip_video_suffix(rel_path)
                    break
            resolved.append(current)

        if all(resolved):
            return tuple(resolved)
        return None

    def _save_video_frames_by_path(self, video):
        import decord
        import portalocker
        from PIL import Image

        frame_key = _safe_frame_key(video)
        vid_path = _video_full_path(self.data_root, video)
        if self.fps > 0:
            vid = decord.VideoReader(vid_path)
            total_frames = len(vid)
            video_fps = vid.get_avg_fps()
            total_duration = total_frames / video_fps
            required_frames = int(total_duration * self.fps)
            step_size = video_fps / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(frame_key, len(indices))
            flag = np.all([osp.exists(p) for p in frame_paths])
            if flag:
                return frame_paths
            lock_path = osp.join(self.frame_root, frame_key + '.lock')
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if np.all([osp.exists(p) for p in frame_paths]):
                    return frame_paths
                images = [vid[i].asnumpy() for i in indices]
                images = [Image.fromarray(arr) for arr in images]
                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        im.save(pth)
            return frame_paths

        frame_paths = self.frame_paths(frame_key)
        flag = np.all([osp.exists(p) for p in frame_paths])
        if flag:
            return frame_paths
        lock_path = osp.join(self.frame_root, frame_key + '.lock')
        with portalocker.Lock(lock_path, 'w', timeout=30):
            if np.all([osp.exists(p) for p in frame_paths]):
                return frame_paths
            vid = decord.VideoReader(vid_path)
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)
        return frame_paths

    def _sample_frame_timestamps(self, video):
        import decord

        vid_path = _video_full_path(self.data_root, video)
        vid = decord.VideoReader(vid_path)
        total_frames = len(vid)
        video_fps = vid.get_avg_fps()
        if self.fps > 0:
            total_duration = total_frames / video_fps
            required_frames = max(1, int(total_duration * self.fps))
            step_size = video_fps / self.fps
            indices = [min(int(i * step_size), total_frames - 1) for i in range(required_frames)]
        else:
            step_size = total_frames / (self.nframe + 1)
            indices = [min(int(i * step_size), total_frames - 1) for i in range(1, self.nframe + 1)]
        return [idx / video_fps for idx in indices]

    def _append_timestamped_frames(self, message, video, prefix=None):
        frames = self.save_video_frames(video)
        timestamps = self._sample_frame_timestamps(video)
        if prefix:
            message.append(dict(type='text', value=prefix))
        message.append(dict(type='text', value=_format_frame_timestamp_text(timestamps)))
        for frame in frames:
            message.append(dict(type='image', value=frame))

    @staticmethod
    def _resolve_duration(data_root, video, item):
        for key in ['duration', 'video_duration']:
            if key in item:
                try:
                    return float(item[key])
                except (TypeError, ValueError):
                    pass

        duration = _video_duration(_video_full_path(data_root, video))
        if not np.isnan(duration):
            return duration

        label = item.get('label')
        if isinstance(label, dict) and label.get('end') is not None:
            try:
                return float(label['end'])
            except (TypeError, ValueError):
                pass
        return np.nan

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            line = self.data.iloc[line]

        message = []
        video_paths = _literal_eval(line.get('video_paths'), default=[line['video']]) or [line['video']]
        if line['task_type'] == 'mv' and len(video_paths) < 2:
            raise ValueError(f"MV sample id={line['id']} must contain two videos.")

        if video_llm:
            if line['task_type'] == 'mv':
                message.append(dict(type='text', value='The first video is the upstream view.\n'))
                message.append(dict(type='video', value=_video_full_path(self.data_root, video_paths[0])))
                message.append(dict(type='text', value='The second video is the downstream view.\n'))
                message.append(dict(type='video', value=_video_full_path(self.data_root, video_paths[1])))
            else:
                message.append(dict(type='video', value=_video_full_path(self.data_root, line['video'])))
        else:
            if line['task_type'] == 'mv':
                message.append(dict(type='text', value='The following frames are from the upstream view.\n'))
                for frame in self._save_video_frames_by_path(video_paths[0]):
                    message.append(dict(type='image', value=frame))
                message.append(dict(type='text', value='The following frames are from the downstream view.\n'))
                for frame in self._save_video_frames_by_path(video_paths[1]):
                    message.append(dict(type='image', value=frame))
            else:
                if line['task_type'] == 'tg' and self.transvideobench_config.get('enable_frame_timestamps', False):
                    self._append_timestamped_frames(
                        message,
                        line['video'],
                        prefix='The following sampled frames come from the same video.\n',
                    )
                else:
                    frames = self.save_video_frames(line['video'])
                    for frame in frames:
                        message.append(dict(type='image', value=frame))

        if line['task_type'] == 'mcq':
            options = _literal_eval(line['options'], default={})
            prompt = MCQ_PROMPT.format(question=line['question'], options=_format_options(options))
        elif line['task_type'] == 'mv':
            options = _literal_eval(line['options'], default={})
            prompt = MV_PROMPT.format(question=line['question'], options=_format_options(options))
        else:
            prompt = TG_PROMPT.format(question=line['question'])
        message.append(dict(type='text', value=prompt))
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        data = self._score_predictions(data)

        score_file = get_intermediate_file_path(eval_file, '_score')
        rating_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        dump(data, score_file)

        rating = self._summarize(data)
        dump(rating, rating_file)
        return rating

    @classmethod
    def _score_predictions(cls, data):
        for col in ['mcq_score', 'mv_score', 'primary_score', 'mIoU']:
            data[col] = np.nan
        for th in cls.IOUTHS:
            data[f'R@{th}'] = np.nan

        for idx, row in data.iterrows():
            if row['task_type'] in ['mcq', 'mv']:
                options = _literal_eval(row.get('options'), default={})
                pred = _extract_mcq_answer(row.get('prediction', ''), options)
                score = float(pred == str(row['answer']).strip())
                if row['task_type'] == 'mcq':
                    data.loc[idx, 'mcq_score'] = score
                else:
                    data.loc[idx, 'mv_score'] = score
                data.loc[idx, 'primary_score'] = score
            else:
                metrics = _score_timelens_tg(row.get('prediction', ''), row['answer'])
                for key, value in metrics.items():
                    data.loc[idx, key] = value
                data.loc[idx, 'primary_score'] = metrics['mIoU']
        return data

    @classmethod
    def _summarize(cls, data):
        results = {}
        for dim, values in cls._dimension_values(data):
            for value in values:
                sub = data if dim == 'overall' else data[data[dim] == value]
                prefix = 'overall' if dim == 'overall' else f'{dim}|{value}'
                cls._update_summary(results, prefix, sub)
        return results

    @staticmethod
    def _dimension_values(data):
        yield 'overall', ['overall']
        for dim in ['static_dynamic', 'category', 'duration_bucket']:
            values = [x for x in data[dim].dropna().unique().tolist() if str(x)]
            yield dim, sorted(values)

    @classmethod
    def _update_summary(cls, results, prefix, data):
        results[f'{prefix}|count'] = int(len(data))
        results[f'{prefix}|overall'] = cls._mean_percent(data['primary_score'])

        mcq = data[data['task_type'] == 'mcq']
        results[f'{prefix}|mcq_count'] = int(len(mcq))
        results[f'{prefix}|mcq_acc'] = cls._mean_percent(mcq['mcq_score']) if len(mcq) else np.nan

        mv = data[data['task_type'] == 'mv']
        results[f'{prefix}|mv_count'] = int(len(mv))
        results[f'{prefix}|mv_acc'] = cls._mean_percent(mv['mv_score']) if len(mv) else np.nan

        tg = data[data['task_type'] == 'tg']
        results[f'{prefix}|tg_count'] = int(len(tg))
        for metric in ['mIoU'] + [f'R@{th}' for th in cls.IOUTHS]:
            results[f'{prefix}|tg_{metric}'] = cls._mean_percent(tg[metric]) if len(tg) else np.nan

    @staticmethod
    def _mean_percent(values):
        values = pd.to_numeric(values, errors='coerce').dropna()
        return round(float(values.mean() * 100), 3) if len(values) else np.nan

    @classmethod
    def report_primary_metric(cls, metrics):
        if not isinstance(metrics, dict) or 'overall|overall' not in metrics:
            return {}
        return {'Overall Score': metrics['overall|overall']}

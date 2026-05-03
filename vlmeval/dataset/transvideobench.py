import ast
import json
import os
import os.path as osp
import re
from pathlib import Path

import numpy as np
import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load
from .video_base import VideoBaseDataset


IOU_THRESHOLDS = [0.3, 0.5, 0.7]

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


# Official release uses hyphens (results-en.json); older layouts used underscores.
_MCQ_ANNO_NAMES = ('results_en.json', 'results-en.json')
_TG_ANNO_NAMES = ('results-tg_en.json', 'results-tg-en.json')


def _first_existing_anno(data_root, names):
    for name in names:
        path = osp.join(data_root, name)
        if osp.exists(path):
            return path
    return None


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

    time_regex = re.compile(r'\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b')
    time_matches = re.findall(time_regex, paragraph)
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

    if not timestamps:
        patterns = [
            r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s+to\s+(\d+\.?\d*)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, paragraph)
            if matches:
                timestamps = [(float(start), float(end)) for start, end in matches]
                break

    if not timestamps:
        matches = re.findall(r'\b(\d+\.\d+|\d+)\b', paragraph)
        matches = matches[:len(matches) // 2 * 2]
        timestamps = [(float(matches[i]), float(matches[i + 1])) for i in range(0, len(matches), 2)]

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
    text = str(prediction).strip()
    text = text.replace('（', '(').replace('）', ')').replace('．', '.')
    prefixes = [
        'The best answer is', 'The correct answer is', 'The answer is',
        'The answer', 'The best option is', 'The correct option is',
        'Best answer:', 'Best option:', 'Answer:', 'Option:', '答案是', '答案：'
    ]
    for prefix in prefixes:
        text = text.replace(prefix, '')

    match = re.search(r'(?<![A-Za-z])[ABCD](?![A-Za-z])', text)
    if match is not None:
        return match.group(0)

    if isinstance(options, dict):
        for key, option in options.items():
            if str(option).strip() and str(option).strip() in text:
                return key
    return ''


class TransVideoBench(VideoBaseDataset):
    TYPE = 'Video-Mixed'
    IOUTHS = IOU_THRESHOLDS
    DATASET_NAMES = ['TransVideoBench', 'TransVideoBench_MCQ', 'TransVideoBench_TG']

    def __init__(self, dataset='TransVideoBench', nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return cls.DATASET_NAMES

    def prepare_dataset(self, dataset_name='TransVideoBench'):
        data_root = self._find_data_root()
        data_file = osp.join(data_root, f'{dataset_name}.tsv')
        data = self._build_dataframe(data_root, dataset_name)
        data.to_csv(data_file, sep='\t', index=False)
        return dict(root=data_root, data_file=data_file)

    @staticmethod
    def _find_data_root():
        candidates = []
        if os.environ.get('TRANSVIDEOBENCH_ROOT'):
            candidates.append(os.environ['TRANSVIDEOBENCH_ROOT'])
        candidates.append(osp.join(LMUDataRoot(), 'TransVideoBench'))
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(str(repo_root / 'trans_data_example'))

        for root in candidates:
            if not root:
                continue
            if _first_existing_anno(root, _MCQ_ANNO_NAMES) or _first_existing_anno(root, _TG_ANNO_NAMES):
                return root
        raise FileNotFoundError(
            'Cannot find TransVideoBench annotations. Set TRANSVIDEOBENCH_ROOT or put '
            'results_en.json or results-en.json, and results-tg_en.json or results-tg-en.json '
            'under $LMUData/TransVideoBench.'
        )

    @classmethod
    def _build_dataframe(cls, data_root, dataset_name):
        records = []
        if dataset_name != 'TransVideoBench_TG':
            records.extend(cls._load_mcq_records(data_root))
        if dataset_name != 'TransVideoBench_MCQ':
            records.extend(cls._load_tg_records(data_root))

        data = pd.DataFrame(records)
        if data.empty:
            raise ValueError(f'No samples found for {dataset_name} in {data_root}')
        data['index'] = np.arange(len(data))
        return data[[
            'index', 'id', 'video', 'video_path', 'question', 'answer', 'options',
            'task_type', 'static_dynamic', 'category', 'specific_interaction',
            'duration', 'duration_bucket'
        ]]

    @staticmethod
    def _resolve_video(data_root, item, task_type):
        if item.get('video_path'):
            return _strip_video_suffix(item['video_path'])
        if item.get('video'):
            return _strip_video_suffix(item['video'])

        sample_id = item['id']
        task_dirs = ['videos', 'videos-mcq', 'videos_mcq'] if task_type == 'mcq' else ['videos-tg', 'videos_tg', 'tg']
        candidates = [f'{sample_id}.mp4']
        candidates.extend(f'{folder}/{sample_id}.mp4' for folder in task_dirs)
        candidates.extend(f'videos/{sample_id}.mp4' for _ in [0])

        for rel_path in candidates:
            if osp.exists(osp.join(data_root, rel_path)):
                return _strip_video_suffix(rel_path)
        return _strip_video_suffix(candidates[1])

    @staticmethod
    def _resolved_video_filepath(data_root, video):
        """Join data_root with relative video id, or honor absolute paths from annotations."""
        rel = video + '.mp4'
        return rel if osp.isabs(rel) else osp.join(data_root, rel)

    @classmethod
    def _load_mcq_records(cls, data_root):
        anno_path = _first_existing_anno(data_root, _MCQ_ANNO_NAMES)
        if not anno_path:
            return []
        records = []
        for item in _safe_json_load(anno_path):
            video = cls._resolve_video(data_root, item, 'mcq')
            full_path = cls._resolved_video_filepath(data_root, video)
            if not osp.isfile(full_path):
                print(
                    f'[TransVideoBench] Skip MCQ id={item.get("id")}: video file not found: {full_path}'
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
                'duration': duration,
                'duration_bucket': _duration_bucket(duration),
            })
        return records

    @classmethod
    def _load_tg_records(cls, data_root):
        anno_path = _first_existing_anno(data_root, _TG_ANNO_NAMES)
        if not anno_path:
            return []
        records = []
        for item in _safe_json_load(anno_path):
            video = cls._resolve_video(data_root, item, 'tg')
            full_path = cls._resolved_video_filepath(data_root, video)
            if not osp.isfile(full_path):
                print(
                    f'[TransVideoBench] Skip TG id={item.get("id")}: video file not found: {full_path}'
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
                'duration': duration,
                'duration_bucket': _duration_bucket(duration),
            })
        return records

    @classmethod
    def _resolve_duration(cls, data_root, video, item):
        for key in ['duration', 'video_duration']:
            if key in item:
                try:
                    return float(item[key])
                except (TypeError, ValueError):
                    pass

        duration = _video_duration(cls._resolved_video_filepath(data_root, video))
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
        if video_llm:
            video_msg = dict(type='video', value=osp.join(self.data_root, line['video'] + '.mp4'))
            if self.nframe > 0:
                video_msg['nframes'] = int(self.nframe)
            if self.fps > 0:
                video_msg['fps'] = float(self.fps)
            message.append(video_msg)
        else:
            frames = self.save_video_frames(line['video'])
            for frame in frames:
                message.append(dict(type='image', value=frame))

        if line['task_type'] == 'mcq':
            options = _literal_eval(line['options'], default={})
            prompt = MCQ_PROMPT.format(question=line['question'], options=_format_options(options))
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
        for col in ['mcq_score', 'primary_score', 'mIoU']:
            data[col] = np.nan
        for th in cls.IOUTHS:
            data[f'R@{th}'] = np.nan

        for idx, row in data.iterrows():
            if row['task_type'] == 'mcq':
                options = _literal_eval(row.get('options'), default={})
                pred = _extract_mcq_answer(row.get('prediction', ''), options)
                score = float(pred == str(row['answer']).strip())
                data.loc[idx, 'mcq_score'] = score
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

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


DYNAMIC_MCQ_CATEGORIES = [
    "行人和交通管理者",
    "单一车辆行为",
    "车辆间交互",
    "车流级别行为",
    "计数",
]

STATIC_MCQ_CATEGORIES = [
    "行人和交通管理者",
    "车辆",
    "道路与设施",
    "计数",
]

MV_STATIC_DYNAMIC = ["动态", "静态"]

TG_CATEGORIES = [
    "行人和交通管理者",
    "单一车辆行为",
    "车辆间交互",
    "车流级别行为",
    "交通信号感知",
]

TG_METRICS = ["mIoU", "R@0.3", "R@0.5", "R@0.7"]


def mean_percent(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return round(float(values.mean() * 100), 3)


def normalize_dataframe(df):
    data = df.copy()
    for col in ["task_type", "static_dynamic", "category"]:
        if col in data.columns:
            data[col] = data[col].fillna("").astype(str).str.strip()
    if "task_type" in data.columns:
        data["task_type"] = data["task_type"].str.lower()
    return data


def summarize_acc_by_category(df, categories, category_col="category", score_col="mcq_score"):
    items = {}
    avg_values = []
    for category in categories:
        subset = df[df[category_col] == category]
        if subset.empty:
            continue
        acc = mean_percent(subset[score_col])
        items[category] = {
            "count": int(len(subset)),
            "acc": acc,
        }
        if acc is not None:
            avg_values.append(acc)
    return {
        "items": items,
        "avg_acc": round(sum(avg_values) / len(avg_values), 3) if avg_values else None,
    }


def summarize_tg_by_category(df, categories):
    items = {}
    metric_avgs = {metric: [] for metric in TG_METRICS}
    for category in categories:
        subset = df[df["category"] == category]
        if subset.empty:
            continue
        entry = {"count": int(len(subset))}
        for metric in TG_METRICS:
            value = mean_percent(subset[metric]) if metric in subset.columns else None
            entry[metric] = value
            if value is not None:
                metric_avgs[metric].append(value)
        items[category] = entry

    overall_avg = {
        metric: round(sum(values) / len(values), 3) if values else None
        for metric, values in metric_avgs.items()
    }
    return {
        "items": items,
        "overall_avg": overall_avg,
    }


def summarize_groups(df):
    result = {
        "input_count": int(len(df)),
        "available_sections": [],
    }

    mcq = df[df["task_type"] == "mcq"]
    mv = df[df["task_type"] == "mv"]
    tg = df[df["task_type"] == "tg"]

    big_group_avgs = []

    dynamic_mcq = mcq[mcq["static_dynamic"] == "动态"]
    dynamic_summary = summarize_acc_by_category(dynamic_mcq, DYNAMIC_MCQ_CATEGORIES)
    if dynamic_summary["items"]:
        result["dynamic_mcq"] = dynamic_summary
        result["available_sections"].append("dynamic_mcq")
        if dynamic_summary["avg_acc"] is not None:
            big_group_avgs.append(dynamic_summary["avg_acc"])

    static_mcq = mcq[mcq["static_dynamic"] == "静态"]
    static_summary = summarize_acc_by_category(static_mcq, STATIC_MCQ_CATEGORIES)
    if static_summary["items"]:
        result["static_mcq"] = static_summary
        result["available_sections"].append("static_mcq")
        if static_summary["avg_acc"] is not None:
            big_group_avgs.append(static_summary["avg_acc"])

    mv_summary = summarize_acc_by_category(mv, MV_STATIC_DYNAMIC, category_col="static_dynamic", score_col="mv_score")
    if mv_summary["items"]:
        result["mv"] = mv_summary
        result["available_sections"].append("mv")
        if mv_summary["avg_acc"] is not None:
            big_group_avgs.append(mv_summary["avg_acc"])

    if big_group_avgs:
        result["overall_acc"] = round(sum(big_group_avgs) / len(big_group_avgs), 3)
    else:
        result["overall_acc"] = None

    tg_summary = summarize_tg_by_category(tg, TG_CATEGORIES)
    if tg_summary["items"]:
        result["tg"] = tg_summary
        result["available_sections"].append("tg")

    return result


def print_acc_section(title, summary):
    print(f"\n## {title}")
    for name, item in summary["items"].items():
        print(f"- {name}: count={item['count']}, acc={item['acc']}")
    print(f"- avg_acc: {summary['avg_acc']}")


def print_tg_section(summary):
    print("\n## TG")
    for name, item in summary["items"].items():
        print(
            f"- {name}: count={item['count']}, "
            f"mIoU={item['mIoU']}, R@0.3={item['R@0.3']}, "
            f"R@0.5={item['R@0.5']}, R@0.7={item['R@0.7']}"
        )
    overall = summary["overall_avg"]
    print(
        f"- overall_avg: mIoU={overall['mIoU']}, "
        f"R@0.3={overall['R@0.3']}, R@0.5={overall['R@0.5']}, "
        f"R@0.7={overall['R@0.7']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a TransVideoBench *_score.xlsx file by grouped dimensions."
    )
    parser.add_argument("score_file", help="Path to a TransVideoBench *_score.xlsx file")
    parser.add_argument(
        "--json-output",
        help="Optional path to save the analysis result as JSON",
    )
    args = parser.parse_args()

    score_path = Path(args.score_file)
    df = pd.read_excel(score_path)
    df = normalize_dataframe(df)
    summary = summarize_groups(df)
    summary["input_file"] = str(score_path)

    print(f"# Analysis for {score_path.name}")
    print(f"- input_count: {summary['input_count']}")
    print(f"- available_sections: {', '.join(summary['available_sections']) if summary['available_sections'] else 'none'}")
    print(f"- overall_acc: {summary['overall_acc']}")

    if "dynamic_mcq" in summary:
        print_acc_section("Dynamic MCQ", summary["dynamic_mcq"])
    if "static_mcq" in summary:
        print_acc_section("Static MCQ", summary["static_mcq"])
    if "mv" in summary:
        print_acc_section("MV", summary["mv"])
    if "tg" in summary:
        print_tg_section(summary["tg"])

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON to {output_path}")


if __name__ == "__main__":
    main()

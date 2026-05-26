"""
CI 评估守门脚本：读取 Ragas 评估结果，按阈值判断是否通过。

使用方式：
    python scripts/ci_eval_gate.py --results data/evaluation/ragas_results.json

退出码：
    0 - 所有指标通过阈值
    1 - 至少一个指标未达标
"""

import argparse
import json
import sys
from pathlib import Path

# 阈值配置（可根据项目实际情况调整）
THRESHOLDS = {
    "faithfulness": 0.70,  # 答案忠实度：至少 70% 的答案有上下文支撑
    "answer_relevancy": 0.65,  # 答案相关性：至少 65% 相关
    "context_precision": 0.60,  # 上下文精确度：相关 chunk 排名靠前
    "context_recall": 0.55,  # 上下文召回率：ground truth 被检索到
}


def load_results(path: str) -> dict:
    """加载评估结果"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_thresholds(metrics: dict, thresholds: dict) -> tuple[bool, list[str]]:
    """
    检查指标是否达标

    返回：
        (是否全部通过, 失败的指标列表)
    """
    failures = []

    for metric, threshold in thresholds.items():
        actual = metrics.get(metric, 0.0)
        if actual < threshold:
            failures.append(
                f"{metric}: {actual:.4f} < {threshold:.4f} (未达标)"
            )

    return len(failures) == 0, failures


def print_report(metrics: dict, thresholds: dict, passed: bool, failures: list[str]) -> None:
    """打印评估报告"""
    print("\n" + "=" * 70)
    print("RAG 评估守门报告")
    print("=" * 70)

    print("\n指标对比：")
    print(f"{'指标':<25} {'实际值':<12} {'阈值':<12} {'状态':<10}")
    print("-" * 70)

    for metric, threshold in thresholds.items():
        actual = metrics.get(metric, 0.0)
        status = "PASS" if actual >= threshold else "FAIL"
        print(f"{metric:<25} {actual:<12.4f} {threshold:<12.4f} {status:<10}")

    print("=" * 70)

    if passed:
        print("\n[PASS] 所有指标均达标，允许合并")
    else:
        print("\n[FAIL] 以下指标未达标：")
        for failure in failures:
            print(f"  - {failure}")
        print("\n请优化 RAG 系统后重新评估")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="CI 评估守门脚本")
    parser.add_argument(
        "--results",
        required=True,
        help="Ragas 评估结果文件路径（JSON 格式）",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="自定义阈值配置文件（JSON 格式，可选）",
    )

    args = parser.parse_args()

    # 加载阈值配置
    thresholds = THRESHOLDS
    if args.thresholds:
        with open(args.thresholds, "r", encoding="utf-8") as f:
            thresholds = json.load(f)

    # 加载评估结果
    if not Path(args.results).exists():
        print(f"❌ 评估结果文件不存在: {args.results}", file=sys.stderr)
        sys.exit(1)

    results = load_results(args.results)
    metrics = results.get("metrics", {})

    if not metrics:
        print("❌ 评估结果中未找到 metrics 字段", file=sys.stderr)
        sys.exit(1)

    # 检查阈值
    passed, failures = check_thresholds(metrics, thresholds)

    # 打印报告
    print_report(metrics, thresholds, passed, failures)

    # 返回退出码
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

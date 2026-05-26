import json
import os
import statistics
from datetime import datetime


class StatsCollector:
    def __init__(self):
        self.started_at = datetime.now().isoformat()
        self.total_tasks = 0
        self.success_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.retry_total = 0
        self.duration_list = []
        self.http_status_count = {}
        self.errors = []

    def set_total_tasks(self, total: int) -> None:
        self.total_tasks = total

    def add_success(self, duration_sec: float, retries: int, http_status: int | None) -> None:
        self.success_count += 1
        self.duration_list.append(duration_sec)
        self.retry_total += max(retries, 0)
        if http_status is not None:
            key = str(http_status)
            self.http_status_count[key] = self.http_status_count.get(key, 0) + 1

    def add_failed(self, duration_sec: float, retries: int, http_status: int | None, error: str) -> None:
        self.failed_count += 1
        self.duration_list.append(duration_sec)
        self.retry_total += max(retries, 0)
        if http_status is not None:
            key = str(http_status)
            self.http_status_count[key] = self.http_status_count.get(key, 0) + 1
        if error:
            self.errors.append(error)

    def add_skipped(self) -> None:
        self.skipped_count += 1

    def _calc_duration_stats(self) -> dict:
        if not self.duration_list:
            return {"avg_sec": 0, "median_sec": 0, "p95_sec": 0, "max_sec": 0}
        values = sorted(self.duration_list)
        p95_index = max(int(len(values) * 0.95) - 1, 0)
        return {
            "avg_sec": round(sum(values) / len(values), 3),
            "median_sec": round(statistics.median(values), 3),
            "p95_sec": round(values[p95_index], 3),
            "max_sec": round(max(values), 3),
        }

    def build_summary(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": datetime.now().isoformat(),
            "total_tasks": self.total_tasks,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "retry_total": self.retry_total,
            "http_status_count": self.http_status_count,
            "duration_stats": self._calc_duration_stats(),
            "error_samples": self.errors[:20],
        }

    def export(self, output_dir: str) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        summary = self.build_summary()

        json_path = os.path.join(output_dir, "task_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        md_lines = [
            "# 任务汇总",
            "",
            f"- 启动时间: {summary['started_at']}",
            f"- 结束时间: {summary['finished_at']}",
            f"- 总任务数: {summary['total_tasks']}",
            f"- 成功: {summary['success_count']}",
            f"- 失败: {summary['failed_count']}",
            f"- 跳过: {summary['skipped_count']}",
            f"- 总重试次数: {summary['retry_total']}",
            "",
            "## 耗时统计(秒)",
            f"- avg: {summary['duration_stats']['avg_sec']}",
            f"- median: {summary['duration_stats']['median_sec']}",
            f"- p95: {summary['duration_stats']['p95_sec']}",
            f"- max: {summary['duration_stats']['max_sec']}",
            "",
            "## HTTP 状态码分布",
        ]
        if summary["http_status_count"]:
            for code, cnt in sorted(summary["http_status_count"].items(), key=lambda x: x[0]):
                md_lines.append(f"- {code}: {cnt}")
        else:
            md_lines.append("- 无")

        if summary["error_samples"]:
            md_lines.append("")
            md_lines.append("## 错误样本")
            for err in summary["error_samples"]:
                md_lines.append(f"- {err}")

        md_path = os.path.join(output_dir, "task_summary.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        return summary

"""
QwenVL-Tutor 终端实时可视化 Dashboard

提供训练/评估/优化时的实时进度窗口:
    - 进度条（epoch / step / ETA）
    - 实时指标（loss / reward / accuracy）
    - GPU / 显存监控
    - 系统状态表格

使用示例:
    from trainer.terminal_dashboard import Dashboard, dashboard_context

    with dashboard_context(title="SFT Training") as dash:
        for step in range(steps):
            dash.update(step=step, total=steps, loss=0.5, lr=1e-4, gpu_mem=20.5)
"""
import os
import sys
import time
import shutil
from contextlib import contextmanager
from typing import Optional, Dict, Any

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
    )
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def _check_rich():
    if not RICH_AVAILABLE:
        print("[WARN] rich 不可用，请运行: pip install rich>=13.7.0")
        return False
    return True


class Dashboard:
    """
    通用终端 Dashboard

    显示内容:
        - 顶部: 标题
        - 中部: 进度条
        - 下部: 实时指标表格
    """

    def __init__(
        self,
        title: str = "QwenVL-Tutor",
        total_steps: Optional[int] = None,
        show_gpu: bool = True,
        refresh_per_second: int = 4,
    ):
        if not _check_rich():
            self.enabled = False
            return
        self.enabled = True
        self.console = Console()
        self.title = title
        self.show_gpu = show_gpu
        self.refresh_per_second = refresh_per_second

        # 进度条
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.task_id = self.progress.add_task(
            "Training", total=total_steps or 100
        )

        # 历史指标（用于显示趋势）
        self.history: Dict[str, list] = {}

    def update(
        self,
        step: Optional[int] = None,
        total: Optional[int] = None,
        **metrics: Any,
    ):
        """更新进度和指标"""
        if not self.enabled:
            return
        if step is not None:
            if total is not None:
                self.progress.update(self.task_id, total=total, completed=step)
            else:
                self.progress.update(self.task_id, completed=step)
        for k, v in metrics.items():
            self.history.setdefault(k, []).append((time.time(), v))
            # 只保留最近 100 个点
            if len(self.history[k]) > 100:
                self.history[k] = self.history[k][-100:]

    def get_metrics_table(self) -> Table:
        """生成实时指标表格"""
        table = Table(title="📊 实时指标", show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan", width=20)
        table.add_column("当前值", justify="right", style="green")
        table.add_column("趋势", justify="right", style="yellow")

        for k, vals in self.history.items():
            if not vals:
                continue
            current = vals[-1][1]
            if isinstance(current, float):
                current_str = f"{current:.4f}"
            else:
                current_str = str(current)

            # 趋势（最近 5 个点的变化方向）
            if len(vals) >= 5:
                recent = [v[1] for v in vals[-5:] if isinstance(v[1], (int, float))]
                if recent:
                    delta = recent[-1] - recent[0]
                    if abs(delta) < 1e-6:
                        trend = "→ 平稳"
                    elif delta > 0:
                        trend = f"↑ +{delta:.4f}"
                    else:
                        trend = f"↓ {delta:.4f}"
                else:
                    trend = "—"
            else:
                trend = "—"

            table.add_row(k, current_str, trend)

        return table

    def get_system_table(self) -> Table:
        """生成系统状态表格"""
        table = Table(title="💻 系统状态", show_header=True, header_style="bold cyan")
        table.add_column("项目", style="cyan", width=15)
        table.add_column("值", justify="right", style="green")

        if self.show_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(min(torch.cuda.device_count(), 4)):
                        mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
                        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                        util = mem_alloc / mem_total * 100 if mem_total > 0 else 0
                        table.add_row(
                            f"GPU {i} 显存",
                            f"{mem_alloc:.1f} / {mem_total:.1f} GB ({util:.0f}%)"
                        )
                else:
                    table.add_row("GPU", "不可用")
            except ImportError:
                table.add_row("GPU", "torch 未安装")

        # 时间统计
        if self.history:
            start_time = min(v[0] for vals in self.history.values() for v in vals)
            elapsed = time.time() - start_time
            table.add_row("已用时", f"{elapsed:.0f} 秒")

        return table

    def render(self) -> Panel:
        """渲染整个 Dashboard"""
        layout = Table.grid()
        layout.add_column()
        layout.add_row(self.progress)
        layout.add_row(self.get_metrics_table())
        if self.show_gpu:
            layout.add_row(self.get_system_table())
        return Panel(
            layout,
            title=f"[bold]{self.title}[/bold]",
            border_style="blue",
            padding=(0, 1),
        )


@contextmanager
def dashboard_context(title: str = "QwenVL-Tutor", total_steps: Optional[int] = None):
    """
    Dashboard 上下文管理器

    使用示例:
        with dashboard_context("SFT Training", total_steps=1000) as dash:
            for step in range(1000):
                dash.update(step=step, loss=0.5, lr=1e-4)
    """
    if not _check_rich():
        yield None
        return

    dash = Dashboard(title=title, total_steps=total_steps)
    with Live(dash.render(), console=dash.console, refresh_per_second=dash.refresh_per_second) as live:
        # 用 update 方法替换 render
        original_update = dash.update

        def new_update(*args, **kwargs):
            original_update(*args, **kwargs)
            live.update(dash.render())

        dash.update = new_update
        try:
            yield dash
        finally:
            pass


# ============================================================================
# 简化版进度条（rich 不可用时）
# ============================================================================

class SimpleProgress:
    """
    简化版进度条（不依赖 rich）

    使用 tqdm 或纯 print
    """
    def __init__(self, total: int, desc: str = "Progress"):
        try:
            from tqdm import tqdm
            self.bar = tqdm(total=total, desc=desc, ncols=80, file=sys.stderr)
            self.enabled = True
            self.kind = "tqdm"
        except ImportError:
            self.bar = None
            self.enabled = False
            self.kind = "print"
            print(f"{desc}: 0/{total}")

    def update(self, n: int = 1, **metrics):
        if self.kind == "tqdm":
            self.bar.update(n)
            if metrics:
                postfix = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in metrics.items())
                self.bar.set_postfix_str(postfix)
        else:
            print(f"  step={n}, {metrics}")

    def close(self):
        if self.kind == "tqdm":
            self.bar.close()


def get_progress(total: int, desc: str = "Progress"):
    """
    工厂函数：自动选择 rich dashboard 或简单进度条
    """
    if RICH_AVAILABLE:
        # 返回一个简单的 dashboard 包装，避免 context manager 复杂性
        return SimpleProgress(total, desc)
    return SimpleProgress(total, desc)


# ============================================================================
# 评估阶段专用 dashboard
# ============================================================================

class EvalDashboard:
    """
    评估阶段专用 dashboard

    显示每个数据集的进度和准确率
    """
    def __init__(self, datasets: list, title: str = "Evaluation"):
        if not _check_rich():
            self.enabled = False
            return
        self.enabled = True
        self.console = Console()
        self.datasets = datasets
        self.results = {}
        self.title = title
        self.current_dataset = None
        self.console.print(f"[bold]{title}[/bold] 开始，共 {len(datasets)} 个数据集")

    def start_dataset(self, name: str, total: int):
        if not self.enabled:
            return
        self.current_dataset = name
        self.console.print(f"\n[cyan]▶ {name}[/cyan] ({total} 样本)")

    def update_dataset(self, name: str, completed: int, total: int, accuracy: float = None):
        if not self.enabled:
            return
        pct = completed / total * 100 if total > 0 else 0
        bar_width = 30
        filled = int(bar_width * completed / max(total, 1))
        bar = "█" * filled + "░" * (bar_width - filled)
        if accuracy is not None:
            self.console.print(
                f"\r  [{bar}] {completed}/{total} ({pct:.0f}%) acc={accuracy:.4f}",
                end="", highlight=False
            )
        else:
            self.console.print(
                f"\r  [{bar}] {completed}/{total} ({pct:.0f}%)",
                end="", highlight=False
            )

    def finish_dataset(self, name: str, accuracy: float, samples: int):
        if not self.enabled:
            return
        self.results[name] = (accuracy, samples)
        self.console.print(f"  → acc={accuracy:.4f} ({samples} 样本)")

    def finish_all(self):
        if not self.enabled:
            return
        if not self.results:
            return
        self.console.print(f"\n\n[bold green]✓ {self.title} 完成[/bold green]")
        table = Table(title="评估结果汇总", show_header=True, header_style="bold magenta")
        table.add_column("数据集", style="cyan")
        table.add_column("准确率", justify="right", style="green")
        table.add_column("样本数", justify="right", style="yellow")
        for name, (acc, n) in self.results.items():
            table.add_row(name, f"{acc:.4f}", str(n))
        # 平均
        if self.results:
            avg = sum(acc for acc, _ in self.results.values()) / len(self.results)
            table.add_row("[bold]平均[/bold]", f"[bold green]{avg:.4f}[/bold green]", "—")
        self.console.print(table)


# ============================================================================
# 优化阶段专用 dashboard
# ============================================================================

class OptimizeDashboard:
    """优化阶段专用 dashboard"""
    def __init__(self, title: str = "Optimization"):
        if not _check_rich():
            self.enabled = False
            return
        self.enabled = True
        self.console = Console()
        self.title = title

    def show_weights(self, weights: Dict[str, float], max_show: int = 10):
        if not self.enabled:
            return
        # 排序：权重高的先显示
        sorted_w = sorted(weights.items(), key=lambda x: -x[1])[:max_show]
        self.console.print(f"\n[bold]{self.title}[/bold] 数据集权重（Top {max_show}）:")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("数据集", style="cyan")
        table.add_column("权重", justify="right", style="green")
        table.add_column("条形", style="yellow")
        for name, w in sorted_w:
            bar = "█" * int(w * 20)
            table.add_row(name, f"{w:.2f}", bar)
        self.console.print(table)


if __name__ == "__main__":
    # 简单测试
    print(f"rich 可用: {RICH_AVAILABLE}")

    with dashboard_context("测试 Dashboard", total_steps=100) as dash:
        for i in range(100):
            time.sleep(0.05)
            loss = 2.0 * (1 - i / 100) + 0.1 * (i % 10) / 10
            lr = 1e-4 * (1 - i / 100)
            dash.update(step=i, loss=loss, lr=lr, gpu_mem=20 + i * 0.01)

    print("测试完成")

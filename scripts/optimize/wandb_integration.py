"""
QwenVL-Tutor 训练过程监控与可视化集成

支持:
    1. Weights & Biases (wandb) - 国际通用
    2. SwanLab - 国产替代，国内访问快
    3. TensorBoard - 离线方案
    4. 本地 JSON 日志（无外部依赖�?
用法:
    # 在训练脚本中:
    from scripts.wandb_integration import TrainingLogger
    logger = TrainingLogger(backend="wandb", project="QwenVL-Tutor", config={...})
    logger.log({"train/loss": loss, "train/lr": lr}, step=step)
    logger.log_metrics({"eval/accuracy": 0.72, "eval/reward": 0.55})
    logger.finish()

    # 命令�?
    python scripts/wandb_integration.py --backend swanlab --test
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR.parent.parent)


class TrainingLogger:
    """
    统一的训练日志接口，支持多种后端

    自动降级:
        1. 优先使用指定 backend
        2. 若不可用，自动降级到 local JSON
    """

    def __init__(self, backend: str = "auto", project: str = "QwenVL-Tutor",
                 name: str = None, config: dict = None, log_dir: str = "logs"):
        """
        Args:
            backend: "wandb" | "swanlab" | "tensorboard" | "local" | "auto"
            project: 项目名称
            name: 本次运行名称
            config: 超参数配�?            log_dir: 本地日志目录
        """
        self.backend = backend
        self.project = project
        self.name = name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = config or {}
        self.log_dir = log_dir
        self.step = 0

        # 初始化后�?        self.logger = None
        if backend == "auto":
            backend = self._auto_detect()

        if backend == "wandb":
            self.logger = self._init_wandb()
        elif backend == "swanlab":
            self.logger = self._init_swanlab()
        elif backend == "tensorboard":
            self.logger = self._init_tensorboard()
        else:
            self.backend = "local"
            self.logger = self._init_local()

        # 记录启动信息
        self.log({
            "system/backend": self.backend,
            "system/project": self.project,
            "system/name": self.name,
            "system/timestamp": datetime.now().isoformat(),
        })

    def _auto_detect(self) -> str:
        """自动检测可用的后端"""
        try:
            import wandb
            return "wandb"
        except ImportError:
            pass
        try:
            import swanlab
            return "swanlab"
        except ImportError:
            pass
        return "local"

    def _init_wandb(self):
        """初始�?wandb"""
        try:
            import wandb
            wandb.init(project=self.project, name=self.name, config=self.config)
            return wandb
        except ImportError:
            print("⚠️ wandb 未安装，降级�?local 后端")
            print("   安装: pip install wandb")
            self.backend = "local"
            return self._init_local()
        except Exception as e:
            print(f"⚠️ wandb 初始化失�? {e}，降级到 local")
            self.backend = "local"
            return self._init_local()

    def _init_swanlab(self):
        """初始�?SwanLab"""
        try:
            import swanlab
            swanlab.init(project=self.project, experiment_name=self.name, config=self.config)
            return swanlab
        except ImportError:
            print("⚠️ swanlab 未安装，降级�?local 后端")
            print("   安装: pip install swanlab")
            self.backend = "local"
            return self._init_local()
        except Exception as e:
            print(f"⚠️ SwanLab 初始化失�? {e}，降级到 local")
            self.backend = "local"
            return self._init_local()

    def _init_tensorboard(self):
        """初始�?TensorBoard"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(self.log_dir, "tensorboard", self.name)
            os.makedirs(tb_dir, exist_ok=True)
            return SummaryWriter(tb_dir)
        except ImportError:
            print("⚠️ tensorboard 未安装，降级�?local 后端")
            print("   安装: pip install tensorboard")
            self.backend = "local"
            return self._init_local()

    def _init_local(self):
        """初始化本�?JSON 日志"""
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, f"{self.name}.jsonl")
        return open(log_file, 'a', encoding='utf-8')

    def log(self, metrics: dict, step: int = None):
        """
        记录指标

        Args:
            metrics: 指标字典，如 {"train/loss": 0.5, "train/lr": 1e-4}
            step: 全局步数（可选）
        """
        if step is not None:
            self.step = step

        record = {"step": self.step, "time": datetime.now().isoformat(), **metrics}

        if self.backend == "wandb":
            self.logger.log(metrics, step=self.step)
        elif self.backend == "swanlab":
            self.logger.log(metrics, step=self.step)
        elif self.backend == "tensorboard":
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.logger.add_scalar(k, v, self.step)
        else:  # local
            self.logger.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.logger.flush()

    def log_metrics(self, metrics: dict, step: int = None):
        """log 的别�?""
        self.log(metrics, step)

    def log_config(self, config: dict):
        """记录超参数配�?""
        self.config.update(config)
        if self.backend == "wandb":
            self.logger.config.update(config)
        elif self.backend == "swanlab":
            self.logger.config.update(config)
        elif self.backend == "tensorboard":
            self.logger.add_text("config", json.dumps(config, ensure_ascii=False), 0)
        else:
            self.log({"config": config}, step=0)

    def log_artifact(self, file_path: str, name: str = None):
        """记录产物文件（模型权重、评估结果等�?""
        if self.backend == "wandb":
            artifact = self.logger.Artifact(name or os.path.basename(file_path), type="file")
            artifact.add_file(file_path)
            self.logger.log_artifact(artifact)
        elif self.backend == "swanlab":
            # SwanLab 的文件上�?            try:
                self.logger.log_artifact(file_path)
            except Exception:
                pass
        # local / tensorboard 不支持，记录路径即可
        self.log({"artifact/path": file_path, "artifact/name": name})

    def finish(self):
        """结束本次运行"""
        if self.backend == "wandb":
            self.logger.finish()
        elif self.backend == "swanlab":
            self.logger.finish()
        elif self.backend == "tensorboard":
            self.logger.close()
        else:
            if self.logger:
                self.logger.close()
        print(f"�?日志记录完成 (backend: {self.backend}, run: {self.name})")


# ============================================================================
# 训练脚本集成示例（仅作参考，复制�?train_*.py 中）
# ============================================================================

TRAIN_SFT_INTEGRATION = """
# �?trainer/train_sft.py 中添�?

from scripts.wandb_integration import TrainingLogger

# �?main() 中初始化
logger = TrainingLogger(
    backend="auto",  # 自动检�?wandb > swanlab > local
    project="QwenVL-Tutor-sft",
    name=f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config={
        "model": "qwen2-vl-2b",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "data_paths": args.data_paths,
    },
)

# 在每�?step/epoch �?for step, batch in enumerate(dataloader):
    loss = train_step(model, batch)
    logger.log({
        "train/loss": loss,
        "train/lr": scheduler.get_last_lr()[0],
        "train/epoch": epoch,
    }, step=step)

# 训练结束�?logger.log_artifact("out/edu_sft/pytorch_model.bin", name="model_weights")
logger.finish()
"""


def main():
    parser = argparse.ArgumentParser(description="QwenVL-Tutor 训练监控")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "wandb", "swanlab", "tensorboard", "local"],
                        help="日志后端")
    parser.add_argument("--project", type=str, default="QwenVL-Tutor", help="项目�?)
    parser.add_argument("--test", action="store_true", help="测试后端连接")
    parser.add_argument("--show_integration", action="store_true",
                        help="显示训练脚本集成代码")
    args = parser.parse_args()

    if args.show_integration:
        print(TRAIN_SFT_INTEGRATION)
        return

    if args.test:
        # 测试后端连接
        logger = TrainingLogger(backend=args.backend, project=args.project,
                                name="test_run", config={"test": True})
        for i in range(5):
            logger.log({
                "test/loss": 1.0 / (i + 1),
                "test/accuracy": 0.5 + i * 0.05,
            }, step=i)
        logger.log_artifact("README.md", name="test_file")
        logger.finish()
        return

    print("用法:")
    print("  python scripts/wandb_integration.py --backend wandb --test")
    print("  python scripts/wandb_integration.py --show_integration")


if __name__ == "__main__":
    main()

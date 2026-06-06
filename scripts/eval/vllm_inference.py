"""
QwenVL-Tutor vLLM 推理封装

提供基于 vLLM 的批量推理后端，可替代原生 transformers.generate：
    - 吞吐提升 5-20x
    - 自动批处理 + Continuous batching
    - PagedAttention 显存优化
    - 支持多模态（Qwen2-VL）
"""
import os
import sys
import time
import warnings
from typing import List, Dict, Optional, Union

import torch
from PIL import Image

warnings.filterwarnings('ignore')


class VLLMBackend:
    """
    vLLM 推理后端

    使用示例:
        backend = VLLMBackend(
            model_path="./out/edu_sft",
            base_model_path="./model/Qwen2-VL-2B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
        )
        outputs = backend.generate_batch(
            prompts=["<image>\\n这道题怎么做？"],
            images=[image1],
            max_tokens=512,
        )
    """

    def __init__(
        self,
        model_path: str,
        base_model_path: str = "./model/Qwen2-VL-2B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
        quantization: Optional[str] = None,
        enforce_eager: bool = False,
    ):
        """
        Args:
            model_path: LoRA 适配器路径或完整模型路径
            base_model_path: 基座模型路径（加载 LoRA 时需要）
            tensor_parallel_size: 张量并行 GPU 数量
            gpu_memory_utilization: GPU 显存使用率（0.0-1.0）
            max_model_len: 最大序列长度
            dtype: 推理精度（bfloat16 / float16 / float32）
            quantization: 量化方式（awq / gptq / None）
            enforce_eager: 强制使用 eager 模式（调试用）
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM 未安装。请运行: pip install vllm>=0.5.0\n"
                "注：vLLM 需要 CUDA 12.1+ 和对应的 PyTorch"
            )

        print(f"[vLLM] 加载模型: {model_path} (TP={tensor_parallel_size})")

        # 决定加载策略
        is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))

        llm_kwargs = {
            "model": base_model_path if is_lora else model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "dtype": dtype,
            "trust_remote_code": True,
            "enforce_eager": enforce_eager,
            "limit_mm_per_prompt": {"image": 4},  # Qwen2-VL 最多 4 张图
        }

        if quantization:
            llm_kwargs["quantization"] = quantization

        # 多模态支持（Qwen2-VL）
        try:
            from vllm.model_executor.models import _MULTIMODAL_MODELS
            llm_kwargs["model"] = llm_kwargs["model"]  # vLLM 自动识别 Qwen2-VL
        except Exception:
            pass

        self.llm = LLM(**llm_kwargs)
        self.sampling_params = None

        # 如果是 LoRA，启用 LoRA 支持
        if is_lora:
            print(f"[vLLM] 加载 LoRA 适配器: {model_path}")
            from peft import PeftModel
            # vLLM 0.5+ 支持 LoRA 热加载
            self.lora_path = model_path
            self.llm.enable_lora()
        else:
            self.lora_path = None

        self.is_lora = is_lora
        self.model_path = model_path
        self.base_model_path = base_model_path

    def generate_batch(
        self,
        prompts: List[str],
        images: Optional[List[Optional[Image.Image]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
        n: int = 1,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        批量生成

        Args:
            prompts: 文本提示列表
            images: 与 prompts 对应的图像列表（None 表示无图）
            max_tokens: 最大生成 token 数
            temperature: 采样温度（0.0 = greedy）
            top_p: nucleus sampling 参数
            n: 每个 prompt 生成的候选数
            stop: 停止字符串

        Returns:
            生成的文本列表（与 prompts 等长）
        """
        from vllm import SamplingParams
        from vllm.utils import is_list_of

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            **kwargs,
        )

        # 构造 vLLM 输入
        inputs = []
        for i, prompt in enumerate(prompts):
            input_item = {"prompt": prompt}
            if images and i < len(images) and images[i] is not None:
                input_item["multi_modal_data"] = {"image": images[i]}
            inputs.append(input_item)

        # 批量推理
        start_time = time.time()
        if self.is_lora:
            outputs = self.llm.generate(
                inputs,
                sampling_params,
                lora_request=self._get_lora_request(),
            )
        else:
            outputs = self.llm.generate(inputs, sampling_params)

        elapsed = time.time() - start_time
        n_samples = len(prompts)
        print(f"[vLLM] 推理完成: {n_samples} 样本, 耗时 {elapsed:.2f}s, "
              f"吞吐 {n_samples / elapsed:.1f} 样本/秒")

        # 提取文本
        results = []
        for output in outputs:
            results.append(output.outputs[0].text)
        return results

    def generate_with_score(
        self,
        prompts: List[str],
        images: Optional[List[Optional[Image.Image]]] = None,
        max_tokens: int = 512,
        **kwargs,
    ) -> List[Dict]:
        """
        批量生成（返回 token 级 logprobs，用于 GRPO 训练）
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 0.9),
            logprobs=1,  # 返回 token logprobs
        )

        inputs = []
        for i, prompt in enumerate(prompts):
            input_item = {"prompt": prompt}
            if images and i < len(images) and images[i] is not None:
                input_item["multi_modal_data"] = {"image": images[i]}
            inputs.append(input_item)

        outputs = self.llm.generate(
            inputs, sampling_params,
            lora_request=self._get_lora_request() if self.is_lora else None,
        )

        results = []
        for output in outputs:
            results.append({
                "text": output.outputs[0].text,
                "token_ids": output.outputs[0].token_ids,
                "logprobs": output.outputs[0].logprobs,
                "finish_reason": output.outputs[0].finish_reason,
            })
        return results

    def _get_lora_request(self):
        """获取 LoRA 请求对象"""
        from vllm.lora.request import LoRARequest
        if not hasattr(self, "_lora_request_cached"):
            self._lora_request_cached = LoRARequest("edu_lora", 1, self.lora_path)
        return self._lora_request_cached

    def benchmark(self, n_samples: int = 100, max_tokens: int = 256):
        """基准测试：测量吞吐"""
        print(f"\n[vLLM] 基准测试: {n_samples} 样本, max_tokens={max_tokens}")
        dummy_prompts = [f"测试 {i}: " + "请回答。" * 20 for i in range(n_samples)]
        start = time.time()
        _ = self.generate_batch(dummy_prompts, max_tokens=max_tokens, temperature=0.0)
        elapsed = time.time() - start
        print(f"[vLLM] 吞吐: {n_samples / elapsed:.1f} 样本/秒, "
              f"总耗时 {elapsed:.1f}s")
        return n_samples / elapsed


class HFBackend:
    """
    原生 transformers 推理后端（fallback）

    当 vLLM 不可用时使用（如无 CUDA 12.1、显存不足等）
    """

    def __init__(self, model_path, base_model_path="./model/Qwen2-VL-2B-Instruct"):
        from transformers import AutoProcessor
        from model.qwen_vlm import QwenVLTutor, QwenVLTutorConfig

        print(f"[HF] 加载模型: {model_path}")
        config = QwenVLTutorConfig(
            model_name_or_path=base_model_path,
            use_lora=True,
        )
        self.model = QwenVLTutor(config)
        if os.path.exists(model_path):
            from peft import PeftModel
            self.model.base_model = PeftModel.from_pretrained(
                self.model.base_model, model_path
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()
        self.processor = self.model.processor

    def generate_batch(
        self,
        prompts,
        images=None,
        max_tokens=512,
        temperature=0.0,
        batch_size=8,
        **kwargs,
    ):
        from tqdm import tqdm
        results = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="HF 推理"):
            batch_prompts = prompts[i:i+batch_size]
            batch_images = images[i:i+batch_size] if images else None
            for j, prompt in enumerate(batch_prompts):
                img = batch_images[j] if batch_images else None
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": img} if img else {"type": "text", "text": ""},
                    {"type": "text", "text": prompt},
                ]}]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text],
                    images=[img] if img else None,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                with torch.no_grad():
                    output_ids = self.model.base_model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=(temperature > 0),
                        temperature=temperature if temperature > 0 else None,
                    )
                generated = output_ids[0][inputs.input_ids.shape[1]:]
                results.append(self.processor.decode(generated, skip_special_tokens=True))
        return results


def get_inference_backend(
    model_path,
    base_model_path="./model/Qwen2-VL-2B-Instruct",
    use_vllm=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    **kwargs,
):
    """
    工厂函数：根据环境自动选择 vLLM 或 HF 后端

    Args:
        use_vllm: True=优先 vLLM（不可用时降级到 HF）
        model_path: 模型路径
        base_model_path: 基座模型路径
        tensor_parallel_size: 张量并行数

    Returns:
        VLLMBackend 或 HFBackend 实例
    """
    if use_vllm:
        try:
            return VLLMBackend(
                model_path=model_path,
                base_model_path=base_model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                **kwargs,
            )
        except ImportError as e:
            print(f"[警告] vLLM 不可用: {e}")
            print(f"[降级] 使用 HF Backend")
        except Exception as e:
            print(f"[警告] vLLM 初始化失败: {e}")
            print(f"[降级] 使用 HF Backend")

    return HFBackend(model_path=model_path, base_model_path=base_model_path)


if __name__ == "__main__":
    # 简单测试
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./out/edu_sft")
    parser.add_argument("--base_model", type=str, default="./model/Qwen2-VL-2B-Instruct")
    parser.add_argument("--use_vllm", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()

    backend = get_inference_backend(
        model_path=args.model_path,
        base_model_path=args.base_model,
        use_vllm=bool(args.use_vllm),
        tensor_parallel_size=args.tp,
    )
    print(f"使用后端: {type(backend).__name__}")
    if hasattr(backend, "benchmark"):
        backend.benchmark()

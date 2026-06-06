"""
QwenVL-Tutor LLM-as-Judge 奖励模型

提供 4 种 LLM Reward 后端：
    1. APILLMRewardModel  - 调用闭源 API（GPT-4o / Claude）
    2. LocalVLLMRewardModel - 本地 vLLM 推理（Qwen2.5-72B-AWQ）
    3. LocalHFRewardModel - 本地 HF transformers（兼容性 fallback）
    4. HybridLLMRewardModel - LLM + 规则混合（防 reward hacking）

设计原则:
    - 与 EduRewardModel 保持 API 一致（compute_reward / compute_group_rewards）
    - 支持批量推理（vLLM 加速）
    - 失败时自动降级到规则模型
    - 输出确定性结果（temperature=0）
"""
import os
import re
import json
import time
from typing import List, Optional, Union
from abc import ABC, abstractmethod

import torch


# ============================================================================
# 评分 Prompt 模板
# ============================================================================

JUDGE_PROMPT_TEMPLATE = """你是一位经验丰富的 K12 教育专家，正在评判 AI 助教的回答质量。

【题目】
{question}

【标准答案】
{gt}

【学生回答】
{response}

请按以下 5 个维度严格评分（每项 0-10 分，整数）：

1. **答案准确性**（与标准答案的一致性）
   - 10: 完全正确，含关键推理过程
   - 7-9: 基本正确，关键点都对
   - 4-6: 部分正确，有明显遗漏
   - 1-3: 部分正确，有明显错误
   - 0: 错误或无关

2. **步骤完整性**（是否给出清晰解题步骤）
   - 10: 完整步骤（"第一步...然后...最后..."）
   - 5: 有步骤但不完整
   - 0: 直接给答案无步骤

3. **启发式引导**（是否用引导式提问/启发而非直接给答案）
   - 10: 强烈启发（"你发现什么？""试试看"等）
   - 7-9: 较多引导
   - 4-6: 有少量引导
   - 1-3: 几乎直接给答案
   - 0: 完全直接给答案（"答案是 A"）

4. **语言流畅度**（中文表达质量）
   - 10: 流畅准确，无错别字
   - 5-9: 基本流畅，偶有小问题
   - 0-4: 不流畅或难懂

5. **格式规范性**（是否符合"答案是 X"或"最终答案为 X"格式）
   - 10: 完美格式
   - 0: 无明确答案

请只输出一个 JSON 对象，不要任何解释：
{{"accuracy": X, "completeness": X, "scaffolding": X, "fluency": X, "format": X}}
"""


# ============================================================================
# 抽象基类
# ============================================================================

class BaseLLMRewardModel(ABC):
    """LLM Reward 模型抽象基类"""

    # 5 维权重（可由子类覆盖）
    DEFAULT_WEIGHTS = {
        "accuracy": 0.40,     # 准确性最重要
        "completeness": 0.20,
        "scaffolding": 0.20,  # 引导性
        "fluency": 0.10,
        "format": 0.10,
    }

    def __init__(self, weights: dict = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, \
            f"权重和必须为 1.0，当前: {sum(self.weights.values())}"

    def _make_prompt(self, question: str, response: str, gt: str) -> str:
        """构造评分 prompt"""
        return JUDGE_PROMPT_TEMPLATE.format(
            question=question or "（无题目）",
            response=response or "（无回答）",
            gt=gt or "（无标准答案）",
        )

    def _parse_scores(self, text: str) -> dict:
        """解析 LLM 输出的 JSON 评分"""
        # 尝试提取 JSON
        match = re.search(r'\{[^}]+\}', text)
        if not match:
            return None
        try:
            scores = json.loads(match.group())
            # 验证键
            required = {"accuracy", "completeness", "scaffolding", "fluency", "format"}
            if not required.issubset(scores.keys()):
                return None
            # 限制到 0-10
            for k in required:
                scores[k] = max(0, min(10, float(scores[k])))
            return scores
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    def _aggregate(self, scores: dict) -> float:
        """加权聚合 5 维分数为 0-1 标量 reward"""
        return sum(scores[k] * self.weights[k] for k in self.weights) / 10.0

    @abstractmethod
    def _call_llm(self, prompts: List[str]) -> List[str]:
        """调用 LLM，返回每条 prompt 的输出文本（子类实现）"""
        pass

    def compute_reward(
        self,
        response_text: str,
        gt_text: str = "",
        question: str = "",
    ) -> float:
        """计算单条 reward"""
        scores = self.compute_scores(response_text, gt_text, question)
        if scores is None:
            return 0.0
        return self._aggregate(scores)

    def compute_scores(
        self,
        response_text: str,
        gt_text: str = "",
        question: str = "",
    ) -> Optional[dict]:
        """计算 5 维分数字典（子类或调试用）"""
        prompt = self._make_prompt(question, response_text, gt_text)
        outputs = self._call_llm([prompt])
        if not outputs:
            return None
        return self._parse_scores(outputs[0])

    def compute_group_rewards(
        self,
        response_texts: List[str],
        gt_texts: List[str],
        questions: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """批量计算一组回答的奖励（GRPO 用，K 个候选）"""
        if questions is None:
            questions = [""] * len(response_texts)
        prompts = [self._make_prompt(q, r, g)
                   for q, r, g in zip(questions, response_texts, gt_texts)]
        outputs = self._call_llm(prompts)
        rewards = []
        for out in outputs:
            scores = self._parse_scores(out)
            rewards.append(self._aggregate(scores) if scores else 0.0)
        return torch.tensor(rewards, dtype=torch.float32)


# ============================================================================
# 后端 1: 闭源 API（GPT-4o / Claude / DeepSeek）
# ============================================================================

class APILLMRewardModel(BaseLLMRewardModel):
    """
    调用闭源 API 的 LLM Reward 模型

    支持的 API:
        - OpenAI 兼容: GPT-4o, GPT-4o-mini, DeepSeek-V3, Qwen-API 等
        - Anthropic: Claude-3.5-Sonnet

    使用示例:
        reward = APILLMRewardModel(
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4o-mini",
            base_url=None,  # 默认 OpenAI
        )
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o-mini",
        base_url: str = None,
        weights: dict = None,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        super().__init__(weights=weights)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
        self._init_client()

    def _init_client(self):
        """初始化 OpenAI 兼容客户端"""
        if not self.api_key:
            print("[WARN] APILLMRewardModel: 未提供 API key，将无法工作")
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        except ImportError:
            print("[WARN] openai 包未安装，请运行: pip install openai>=1.0.0")

    def _call_llm(self, prompts: List[str]) -> List[str]:
        """调用 LLM API"""
        if self._client is None:
            return [""] * len(prompts)
        results = []
        for prompt in prompts:
            output = self._call_single(prompt)
            results.append(output)
        return results

    def _call_single(self, prompt: str) -> str:
        """调用单条 prompt（带重试）"""
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,  # 评估需要确定性
                    max_tokens=200,
                    response_format={"type": "json_object"},
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    print(f"[WARN] API 调用失败: {e}")
                    return ""
        return ""


# ============================================================================
# 后端 2: 本地 vLLM（Qwen2.5-72B-AWQ 推荐）
# ============================================================================

class LocalVLLMRewardModel(BaseLLMRewardModel):
    """
    本地 vLLM 推理的 LLM Reward 模型

    推荐模型:
        - Qwen/Qwen2.5-72B-Instruct-AWQ          (24GB 显存)
        - Qwen/Qwen2.5-32B-Instruct-AWQ           (16GB 显存)
        - Qwen/Qwen2.5-14B-Instruct              (28GB 显存)

    使用示例:
        reward = LocalVLLMRewardModel(
            model_path="Qwen/Qwen2.5-72B-Instruct-AWQ",
            tensor_parallel_size=1,
        )
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-72B-Instruct-AWQ",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        weights: dict = None,
    ):
        super().__init__(weights=weights)
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._llm = None
        self._sampling_params = None
        self._init_llm()

    def _init_llm(self):
        """初始化 vLLM"""
        try:
            from vllm import LLM, SamplingParams
            self._llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                trust_remote_code=True,
                enforce_eager=False,
            )
            self._sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=200,
            )
        except ImportError:
            print("[WARN] vllm 包未安装，请运行: pip install vllm>=0.5.0")
        except Exception as e:
            print(f"[WARN] vLLM 初始化失败: {e}")

    def _call_llm(self, prompts: List[str]) -> List[str]:
        """vLLM 批量推理（vLLM 优势：连续批处理）"""
        if self._llm is None:
            return [""] * len(prompts)
        try:
            # vLLM 的 chat 模式（自动应用 tokenizer 的 chat template）
            conversations = [[{"role": "user", "content": p}] for p in prompts]
            outputs = self._llm.chat(
                conversations,
                self._sampling_params,
            )
            return [o.outputs[0].text for o in outputs]
        except Exception as e:
            print(f"[WARN] vLLM 推理失败: {e}")
            return [""] * len(prompts)


# ============================================================================
# 后端 3: 本地 HF transformers（兼容性 fallback）
# ============================================================================

class LocalHFRewardModel(BaseLLMRewardModel):
    """
    本地 HF transformers 推理（兼容性 fallback，无 vLLM 时使用）

    比 vLLM 慢 5-10x，但兼容性最好
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        dtype: str = "bfloat16",
        weights: dict = None,
    ):
        super().__init__(weights=weights)
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self._model = None
        self._tokenizer = None
        self._init_model()

    def _init_model(self):
        """初始化 HF 模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch as _torch
            dtype = getattr(_torch, self.dtype) if isinstance(self.dtype, str) else self.dtype
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map=self.device,
            )
            self._model.eval()
        except ImportError:
            print("[WARN] transformers 包未安装")
        except Exception as e:
            print(f"[WARN] HF 模型初始化失败: {e}")

    def _call_llm(self, prompts: List[str]) -> List[str]:
        """HF 批量推理"""
        if self._model is None:
            return [""] * len(prompts)
        results = []
        try:
            import torch as _torch
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
                with _torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.0,
                        do_sample=False,
                    )
                response = self._tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                results.append(response)
            return results
        except Exception as e:
            print(f"[WARN] HF 推理失败: {e}")
            return [""] * len(prompts)


# ============================================================================
# 后端 4: 混合（LLM 主 + 规则辅，防 reward hacking）
# ============================================================================

class HybridLLMRewardModel(BaseLLMRewardModel):
    """
    混合奖励：LLM 评分 + 规则评分 + 格式分

    防 reward hacking 设计:
        - LLM reward 70%: 语义级质量
        - 规则 reward 20%: 关键词/TF-IDF（防幻觉）
        - 格式 reward 10%: 基础格式（防乱码）

    使用示例:
        from trainer.reward_model import EduRewardModel
        rule_model = EduRewardModel()
        llm = LocalVLLMRewardModel(model_path="Qwen/Qwen2.5-72B-Instruct-AWQ")
        reward = HybridLLMRewardModel(llm_model=llm, rule_model=rule_model)
    """

    def __init__(
        self,
        llm_model: BaseLLMRewardModel,
        rule_model=None,  # 默认使用 EduRewardModel
        llm_weight: float = 0.7,
        rule_weight: float = 0.2,
        format_weight: float = 0.1,
    ):
        # 校验权重
        assert abs(llm_weight + rule_weight + format_weight - 1.0) < 1e-6
        # 构造 LLM 权重（覆盖基类）
        weights = {
            "accuracy": llm_weight * 0.40,
            "completeness": llm_weight * 0.20,
            "scaffolding": llm_weight * 0.20,
            "fluency": llm_weight * 0.10,
            "format": llm_weight * 0.10,
        }
        super().__init__(weights=weights)
        self.llm_model = llm_model
        self.rule_model = rule_model
        self.llm_weight = llm_weight
        self.rule_weight = rule_weight
        self.format_weight = format_weight

        if self.rule_model is None:
            try:
                from trainer.reward_model import EduRewardModel
                self.rule_model = EduRewardModel()
            except ImportError:
                print("[WARN] EduRewardModel 不可用，混合模式将退化为纯 LLM")
                self.rule_weight = 0.0
                self.format_weight = 0.0
                self.llm_weight = 1.0

    def _call_llm(self, prompts: List[str]) -> List[str]:
        return self.llm_model._call_llm(prompts)

    def compute_reward(
        self,
        response_text: str,
        gt_text: str = "",
        question: str = "",
    ) -> float:
        """混合评分"""
        # LLM 主分
        llm_r = super().compute_reward(response_text, gt_text, question)
        # 规则辅分
        rule_r = self.rule_model.compute_reward(response_text, gt_text) if self.rule_model else 0.0
        # 格式分（提取 "答案是 X" 检查）
        fmt_r = self._format_only(response_text)
        # 加权
        return self.llm_weight * llm_r + self.rule_weight * rule_r + self.format_weight * fmt_r

    def _format_only(self, response: str) -> float:
        """仅检查格式规范性"""
        if not response:
            return 0.0
        if re.search(r"答案[是为：:]\s*[A-D\d]", response):
            return 1.0
        if re.search(r"最终答案[为是：:]?\s*[A-D\d]", response):
            return 0.9
        if re.search(r"正确[选项答案][为是：:]?\s*[A-D\d]", response):
            return 0.8
        return 0.3  # 没有明确答案格式

    def compute_group_rewards(
        self,
        response_texts: List[str],
        gt_texts: List[str],
        questions: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """批量混合评分"""
        rewards = []
        for i, (r, g) in enumerate(zip(response_texts, gt_texts)):
            q = questions[i] if questions else ""
            rewards.append(self.compute_reward(r, g, q))
        return torch.tensor(rewards, dtype=torch.float32)


# ============================================================================
# 工厂函数
# ============================================================================

def create_llm_reward(
    backend: str = "auto",
    **kwargs,
) -> BaseLLMRewardModel:
    """
    工厂函数：根据 backend 自动创建 LLM Reward 模型

    Args:
        backend: auto / api / vllm / hf / hybrid
        **kwargs: 透传给对应后端

    Returns:
        BaseLLMRewardModel 实例
    """
    if backend == "auto":
        # 自动选择：优先 vLLM > HF > API
        try:
            import vllm
            backend = "vllm"
        except ImportError:
            try:
                from transformers import AutoModelForCausalLM
                backend = "hf"
            except ImportError:
                backend = "api"
        print(f"[INFO] 自动选择 backend: {backend}")

    if backend == "api":
        return APILLMRewardModel(**kwargs)
    elif backend == "vllm":
        return LocalVLLMRewardModel(**kwargs)
    elif backend == "hf":
        return LocalHFRewardModel(**kwargs)
    elif backend == "hybrid":
        llm_model = kwargs.pop("llm_model", None) or LocalVLLMRewardModel()
        rule_model = kwargs.pop("rule_model", None)
        return HybridLLMRewardModel(llm_model=llm_model, rule_model=rule_model, **kwargs)
    else:
        raise ValueError(f"未知 backend: {backend}")


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    # 简单测试（不调用 LLM）
    print("LLM Reward 模型测试")
    print(f"  默认权重: {BaseLLMRewardModel.DEFAULT_WEIGHTS}")

    # 测试 prompt 构造
    rm = APILLMRewardModel.__new__(APILLMRewardModel)  # 不调用 __init__
    rm.DEFAULT_WEIGHTS = BaseLLMRewardModel.DEFAULT_WEIGHTS
    rm.weights = rm.DEFAULT_WEIGHTS
    prompt = rm._make_prompt(
        question="2+3=?",
        response="想一想，我们先看 2 个苹果，再加 3 个，一共是 5 个。答案是 5。",
        gt="5",
    )
    print(f"\nPrompt 示例（前 200 字）:\n{prompt[:200]}...")

    # 测试 JSON 解析
    test_output = '{"accuracy": 9, "completeness": 8, "scaffolding": 9, "fluency": 8, "format": 9}'
    scores = rm._parse_scores(test_output)
    print(f"\n解析测试: {scores}")
    print(f"聚合 reward: {rm._aggregate(scores):.4f}")

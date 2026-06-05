"""
教育场景奖励模型
多维度评分：答案准确性、步骤完整性、语言流畅度、启发式引导质量
"""
import re
import math
import torch


class EduRewardModel:
    """
    教育 VLM 奖励函数

    评分维度（总分 0~1）：
    1. 答案准确性（0.30）：与标准答案的匹配度
    2. 步骤完整性（0.25）：是否包含解题步骤
    3. 语言流畅度（0.15）：中文表达流畅度
    4. 启发式引导（0.20）：是否使用引导而非直接给答案
    5. 格式规范性（0.10）：输出格式是否符合要求
    """

    # 启发式引导的关键词
    SCAFFOLDING_KEYWORDS = [
        "观察", "思考", "想一想", "你能发现", "注意到",
        "试试看", "你觉得", "第一步", "第二步", "步骤",
        "为什么", "原因是", "我们来", "接下来", "首先",
        "然后", "最后", "所以", "因此", "答案是",
    ]

    # 直接给答案的负面模式
    DIRECT_ANSWER_PATTERNS = [
        r"^[A-D][\.\)）]\s*\n",
        r"^答案[是为]：?[A-D]",
        r"^正确[选项答案]",
        r"^选\s*[A-D]",
    ]

    def __init__(self, tokenizer=None, ideal_length=150, max_length=512):
        self.tokenizer = tokenizer
        self.ideal_length = ideal_length
        self.max_length = max_length

    def _extract_keywords(self, text: str):
        """从文本中提取中文关键词"""
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        english_words = re.findall(r'[a-zA-Z]{2,}', text.lower())
        return set(chinese_words + english_words)

    def _accuracy_score(self, response_text: str, gt_text: str):
        """答案准确性评分：基于关键词重叠率"""
        if not gt_text:
            return 0.5
        gt_kw = self._extract_keywords(gt_text)
        resp_kw = self._extract_keywords(response_text)
        if len(gt_kw) == 0:
            return 0.5
        overlap = len(gt_kw & resp_kw)
        return min(overlap / max(len(gt_kw), 1), 1.0)

    def _completeness_score(self, response_text: str):
        """步骤完整性评分：检查是否包含分步说明"""
        score = 0.0

        # 检查是否包含步骤指示词
        step_indicators = [
            r"第[一二三四五六七八九十\d]+步",
            r"步骤[一二三四五六七八九十\d]+",
            r"首先", r"然后", r"接着", r"最后",
            r"([1-9][\.\)、])",
            r"Step\s*\d+",
        ]
        found_steps = sum(
            1 for pattern in step_indicators
            if re.search(pattern, response_text)
        )

        if found_steps >= 4:
            score = 1.0
        elif found_steps >= 2:
            score = 0.7
        elif found_steps >= 1:
            score = 0.4
        else:
            score = 0.1  # 完全没有步骤，给低分

        return score

    def _fluency_score(self, response_text: str):
        """语言流畅度：基于 bigram 不重复率"""
        chars = list(response_text)
        if len(chars) < 2:
            return 0.0
        bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
        if len(bigrams) == 0:
            return 0.0
        unique_ratio = len(set(bigrams)) / len(bigrams)
        return unique_ratio

    def _scaffolding_score(self, response_text: str):
        """启发式引导评分"""
        score = 0.0

        # 检查是否是直接给答案的风格（扣分）
        is_direct = any(
            re.search(pattern, response_text.strip())
            for pattern in self.DIRECT_ANSWER_PATTERNS
        )
        if is_direct:
            score -= 0.3

        # 检查引导关键词
        scaffolding_count = sum(
            response_text.count(kw) for kw in self.SCAFFOLDING_KEYWORDS
        )

        # 根据文本长度归一化
        text_len = max(len(response_text), 1)
        density = scaffolding_count / (text_len / 50)  # 每50字期望出现的关键词数

        if density > 3:
            score += 0.5
        elif density > 1.5:
            score += 0.35
        elif density > 0.5:
            score += 0.2
        else:
            score += 0.05

        return max(min(score + 0.3, 1.0), 0.0)

    def _format_score(self, response_text: str):
        """格式规范性评分"""
        score = 1.0

        # 过短扣分
        if len(response_text) < 20:
            score -= 0.4

        # 乱码符号过多扣分
        garbage_chars = sum(
            1 for c in response_text if c in '@#$%^&*_+=[]{}|\\`~'
        )
        if len(response_text) > 0:
            garbage_ratio = garbage_chars / len(response_text)
            if garbage_ratio > 0.05:
                score -= 0.3

        # 没有句号/问号结尾扣分
        if response_text.strip() and response_text.strip()[-1] not in '.。!！?？…—)）':
            score -= 0.1

        return max(score, 0.0)

    def compute_reward(
        self,
        response_text: str,
        gt_text: str = "",
    ) -> float:
        """
        计算单条回答的奖励分数

        Args:
            response_text: 模型生成的回答
            gt_text: ground-truth 标准答案

        Returns:
            float: 0~1 之间的奖励分数
        """
        a_score = self._accuracy_score(response_text, gt_text)
        c_score = self._completeness_score(response_text)
        f_score = self._fluency_score(response_text)
        s_score = self._scaffolding_score(response_text)
        fmt_score = self._format_score(response_text)

        total = (
            0.30 * a_score
            + 0.25 * c_score
            + 0.15 * f_score
            + 0.20 * s_score
            + 0.10 * fmt_score
        )
        return total

    def compute_group_rewards(
        self,
        response_texts: list,
        gt_texts: list,
    ) -> torch.Tensor:
        """
        批量计算一组回答的奖励

        Args:
            response_texts: 多个回答的文本列表
            gt_texts: 对应的标准答案列表

        Returns:
            torch.Tensor: shape [K] 的奖励张量
        """
        rewards = []
        for resp_text, gt_text in zip(response_texts, gt_texts):
            r = self.compute_reward(resp_text, gt_text)
            rewards.append(r)
        return torch.tensor(rewards, dtype=torch.float32)


def edu_grpo_advantage(rewards: torch.Tensor, eps: float = 1e-8):
    """
    教育 GRPO 组内优势计算

    Args:
        rewards: shape [K] 的一组奖励
        eps: 数值稳定常数

    Returns:
        advantages: shape [K] 的标准化优势
    """
    mean = rewards.mean()
    std = rewards.std()
    if std < eps:
        return torch.zeros_like(rewards)
    return (rewards - mean) / (std + eps)


def edu_grpo_policy_loss(
    log_probs_old: torch.Tensor,
    log_probs_new: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """
    教育 GRPO 策略梯度损失（PPO-clip 风格）

    Args:
        log_probs_old: 旧策略的 log probs
        log_probs_new: 新策略的 log probs
        advantages: 优势值
        clip_eps: 裁剪范围

    Returns:
        loss: 标量损失
    """
    ratio = torch.exp(log_probs_new - log_probs_old)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss
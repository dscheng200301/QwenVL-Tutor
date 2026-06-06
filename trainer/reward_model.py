"""
教育场景奖励模型
多维度评分：答案准确性、步骤完整性、语言流畅度、启发式引导质量
"""
import re
import math
import torch
from collections import Counter


class EduRewardModel:
    """
    教育 VLM 奖励函数

    评分维度（总分 0~1）：
    1. 答案准确性（0.30）：与标准答案的匹配度（关键词 + 语义相似度）
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

    # 数值类关键词（同义词映射）
    NUMBER_KEYWORDS = {
        "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
        "六": "6", "七": "7", "八": "8", "九": "9", "十": "10",
        "两": "2", "几个": "若干", "多少": "未知数",
        "加倍": "×2", "翻倍": "×2", "减半": "÷2",
    }

    def __init__(self, tokenizer=None, ideal_length=150, max_length=512):
        self.tokenizer = tokenizer
        self.ideal_length = ideal_length
        self.max_length = max_length

    def _normalize_text(self, text: str):
        """文本规范化：统一数字、标点、空白"""
        # 统一数字表示
        for cn_num, ar_num in self.NUMBER_KEYWORDS.items():
            text = text.replace(cn_num, ar_num)
        
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text)
        
        # 去除标点符号（保留中文和英文）
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        return text.strip()

    def _extract_keywords(self, text: str):
        """从文本中提取中文关键词"""
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        english_words = re.findall(r'[a-zA-Z]{2,}', text.lower())
        return set(chinese_words + english_words)

    def _tfidf_similarity(self, text1: str, text2: str):
        """
        TF-IDF 余弦相似度计算
        
        这是一种轻量级的语义相似度方法，
        不需要额外的模型或 API 调用
        """
        # 分词
        words1 = self._tokenize(text1)
        words2 = self._tokenize(text2)
        
        if not words1 or not words2:
            return 0.0
        
        # 构建词频向量
        all_words = list(set(words1) | set(words2))
        if not all_words:
            return 0.0
        
        # 计算 TF
        tf1 = Counter(words1)
        tf2 = Counter(words2)
        
        # 计算 IDF（简化版本）
        idf = {}
        for word in all_words:
            df = (1 if word in tf1 else 0) + (1 if word in tf2 else 0)
            idf[word] = math.log(2.0 / (df + 1)) + 1
        
        # 构建 TF-IDF 向量
        vec1 = [tf1.get(w, 0) * idf[w] for w in all_words]
        vec2 = [tf2.get(w, 0) * idf[w] for w in all_words]
        
        # 计算余弦相似度
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        norm1 = math.sqrt(sum(v ** 2 for v in vec1))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def _tokenize(self, text: str):
        """简单分词：中文按字符，英文按单词"""
        # 提取中文字符序列
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
        # 提取英文单词
        english_words = re.findall(r'[a-zA-Z]+', text.lower())
        # 提取数字
        numbers = re.findall(r'\d+\.?\d*', text)
        
        # 组合
        tokens = []
        for chars in chinese_chars:
            tokens.extend(list(chars))  # 中文字符逐个分词
        tokens.extend(english_words)
        tokens.extend(numbers)
        
        return tokens

    def _accuracy_score(self, response_text: str, gt_text: str):
        """
        答案准确性评分：结合关键词重叠率和语义相似度
        
        综合评分 = 0.5 * 关键词重叠率 + 0.5 * TF-IDF 相似度
        """
        if not gt_text:
            return 0.0
        
        # 规范化文本
        gt_normalized = self._normalize_text(gt_text)
        resp_normalized = self._normalize_text(response_text)
        
        # 1. 关键词重叠率（原有逻辑）
        gt_kw = self._extract_keywords(gt_text)
        resp_kw = self._extract_keywords(response_text)
        
        if len(gt_kw) == 0:
            keyword_score = 0.5
        else:
            overlap = len(gt_kw & resp_kw)
            keyword_score = min(overlap / max(len(gt_kw), 1), 1.0)
        
        # 2. TF-IDF 语义相似度（新增）
        semantic_score = self._tfidf_similarity(gt_normalized, resp_normalized)
        
        # 3. 数值答案精确匹配（针对数学题）
        number_score = self._number_match_score(response_text, gt_text)
        
        # 综合评分
        combined_score = 0.4 * keyword_score + 0.4 * semantic_score + 0.2 * number_score
        
        return min(combined_score, 1.0)

    def _number_match_score(self, response_text: str, gt_text: str):
        """
        数值答案匹配评分
        
        对于数学题，数值答案的精确匹配很重要
        """
        # 提取数字
        resp_numbers = set(re.findall(r'\d+\.?\d*', response_text))
        gt_numbers = set(re.findall(r'\d+\.?\d*', gt_text))
        
        if not gt_numbers:
            return 0.5
        
        # 完全匹配
        if resp_numbers == gt_numbers:
            return 1.0
        
        # 部分匹配
        overlap = len(resp_numbers & gt_numbers)
        union = len(resp_numbers | gt_numbers)
        
        if union == 0:
            return 0.5
        
        jaccard = overlap / union
        
        # 如果有交集但不完全匹配，检查数值大小是否接近
        if overlap > 0:
            # 检查是否有非常接近的数值（允许 5% 误差）
            for resp_num in resp_numbers:
                for gt_num in gt_numbers:
                    try:
                        r = float(resp_num)
                        g = float(gt_num)
                        if g != 0 and abs(r - g) / g < 0.05:
                            return 0.7  # 接近但不完全匹配
                    except ValueError:
                        continue
        
        return jaccard

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
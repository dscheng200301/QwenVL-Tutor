"""
教育图文数据集
支持 SFT / DPO / GRPO 三种训练阶段的数据加载

数据格式（Parquet）：
    conversations: JSON string, 对话列表
    image_bytes: bytes or list[bytes], 图像二进制数据
    subject: str (可选), 学科标签
    grade_level: str (可选), 学段
    question_type: str (可选), 题型 (choice/fill/calc)
    difficulty: int (可选), 难度 1-5
"""
import io
import json
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
import pyarrow as pa
import pyarrow.parquet as pq


# ============================
# 教育场景 System Prompts
# ============================
EDU_SYSTEM_PROMPTS = [
    "你是一位耐心的数学老师，请一步步引导孩子思考，用简单易懂的方式解答题目。",
    "你是一位和蔼的科学老师，先用生活中的例子帮助孩子理解概念，再给出答案和解析。",
    "你是一位小学辅导老师，看到题目后先鼓励孩子，然后分步骤讲解，最后给出正确答案。",
    "你是一位会启发式教学的老师，不要直接告诉答案，而是通过提问引导孩子自己找到解题思路。",
    "你是一位中学理科老师，请给出清晰的解题步骤，每一步都解释为什么这样做。",
]

# 启发式引导风格（用于 DPO/GRPO 偏好对齐）
SCAFFOLDING_PROMPT = (
    "请用启发式的方式解答这道题：先引导孩子观察题目中的关键信息，"
    "然后通过提问帮助他们思考解题思路，最后再给出答案和完整解析。"
    "记住：目标是教会孩子方法，而不仅仅是得到答案。"
)

DIRECT_ANSWER_PROMPT = (
    "请直接给出这道题的正确选项和简单解释。"
)


def preprocess_image(image_bytes):
    """预处理图像：解码为 PIL Image"""
    if isinstance(image_bytes, list):
        image_bytes = image_bytes[0]
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


class EduDataset(Dataset):
    """
    教育 SFT 数据集
    用于监督微调阶段，训练模型的基础做题能力

    每条样本包含：
        - 题目图像
        - 标准对话（user 提问 + assistant 回答）
    """

    def __init__(
        self,
        parquet_path: str,
        processor,
        max_length: int = 2048,
        add_system_ratio: float = 1.0,
        use_scaffolding: bool = False,
    ):
        """
        Args:
            parquet_path: Parquet 数据文件路径
            processor: Qwen2VLProcessor
            max_length: 最大 token 长度
            add_system_ratio: 添加 system prompt 的概率
            use_scaffolding: 是否使用启发式引导风格
        """
        super().__init__()
        self.table = pa.Table.from_batches(
            pq.ParquetFile(parquet_path).iter_batches()
        )
        self.processor = processor
        self.max_length = max_length
        self.add_system_ratio = add_system_ratio
        self.use_scaffolding = use_scaffolding

        # 缓存 processor 的 tokenizer
        self.tokenizer = processor.tokenizer

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index: int):
        row = self.table.slice(index, 1)
        conversations = json.loads(row['conversations'][0].as_py())

        # 检测是否包含图像数据（根据 conversations 中的 <image> 标记判断）
        has_image = any("<image>" in turn.get("content", "") for turn in conversations)

        # 纯文本数据用极小的占位图替代真实大图，避免浪费 GPU 内存
        if has_image:
            image_bytes = row['image_bytes'][0].as_py()
            image = preprocess_image(image_bytes)
        else:
            image = Image.new("RGB", (8, 8), color=(255, 255, 255))

        # 添加 system prompt
        if random.random() < self.add_system_ratio:
            system_msg = {
                "role": "system",
                "content": random.choice(EDU_SYSTEM_PROMPTS),
            }
            if conversations[0].get("role") != "system":
                conversations = [system_msg] + conversations

        # 转换对话格式为 Qwen2-VL 格式
        messages = []
        for turn in conversations:
            role = turn["role"]
            content = turn["content"]
            if role == "user" and "<image>" in content:
                # 分离文本和图像
                text_content = content.replace("<image>", "").strip()
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text_content or "请解答这道题"},
                    ],
                })
            elif role == "user":
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": content}],
                })
            else:
                messages.append({"role": role, "content": content})

        # 使用 processor 处理
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # 生成 labels
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()
        # 将 user 和 system 部分的 token 设为 -100
        # 简化处理：找到第一个 "assistant" 回复开始位置
        assistant_start = self._find_assistant_start(text)
        if assistant_start is not None:
            # 重新 tokenize 到 assistant 开始位置
            prefix_text = text[:assistant_start]
            prefix_ids = self.tokenizer(prefix_text, add_special_tokens=False).input_ids
            prefix_len = len(prefix_ids)
            labels[:prefix_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": inputs["pixel_values"][0],
            "image_grid_thw": inputs["image_grid_thw"][0],
            "labels": labels,
        }

    def _find_assistant_start(self, text: str):
        """找到 assistant 回复在文本中的起始位置"""
        markers = ["\nassistant\n", "\nassistant:", "assistant\n"]
        for marker in markers:
            idx = text.find(marker)
            if idx != -1:
                return idx + len(marker)
        return None


class EduGRPODataset(Dataset):
    """
    教育 GRPO 数据集
    仅提供 prompt + 图像 + ground-truth，由训练脚本自行生成候选回答
    """

    def __init__(
        self,
        parquet_path: str,
        processor,
        max_length: int = 2048,
    ):
        super().__init__()
        self.table = pa.Table.from_batches(
            pq.ParquetFile(parquet_path).iter_batches()
        )
        self.processor = processor
        self.max_length = max_length
        self.tokenizer = processor.tokenizer

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index: int):
        row = self.table.slice(index, 1)
        conversations = json.loads(row['conversations'][0].as_py())
        image_bytes = row['image_bytes'][0].as_py()
        image = preprocess_image(image_bytes)

        # 构建 prompt（仅 user + system 部分）
        prompt_messages = []
        for turn in conversations:
            if turn["role"] == "system":
                prompt_messages.append({
                    "role": "system",
                    "content": random.choice(EDU_SYSTEM_PROMPTS),
                })
            elif turn["role"] == "user":
                content_text = turn["content"].replace("<image>", "").strip()
                prompt_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": content_text or "请解答这道题"},
                    ],
                })
            elif turn["role"] == "assistant":
                break

        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # 提取 ground-truth 回答
        gt_response = ""
        for turn in conversations:
            if turn["role"] == "assistant":
                gt_response = turn["content"]
                break

        gt_ids = self.tokenizer(
            gt_response, add_special_tokens=False,
            return_tensors="pt", padding="max_length",
            max_length=self.max_length // 2, truncation=True
        ).input_ids[0]

        return {
            "prompt_input_ids": prompt_inputs["input_ids"][0],
            "prompt_attention_mask": prompt_inputs["attention_mask"][0],
            "pixel_values": prompt_inputs["pixel_values"][0],
            "image_grid_thw": prompt_inputs["image_grid_thw"][0],
            "gt_response_ids": gt_ids,
        }
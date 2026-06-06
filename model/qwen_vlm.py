"""
QwenVL-Tutor 核心模型定义
基于 Qwen3-VL 可插拔基座封装，适配亲子教育场景
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass


@dataclass
class QwenVLTutorOutput(ModelOutput):
    """QwenVL-Tutor 模型输出"""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    hidden_states: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None


class QwenVLTutorConfig(PretrainedConfig):
    """QwenVL-Tutor 模型配置，兼容 Qwen2-VL 配置"""
    model_type = "QwenVL-Tutor"

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct",
        use_lora: bool = True,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        freeze_vision_tower: bool = False,
        freeze_llm: bool = False,
        max_seq_len: int = 2048,
        image_token_len: int = 64,
        system_prompt: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name_or_path = model_name_or_path
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        self.freeze_vision_tower = freeze_vision_tower
        self.freeze_llm = freeze_llm
        self.max_seq_len = max_seq_len
        self.image_token_len = image_token_len
        self.system_prompt = system_prompt


class QwenVLTutor(nn.Module):
    """
    亲子教育 VLM 模型
    封装 Qwen2-VL，支持 LoRA 微调 + 多阶段训练

    使用方法:
        model = QwenVLTutor.from_pretrained("./model/Qwen2-VL-2B-Instruct")
        # 默认启用 LoRA，只训练 adapter 参数
    """

    def __init__(self, config: QwenVLTutorConfig):
        super().__init__()
        self.config = config

        # 加载基座模型
        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.processor = Qwen2VLProcessor.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
        )

        # 冻结策略
        if config.freeze_vision_tower:
            self._freeze_vision_tower()
        if config.freeze_llm:
            self._freeze_language_model()

        # LoRA 微调
        if config.use_lora:
            self._apply_lora()

    def _freeze_vision_tower(self):
        """冻结视觉编码器"""
        if hasattr(self.base_model, 'visual'):
            for param in self.base_model.visual.parameters():
                param.requires_grad = False

    def _freeze_language_model(self):
        """冻结语言模型"""
        if hasattr(self.base_model, 'model'):
            for param in self.base_model.model.parameters():
                param.requires_grad = False

    def _apply_lora(self):
        """应用 LoRA 适配器"""
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )
        self.base_model = get_peft_model(self.base_model, lora_config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> QwenVLTutorOutput:
        """
        前向传播

        Args:
            input_ids: [batch_size, seq_len] 输入 token ids
            attention_mask: [batch_size, seq_len] 注意力掩码
            pixel_values: 图像像素值（Qwen3-VL 格式）
            image_grid_thw: 图像网格信息
            labels: [batch_size, seq_len] 标签（-100 表示忽略）

        Returns:
            QwenVLTutorOutput
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            **kwargs,
        )

        return QwenVLTutorOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
        )

    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ):
        """
        计算 token 级对数概率（用于 DPO/GRPO）

        Args:
            input_ids: 完整输入序列
            labels: 标签（-100 表示忽略）
            pixel_values: 图像像素值
            image_grid_thw: 图像网格信息
            temperature: 温度参数

        Returns:
            seq_log_probs: [batch_size] 每条序列的对数概率和
            seq_lens: [batch_size] 每条序列的有效 token 数
            outputs: 模型输出
        """
        outputs = self.forward(
            input_ids=input_ids,
            labels=None,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        logits = outputs.logits / temperature
        log_probs = F.log_softmax(logits, dim=-1)
        shift_log_probs = log_probs[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        per_token_log_probs = torch.gather(
            shift_log_probs, dim=2, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        mask = (shift_labels != -100).float()
        seq_log_probs = (per_token_log_probs * mask).sum(dim=-1)
        return seq_log_probs, mask.sum(dim=-1), outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        生成回答

        Args:
            input_ids: 输入 token ids
            pixel_values: 图像像素值
            image_grid_thw: 图像网格信息
            max_new_tokens: 最大生成 token 数
            temperature: 温度
            top_p: nucleus 采样阈值
            top_k: top-k 采样
            do_sample: 是否采样

        Returns:
            generated_ids: 生成的 token ids
        """
        with torch.no_grad():
            generated_ids = self.base_model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **kwargs,
            )
        return generated_ids

    def save_pretrained(self, save_path: str):
        """保存模型权重"""
        import os
        os.makedirs(save_path, exist_ok=True)
        self.base_model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """从预训练权重加载"""
        config = QwenVLTutorConfig(model_name_or_path=model_path, **kwargs)
        model = cls(config)
        return model

    @staticmethod
    def process_image(image, processor):
        """
        处理单张图像

        Args:
            image: PIL.Image 或图像路径
            processor: Qwen2VLProcessor

        Returns:
            dict: 包含 pixel_values 和 image_grid_thw
        """
        from PIL import Image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        return inputs
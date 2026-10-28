from __future__ import annotations

import logging
import time
from typing import Optional

import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

log = logging.getLogger(__name__)


class VisionLM:
    """One loaded checkpoint plus its processor.

    ``vision_per_conversation`` reproduces how each original script collected
    images for a batch: Subtask 1 called ``process_vision_info`` once per
    conversation, Subtask 2 once for the whole batch. Both yield the same
    image list; the switch only keeps the call pattern identical.
    """

    def __init__(self, model, processor, *, chat_template_kwargs: Optional[dict] = None,
                 vision_per_conversation: bool = True):
        self.model = model
        self.processor = processor
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.vision_per_conversation = vision_per_conversation
        # Left padding so every sequence ends in a real token during batched generation.
        self.processor.tokenizer.padding_side = "left"

    def encode(self, conversations: list[list[dict]]):
        texts = [
            self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs
            )
            for conv in conversations
        ]
        if self.vision_per_conversation:
            images, videos = [], []
            for conv in conversations:
                img, vid = process_vision_info(conv)
                images.extend(img or [])
                videos.extend(vid or [])
            images, videos = images or None, videos or None
        else:
            images, videos = process_vision_info(conversations)
        return self.processor(
            text=texts, images=images, videos=videos, padding=True, return_tensors="pt",
        ).to(self.model.device)

    def generate(self, inputs, *, max_new_tokens: int, repetition_penalty: float,
                 sample: bool = False, temperature: float = 0.7,
                 top_p: float = 0.9) -> list[str]:
        """Greedy decoding (``do_sample=False``) unless ``sample`` is set."""
        kwargs = dict(**inputs, max_new_tokens=max_new_tokens,
                      repetition_penalty=repetition_penalty, do_sample=sample)
        if sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        else:
            # Unset the checkpoint's sampling defaults so greedy decoding is clean.
            kwargs["temperature"] = None
            kwargs["top_p"] = None
            kwargs["top_k"] = None
        with torch.inference_mode():
            generated = self.model.generate(**kwargs)
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )


def load_vlm(model_path: str, device: str, *, chat_template_kwargs: Optional[dict] = None,
             vision_per_conversation: bool = True) -> VisionLM:
    """Load a dense or mixture-of-experts Qwen3-VL checkpoint in bfloat16."""
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    arch = list(getattr(AutoConfig.from_pretrained(model_path, trust_remote_code=True),
                        "architectures", None) or [])
    cls = (Qwen3VLMoeForConditionalGeneration if any("Moe" in a for a in arch)
           else Qwen3VLForConditionalGeneration)
    model = cls.from_pretrained(model_path, device_map=device, trust_remote_code=True,
                                torch_dtype=torch.bfloat16)
    model.eval()
    log.info(f"Loaded {model_path} ({cls.__name__}) on {device} in {time.time() - t0:.1f}s")
    return VisionLM(model, processor, chat_template_kwargs=chat_template_kwargs,
                    vision_per_conversation=vision_per_conversation)


def free_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

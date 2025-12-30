import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from typing import Any, Dict, List, Optional, Tuple, Union


import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_modeling

class SketchDecoder(nn.Module):
    """
    Autoregressive generative model wrapper for Qwen2.5-VL
    """

    def __init__(self,
                 pix_len,
                 text_len,
                 model_path="Qwen/Qwen2.5-VL-3B-Instruct", 
                 **kwargs):
        super().__init__()
        
        self.pix_len = pix_len
        self.text_len = text_len
        
        self.vocab_size = 197000
        self.bos_token_id = 196998
        self.eos_token_id = 196999
        self.pad_token_id = 151643
        
        print(f"Loading model from {model_path}...")
        
        # 加载配置
        config = AutoConfig.from_pretrained(
            model_path,
            vocab_size=self.vocab_size,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            trust_remote_code=True
        )

        self.transformer = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            #attn_implementation="flash_attention_2", 
            #device_map ="cuda",
            ignore_mismatched_sizes=True
        )

        self.transformer.resize_token_embeddings(self.vocab_size)
        
        self.train()

    def forward(self, 
                    input_ids=None,
                    attention_mask=None,
                    pixel_values=None,
                    image_grid_thw=None,
                    labels=None,
                    past_key_values=None,
                    use_cache=False,
                    **kwargs):
            
            target_device = self.transformer.device 
            
            if input_ids is not None:
                input_ids = input_ids.to(target_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(target_device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(target_device)
                if self.transformer.dtype != pixel_values.dtype:
                    pixel_values = pixel_values.to(self.transformer.dtype)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(target_device)
            if labels is not None:
                labels = labels.to(target_device)
            
            self.transformer.rope_deltas = None
            position_ids, _ = self.transformer.get_rope_index(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw
            )
            position_ids = position_ids * attention_mask[None, ]

            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=True
            )
            return Qwen2_5_VLCausalLMOutputWithPast(
                loss=outputs.loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=getattr(outputs, 'rope_deltas', None)
            )
        
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
pip install pillow

# Tested on 8x H100 GPUs
accelerate launch
    --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir sft-llava-1.5-7b-hf \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing

For LLaVA-NeXT, use: (requires transformers>=4.45)
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf

For meta-llama/Llama-3.2-11B-Vision-Instruct, use: (requires transformers>=4.45.1)
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct
"""

import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets

from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration
from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, LoraConfig

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    print("model config: " + str(model_config))
    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    
    #quantization_config = BitsAndBytesConfig(load_in_8bit=True)    #leads to error -- can't be purely quantized. need lora
    quantization_config = get_quantization_config(model_config)
    print("quantization config: " +  str(quantization_config))
    
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )

#     class ImageCaptioningDataset(Dataset):
#     def __init__(self, dataset, processor):
#         self.dataset = dataset
#         self.processor = processor

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         x = self.dataset[idx]
#         encoding = self.processor(images=x["image"], padding="max_length", return_tensors="pt").to(device)
#         encoding = {k: v.squeeze() for k, v in encoding.items()}
#         encoding["text"] = " ".join(x["captions"])
#         return encoding
        
# def collate_fn(batch):
#     processed_batch = {}
#     for key in batch[0].keys():
#         if key != "text":
#             processed_batch[key] = torch.stack([example[key] for example in batch])
#         else:
#             text_inputs = processor.tokenizer(
#                 [example["text"] for example in batch], padding=True, return_tensors="pt"
#             )
#             processed_batch["input_ids"] = text_inputs["input_ids"]
#             processed_batch["attention_mask"] = text_inputs["attention_mask"]
#     return processed_batch
    
    ################
    # Create a data collator to encode text and image pairs
    ################
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        #print("examples: " + str(examples))
        pixel_values = []

        for example in examples:
            encoding = processor(images=example["image"], padding="max_length", return_tensors="pt")
            encoding = {k: v.squeeze() for k, v in encoding.items()}
            #print("pixel values shape: " + str((encoding["pixel_values"].shape)))
            #print("pixel values: " + str(pixel_values))
            pixel_values.append(encoding["pixel_values"])
            
        text_inputs = processor.tokenizer(
                [" ".join(example["captions"]) for example in examples], padding=True, return_tensors="pt"
            )
        
        processed_batch = {}
        processed_batch["input_ids"] = text_inputs["input_ids"]
        #processed_batch["attention_mask"] = text_inputs["attention_mask"]
        processed_batch["labels"] = text_inputs["input_ids"]
        processed_batch["pixel_values"] = torch.stack(pixel_values)
        #print("final pixel values shape: " + str(processed_batch["pixel_values"].shape))
       
        
   
        return processed_batch

    ################
    # Dataset
    ################
    evalDataset = load_from_disk(script_args.dataset_name+"/test")
    trainDataset = load_from_disk(script_args.dataset_name+"/train")

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=trainDataset,
        eval_dataset=evalDataset,
        processing_class=processor.tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)
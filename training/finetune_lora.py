"""
Fine-tunes LFM2-VL on agricultural satellite imagery for crop stress classification.
Optimized for Google Colab A100 (40GB VRAM) within 3 hours.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import yaml
from datasets import Dataset as HFDataset
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Setup Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "finetune.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("xyralis.finetune")

# ── Custom Dataset ─────────────────────────────────────────────────────────────
class CropStressDataset(Dataset):
    def __init__(self, jsonl_path: str, processor: AutoProcessor, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length
        self.samples = []
        
        if not os.path.exists(jsonl_path):
             logger.error("Dataset file missing: %s", jsonl_path)
             return
             
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # We might skip negative examples without images for vision-language finetuning
                    # Or we handle them. For LFM2-VL, if image is missing, we might use a dummy or skip.
                    # The prompt says: "Handles missing images gracefully (skip + log)"
                    img_path = data.get("image", "")
                    if not img_path or not os.path.exists(img_path):
                        logger.debug("Skipping missing/empty image path: %s", img_path)
                        continue
                    
                    self.samples.append(data)
                except Exception as e:
                    logger.debug("Error parsing line: %s", e)

        logger.info("Loaded %d valid samples from %s", len(self.samples), jsonl_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        try:
            image = Image.open(item["image"]).convert("RGB")
        except Exception as e:
            logger.warning("Failed to open image %s, returning random sample. Error: %s", item["image"], e)
            return self.__getitem__(np.random.randint(0, len(self)))

        # Format prompt according to model's expected conversational format
        prompt_text = f"{item['system']}\n\nUSER: <image>\n{item['prompt']}\nASSISTANT: "
        response_text = item['response']
        full_text = prompt_text + response_text + self.processor.tokenizer.eos_token

        # Tokenize and process
        inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        pixel_values = inputs["pixel_values"][0] if "pixel_values" in inputs else None
        spatial_shapes = inputs["spatial_shapes"][0] if "spatial_shapes" in inputs else None
        pixel_attention_mask = inputs["pixel_attention_mask"][0] if "pixel_attention_mask" in inputs else None
        
        labels = input_ids.clone()
        
        # 1. Mask the prompt tokens (accurate method)
        prompt_inputs = self.processor.tokenizer(
            text=prompt_text,
            return_tensors="pt",
            add_special_tokens=True # Full text has BOS, prompt masking must too
        )
        prompt_len = len(prompt_inputs["input_ids"][0])
        labels[:prompt_len] = -100

        # 2. Mask padding tokens
        labels[attention_mask == 0] = -100

        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        if pixel_values is not None:
             res["pixel_values"] = pixel_values
        if spatial_shapes is not None:
             res["spatial_shapes"] = spatial_shapes
        if pixel_attention_mask is not None:
             res["pixel_attention_mask"] = pixel_attention_mask
             
        return res

def collate_fn(batch):
    # Dataloader padding is handled by the Dataset max_length padding above.
    # We just stack them here.
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    res = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
    if "pixel_values" in batch[0]:
        res["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])
    if "spatial_shapes" in batch[0]:
        res["spatial_shapes"] = torch.stack([item["spatial_shapes"] for item in batch])
    if "pixel_attention_mask" in batch[0]:
        res["pixel_attention_mask"] = torch.stack([item["pixel_attention_mask"] for item in batch])
        
    return res

# Global processor to be used in compute_metrics
processor = None

# ── Custom Callbacks & Metrics ────────────────────────────────────────────────
class LossLoggingCallback(TrainerCallback):
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_data = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.log_data.append({
                "step": state.global_step,
                "loss": logs.get("loss", None),
                "eval_loss": logs.get("eval_loss", None)
            })
            if state.global_step % 10 == 0:
                with open(self.log_path, "w") as f:
                    json.dump(self.log_data, f, indent=2)

def compute_metrics(eval_pred):
    """
    Computes accuracy by decoding predictions and labels, then parsing
    the classification as requested.
    """
    global processor
    predictions, labels = eval_pred
    # predictions: [batch, seq, vocab]
    preds = np.argmax(predictions, axis=-1)
    
    # We only care about the part of the labels that isn't -100
    y_pred_text = []
    y_true_text = []
    
    for i in range(len(labels)):
        # Decode only the response part (labels != -100)
        mask = labels[i] != -100
        true_ids = labels[i][mask]
        pred_ids = preds[i][mask]
        
        true_str = processor.decode(true_ids, skip_special_tokens=True)
        pred_str = processor.decode(pred_ids, skip_special_tokens=True)
        
        # Ensure we can parse "Classification: X"
        y_true_text.append(parse_prediction(true_str)[0])
        y_pred_text.append(parse_prediction(pred_str)[0])
    
    acc = accuracy_score(y_true_text, y_pred_text)
    return {"accuracy": float(acc)}

# ── Evaluation ────────────────────────────────────────────────────────────────
def parse_prediction(text: str) -> Tuple[str, float]:
    """Extracts classification and confidence from generated text."""
    cls_match = re.search(r"Classification:\s*([a-zA-Z_]+)", text, re.IGNORECASE)
    conf_match = re.search(r"Confidence:\s*(\d+)%", text, re.IGNORECASE)
    
    pred_cls = cls_match.group(1).lower() if cls_match else "unknown"
    # normalize
    if "healthy" in pred_cls: pred_cls = "healthy"
    elif "severe" in pred_cls: pred_cls = "severe_stress"
    elif "mild" in pred_cls: pred_cls = "mild_stress"
    
    conf = float(conf_match.group(1)) if conf_match else 0.0
    return pred_cls, conf

def evaluate_model(model, processor, test_jsonl: str, model_label: str, output_dir: str):
    logger.info("Evaluating %s...", model_label)
    
    # We evaluate sequentially to avoid OOM
    model.eval()
    
    y_true = []
    y_pred = []
    confidences = []
    
    # Map classes to standard naming
    cls_map = {"healthy": "healthy", "mild_stress": "mild_stress", "severe_stress": "severe_stress"}
    
    correct_conf = []
    incorrect_conf = []

    with open(test_jsonl, "r") as f:
        samples = [json.loads(line) for line in f]

    for item in samples:
        img_path = item.get("image", "")
        if not img_path or not os.path.exists(img_path): continue
        
        true_cls = item.get("label", "").lower()
        if true_cls not in cls_map: continue
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            continue
            
        prompt_text = f"{item['system']}\n\nUSER: <image>\n{item['prompt']}\nASSISTANT: Classification:"
        
        inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False
            )
            
        generated_text = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # Prepend what we forced to parse correctly
        full_gen = "Classification:" + generated_text
        
        pred_cls, conf = parse_prediction(full_gen)
        
        y_true.append(cls_map[true_cls])
        y_pred.append(pred_cls)
        confidences.append(conf)
        
        if pred_cls == cls_map[true_cls]:
            correct_conf.append(conf)
        else:
            incorrect_conf.append(conf)

    # Compute metrics
    labels_list = ["healthy", "mild_stress", "severe_stress"]
    acc = accuracy_score(y_true, y_pred)
    f1_per_class = f1_score(y_true, y_pred, labels=labels_list, average=None)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    
    mean_conf_correct = np.mean(correct_conf) if correct_conf else 0
    mean_conf_incorrect = np.mean(incorrect_conf) if incorrect_conf else 0
    calibration_gap = abs(mean_conf_correct - (acc * 100)) # Simple metric
    
    metrics = {
        "accuracy": float(acc),
        "f1_healthy": float(f1_per_class[0]),
        "f1_mild_stress": float(f1_per_class[1]),
        "f1_severe_stress": float(f1_per_class[2]),
        "f1_weighted": float(f1_weighted),
        "calibration_gap": float(calibration_gap),
        "confusion_matrix": cm.tolist()
    }
    
    out_path = Path(output_dir) / f"eval_{model_label}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    logger.info("Evaluation for %s complete. Acc: %.2f%%", model_label, acc * 100)
    return metrics

# ── Main Script ────────────────────────────────────────────────────────────────
def main():
    global processor
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="LiquidAI/LFM2-VL-3B")
    parser.add_argument("--train_jsonl", default="data/dataset/train.jsonl")
    parser.add_argument("--val_jsonl", default="data/dataset/val.jsonl")
    parser.add_argument("--test_jsonl", default="data/dataset/test.jsonl")
    parser.add_argument("--hf_username", default=None)
    parser.add_argument("--output_dir", default="training/weights")
    parser.add_argument("--skip_base_eval", action="store_true")
    args = parser.parse_args()

    # Load config
    config_path = "training/configs/lora_config.yaml"
    with open(config_path, "r") as f:
         config = yaml.safe_load(f)

    logger.info("=== Starting LFM2-VL Fine-tuning Pipeline ===")

    # 1. Base Model Evaluation
    if not args.skip_base_eval:
        logger.info("Loading Base Model for Evaluation (CPU)...")
        base_processor = AutoProcessor.from_pretrained(args.model_name)
        base_model = AutoModelForMultimodalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        base_metrics = evaluate_model(base_model, base_processor, args.test_jsonl, "base", "training/results")
        
        # Free memory
        del base_model
    else:
        base_metrics = {"accuracy": 0, "f1_healthy": 0, "f1_mild_stress": 0, "f1_severe_stress": 0, "f1_weighted": 0, "calibration_gap": 0}

    # 2. Loading Model for Training
    logger.info("Loading Model for LoRA Fine-tuning (CPU)...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    processor.tokenizer.padding_side = "right"
    
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # model = prepare_model_for_kbit_training(model)
    # model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias="none",
        task_type=config["lora"]["task_type"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    # 3. Datasets
    logger.info("Preparing Datasets...")
    train_dataset = CropStressDataset(args.train_jsonl, processor, max_length=config["dataset"]["max_length"])
    val_dataset = CropStressDataset(args.val_jsonl, processor, max_length=config["dataset"]["max_length"])

    # 4. Training
    logger.info("Configuring Trainer (CPU Mode)...")
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_ratio=config["training"]["warmup_ratio"],
        bf16=False,
        fp16=False,
        eval_strategy=config["training"]["eval_strategy"],
        eval_steps=config["training"]["eval_steps"],
        save_strategy=config["training"]["save_strategy"],
        save_steps=config["training"]["save_steps"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        logging_steps=config["training"]["logging_steps"],
        dataloader_drop_last=config["training"]["dataloader_drop_last"],
        dataloader_num_workers=config["dataset"]["num_workers"],
        dataloader_pin_memory=config["dataset"]["pin_memory"],
        eval_accumulation_steps=10, # Prevent OOM by offloading predictions periodically
        report_to="none", # Disable wandb for this run
        use_cpu=True,
        resume_from_checkpoint=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[
            LossLoggingCallback(log_path="training/results/loss_curve.json")
        ]
    )

    logger.info("Starting Training...")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    resume = True if os.path.exists(checkpoint_dir) and any(d.startswith("checkpoint") for d in os.listdir(checkpoint_dir)) else False
    
    if resume:
        logger.info(f"Resuming from latest checkpoint in {checkpoint_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        logger.info("No checkpoints found. Starting fresh training.")
        trainer.train()

    # 5. Save LoRA Adapter
    lora_out = os.path.join(args.output_dir, "xyralis-lora")
    trainer.model.save_pretrained(lora_out)
    processor.save_pretrained(lora_out)
    logger.info("Saved LoRA adapter to %s", lora_out)

    # 6. Fine-tuned Evaluation
    ft_metrics = evaluate_model(trainer.model, processor, args.test_jsonl, "ft", "training/results")

    # Update README
    readme_path = "training/results/README.md"
    with open(readme_path, "r") as f:
        readme_content = f.read()

    table_data = f"| Accuracy        | {base_metrics['accuracy']*100:.1f}%       | {ft_metrics['accuracy']*100:.1f}%        | +{(ft_metrics['accuracy']-base_metrics['accuracy'])*100:.1f}% |\n"
    table_data += f"| F1 healthy      | {base_metrics['f1_healthy']:.2f}         | {ft_metrics['f1_healthy']:.2f}         | +{(ft_metrics['f1_healthy']-base_metrics['f1_healthy']):.2f} |\n"
    table_data += f"| F1 mild_stress  | {base_metrics['f1_mild_stress']:.2f}         | {ft_metrics['f1_mild_stress']:.2f}         | +{(ft_metrics['f1_mild_stress']-base_metrics['f1_mild_stress']):.2f} |\n"
    table_data += f"| F1 severe_stress| {base_metrics['f1_severe_stress']:.2f}         | {ft_metrics['f1_severe_stress']:.2f}         | +{(ft_metrics['f1_severe_stress']-base_metrics['f1_severe_stress']):.2f} |\n"
    table_data += f"| Weighted F1     | {base_metrics['f1_weighted']:.2f}         | {ft_metrics['f1_weighted']:.2f}         | +{(ft_metrics['f1_weighted']-base_metrics['f1_weighted']):.2f} |\n"
    table_data += f"| Conf Calibration| {base_metrics['calibration_gap']:.1f}         | {ft_metrics['calibration_gap']:.1f}         | {-(ft_metrics['calibration_gap']-base_metrics['calibration_gap']):.1f} |\n"

    # Replace table body in README
    # Simple regex replace for the TBD section
    import re
    readme_content = re.sub(
        r"\| Accuracy.*\| Conf Calibration\| TBD         \| TBD          \| TBD   \|",
        table_data.strip(),
        readme_content,
        flags=re.DOTALL
    )
    with open(readme_path, "w") as f:
        f.write(readme_content)

    # 7. Merge and Export
    logger.info("Clearing VRAM before merge...")
    trainer.model.to("cpu")
    del trainer.model
    torch.cuda.empty_cache()

    if args.hf_username:
        logger.info("Merging LoRA adapter and exporting to HF Hub...")
        try:
            # Load base model again, this time to merge
            merged_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                device_map="cpu" # Load to CPU for merging
            )
            # Apply PEFT and merge
            from peft import PeftModel
            merged_model = PeftModel.from_pretrained(merged_model, lora_out)
            merged_model = merged_model.merge_and_unload()
            
            merged_out = os.path.join(args.output_dir, "xyralis-merged")
            merged_model.save_pretrained(merged_out)
            processor.save_pretrained(merged_out)
            logger.info("Merged model saved to %s", merged_out)

            # Push to Hub
            repo_id = f"{args.hf_username}/xyralis-lfm2-vl"
            lora_repo_id = f"{args.hf_username}/xyralis-lfm2-vl-lora"
            
            # Model Card
            from huggingface_hub import ModelCard, ModelCardData
            card_data = ModelCardData(
                language='en',
                license='apache-2.0',
                model_name='Xyralis LFM2-VL',
                eval_results={'accuracy': ft_metrics['accuracy']}
            )
            card = ModelCard.from_template(card_data)
            
            merged_model.push_to_hub(repo_id, commit_message="Initial commit: Merged Xyralis model")
            processor.push_to_hub(repo_id)
            card.push_to_hub(repo_id)
            
            # Reload for lora adapter push
            temp_base = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="cpu")
            final_lora_model = PeftModel.from_pretrained(temp_base, lora_out)
            final_lora_model.push_to_hub(lora_repo_id, commit_message="Initial commit: LoRA adapter")
            processor.push_to_hub(lora_repo_id)
            
            logger.info("Successfully pushed models to Hub!")
        except Exception as e:
             logger.error("Failed to merge and push to hub: %s", e)
    else:
        logger.info("Skipping merge and push to Hub (no hf_username provided).")

    logger.info("=== Pipeline Complete ===")

if __name__ == "__main__":
    main()

# CropAlert Model Fine-tuning Results

This directory contains the output artifacts and evaluation metrics from the fine-tuning of the `LiquidAI/LFM2-VL-7B` model for crop stress classification.

## Artifacts Generated

*   `loss_curve.json`: Training and evaluation loss curves recorded during fine-tuning.
*   `eval_base.json`: Post-training evaluation metrics for the pre-trained base model.
*   `eval_ft.json`: Post-training evaluation metrics for the fine-tuned LoRA model.

## Evaluation Metrics Summary

The table below summarizes the performance gains achieved through fine-tuning on the agricultural dataset.

| Metric          | Base LFM2-VL | CropAlert FT | Delta |
|-----------------|-------------|--------------|-------|
| Accuracy        | TBD         | TBD          | TBD   |
| F1 healthy      | TBD         | TBD          | TBD   |
| F1 mild_stress  | TBD         | TBD          | TBD   |
| F1 severe_stress| TBD         | TBD          | TBD   |
| Weighted F1     | TBD         | TBD          | TBD   |
| Conf Calibration| TBD         | TBD          | TBD   |

*Table will be automatically populated after running `finetune_lora.py`.*

import logging
import time
import re
import torch
from typing import List, Dict, Any, Literal, Tuple
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger("cropalert.inference")

@dataclass
class AnalysisResult:
    classification: Literal["healthy", "mild_stress", "severe_stress", "uncertain"]
    confidence: float
    recommendation: str
    days_to_action: int
    detected_issues: List[str]
    ndvi_mean: float
    stress_score: float
    inference_latency_ms: float
    model_version: str

class CropAlertInference:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_version = Path(model_path).name
        
        logger.info(f"Loading CropAlert model from {model_path} on {self.device}...")
        start_time = time.time()

        # Load Processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "right"

        # Load Model
        if model_path.endswith("-lora"):
            # Load base model from config if possible or assume default
            base_model_name = "LiquidAI/LFM2-VL-7B"
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device
            )

        self.model.eval()
        
        # Warm up
        self._warmup()
        
        load_time = time.time() - start_time
        vram = torch.cuda.memory_allocated() / (1024**2) if self.device == "cuda" else 0
        logger.info(f"Model loaded in {load_time:.2f}s. VRAM used: {vram:.1f} MB")

    def _warmup(self):
        """Run a dummy inference to compile kernels."""
        try:
            dummy_image = Image.new("RGB", (224, 224), color="white")
            dummy_indices = {"ndvi_mean": 0.5, "ndwi_mean": 0.0, "stress_score": 0.0}
            logger.info("Running model warmup...")
            with torch.inference_mode():
                self.analyze_raw(dummy_image, dummy_indices, "Warmup")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def _parse_output(self, text: str) -> Dict[str, Any]:
        """Extract structured fields from LLM text output."""
        # Simple regex-based parsing matching the fine-tuning format
        res = {
            "classification": "uncertain",
            "confidence": 0.0,
            "recommendation": "Manual inspection required due to analysis failure.",
            "days_to_action": -1,
            "detected_issues": []
        }
        
        try:
            cls_match = re.search(r"Classification:\s*([a-zA-Z_]+)", text, re.IGNORECASE)
            conf_match = re.search(r"Confidence:\s*(\d+)%", text)
            action_match = re.search(r"Recommended action:\s*(.*?)(?:\ndays_to_action|$)", text, re.DOTALL | re.IGNORECASE)
            days_match = re.search(r"days_to_action:\s*(\d+)", text, re.IGNORECASE)
            
            if cls_match:
                cls = cls_match.group(1).lower()
                if cls in ["healthy", "mild_stress", "severe_stress"]:
                    res["classification"] = cls
            
            if conf_match:
                res["confidence"] = float(conf_match.group(1)) / 100.0
            
            if action_match:
                res["recommendation"] = action_match.group(1).strip()
            
            if days_match:
                res["days_to_action"] = int(days_match.group(1))
                
            # Heuristic for issues
            issues = []
            if "water" in text.lower(): issues.append("water_stress")
            if "vigor" in text.lower() or "ndvi" in text.lower(): issues.append("vigor_loss")
            if "nitrogen" in text.lower(): issues.append("nitrogen_deficiency")
            res["detected_issues"] = issues

        except Exception as e:
            logger.error(f"Output parsing error: {e}")
            
        return res

    def analyze_raw(self, image: Image.Image, indices: Dict[str, float], context: str = "") -> Tuple[Dict[str, Any], float]:
        """Performs raw inference and returns parsed result and latency."""
        start = time.time()
        
        idx_str = ", ".join([f"{k}={v:.3f}" for k, v in indices.items()])
        prompt = (
            f"Expert agronomic analysis. Context: {context}. "
            f"Spectral data: {idx_str}. Analyze the following <image> and provide status."
        )
        
        # Format for LFM2-VL (Liquid AI style typically uses system/user tokens)
        # Here we follow the format defined in the finetuning script
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT: "

        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.1
            )
            
        generated_text = self.processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        latency = (time.time() - start) * 1000
        
        parsed = self._parse_output(generated_text)
        return parsed, latency

    def analyze(self, image_path: str, indices: Dict[str, float], context: str = "") -> AnalysisResult:
        try:
            image = Image.open(image_path).convert("RGB")
            parsed, latency = self.analyze_raw(image, indices, context)
            
            return AnalysisResult(
                classification=parsed["classification"],
                confidence=parsed["confidence"],
                recommendation=parsed["recommendation"],
                days_to_action=parsed["days_to_action"],
                detected_issues=parsed["detected_issues"],
                ndvi_mean=indices.get("ndvi_mean", 0.0),
                stress_score=indices.get("stress_score", 0.0),
                inference_latency_ms=latency,
                model_version=self.model_version
            )
        except Exception as e:
            logger.error(f"Analysis failed for {image_path}: {e}")
            return AnalysisResult(
                classification="uncertain",
                confidence=0.0,
                recommendation=f"System error: {str(e)}",
                days_to_action=-1,
                detected_issues=[],
                ndvi_mean=0.0,
                stress_score=0.0,
                inference_latency_ms=0.0,
                model_version=self.model_version
            )

    def batch_analyze(self, items: List[Dict]) -> List[AnalysisResult]:
        """Process multiple parcels efficiently."""
        results = []
        # Mixed precision for speed
        with torch.cuda.amp.autocast() if self.device == "cuda" else torch.inference_mode():
            for item in items:
                res = self.analyze(item["image_path"], item["indices"], item.get("context", ""))
                results.append(res)
        return results

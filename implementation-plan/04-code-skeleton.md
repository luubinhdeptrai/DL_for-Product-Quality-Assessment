# Section 4: Code Skeleton

This section provides starter code for the core files requested. The code is intentionally modular and realistic, but still compact enough to adapt quickly.

## 4.1 `dataset.py`

```python
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MULTISPACE_RE = re.compile(r"\s+")
REPEAT_RE = re.compile(r"(.)\1{2,}")

DEFAULT_SLANG_MAP = {
    "sp": "san pham",
    "ko": "khong",
    "k": "khong",
    "okla": "ok",
    "auth": "chinh hang",
}


def clean_review_text(text: str, slang_map: dict[str, str] | None = None) -> str:
    """Light cleaning only. Keep emojis, casing, and Vietnamese diacritics intact."""
    slang_map = slang_map or DEFAULT_SLANG_MAP
    text = unicodedata.normalize("NFC", text or "")
    text = URL_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    text = REPEAT_RE.sub(r"\1\1", text)

    words = []
    for token in text.split(" "):
        words.append(slang_map.get(token, token))
    return " ".join(words)


class MultimodalReviewDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        image_root: str | Path,
        image_transform,
        tokenizer_name: str = "xlm-roberta-base",
        max_length: int = 96,
        factor_names: tuple[str, ...] = ("quality", "price", "appearance"),
    ) -> None:
        self.rows = list(rows)
        self.image_root = Path(image_root)
        self.image_transform = image_transform
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_length = max_length
        self.factor_names = factor_names

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(self.image_root / image_path).convert("RGB")
        return self.image_transform(image)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        cleaned_text = clean_review_text(str(row["review_text"]))
        encoded = self.tokenizer(
            cleaned_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        factor_scores = torch.tensor(
            [float(row[name]) for name in self.factor_names],
            dtype=torch.float32,
        )

        return {
            "pixel_values": self._load_image(str(row["image_path"])),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "overall_score": torch.tensor(float(row["overall_score"]), dtype=torch.float32),
            "factor_scores": factor_scores,
            "meta": {
                "review_id": row.get("review_id"),
                "product_id": row.get("product_id"),
                "raw_text": row.get("review_text", ""),
            },
        }
```

## 4.2 `model_image.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn
import timm


class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "convnext_tiny",
        pretrained: bool = True,
        proj_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.feature_dim = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor | None]:
        spatial_features = self.backbone.forward_features(pixel_values)

        if spatial_features.ndim == 4:
            pooled = spatial_features.mean(dim=(-2, -1))
        else:
            pooled = spatial_features
            spatial_features = None

        embedding = self.proj(pooled)
        return {
            "embedding": embedding,
            "spatial_features": spatial_features,
        }
```

## 4.3 `model_text.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class XLMRTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        proj_dim: int = 256,
        dropout: float = 0.1,
        output_attentions: bool = False,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.output_attentions = output_attentions
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor | tuple | None]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled = masked_mean_pool(outputs.last_hidden_state, attention_mask)
        embedding = self.proj(pooled)
        return {
            "embedding": embedding,
            "token_features": outputs.last_hidden_state,
            "attentions": getattr(outputs, "attentions", None),
        }
```

## 4.4 `model_fusion.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.model_image import ConvNeXtEncoder
from src.models.model_text import XLMRTextEncoder


def masked_mean(tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(tokens)
    return (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class MultimodalQualityModel(nn.Module):
    def __init__(
        self,
        image_model_name: str = "convnext_tiny",
        text_model_name: str = "xlm-roberta-base",
        fusion_dim: int = 256,
        mlp_dim: int = 512,
        use_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.image_encoder = ConvNeXtEncoder(model_name=image_model_name, proj_dim=fusion_dim)
        self.text_encoder = XLMRTextEncoder(model_name=text_model_name, proj_dim=fusion_dim)
        self.use_cross_attention = use_cross_attention

        if use_cross_attention:
            self.image_token_proj = nn.Linear(self.image_encoder.feature_dim, fusion_dim)
            self.text_token_proj = nn.Linear(self.text_encoder.hidden_size, fusion_dim)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True,
            )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_dim, fusion_dim),
            nn.GELU(),
        )
        self.overall_head = nn.Linear(fusion_dim, 1)
        self.factor_head = nn.Linear(fusion_dim, 3)

    @staticmethod
    def _flatten_image_tokens(spatial_features: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, H*W, C]
        return spatial_features.flatten(2).transpose(1, 2)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        image_out = self.image_encoder(pixel_values)
        text_out = self.text_encoder(input_ids, attention_mask)

        if self.use_cross_attention and image_out["spatial_features"] is not None:
            image_tokens = self._flatten_image_tokens(image_out["spatial_features"])
            image_tokens = self.image_token_proj(image_tokens)
            text_tokens = self.text_token_proj(text_out["token_features"])

            cross_text, _ = self.cross_attention(
                query=text_tokens,
                key=image_tokens,
                value=image_tokens,
            )
            text_embedding = masked_mean(cross_text, attention_mask)
        else:
            text_embedding = text_out["embedding"]

        fused_input = torch.cat([image_out["embedding"], text_embedding], dim=-1)
        fused = self.fusion_mlp(fused_input)

        return {
            "overall_score": self.overall_head(fused).squeeze(-1),
            "factor_scores": self.factor_head(fused),
            "image_embedding": image_out["embedding"],
            "text_embedding": text_out["embedding"],
            "spatial_features": image_out["spatial_features"],
            "attentions": text_out["attentions"],
        }
```

## 4.5 `train.py`

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from timm.data import create_transform, resolve_model_data_config
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.data.dataset import MultimodalReviewDataset
from src.models.model_fusion import MultimodalQualityModel


class EarlyStopper:
    def __init__(self, patience: int = 4) -> None:
        self.patience = patience
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if value < self.best:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def move_batch(batch: dict, device: torch.device) -> dict:
    output = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device, non_blocking=True)
        else:
            output[key] = value
    return output


def compute_loss(outputs: dict, batch: dict, factor_weight: float = 0.5) -> torch.Tensor:
    overall_loss = F.mse_loss(outputs["overall_score"], batch["overall_score"])
    factor_loss = F.mse_loss(outputs["factor_scores"], batch["factor_scores"])
    return overall_loss + factor_weight * factor_loss


def run_epoch(model, loader, optimizer, scaler, device, train: bool = True) -> dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_abs_error = 0.0
    total_samples = 0

    for batch in loader:
        batch = move_batch(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = compute_loss(outputs, batch)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * batch["overall_score"].size(0)
        total_abs_error += (outputs["overall_score"] - batch["overall_score"]).abs().sum().item()
        total_samples += batch["overall_score"].size(0)

    return {
        "loss": total_loss / max(total_samples, 1),
        "mae": total_abs_error / max(total_samples, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="artifacts/checkpoints/run_001")
    parser.add_argument("--image_model", type=str, default="convnext_tiny")
    parser.add_argument("--text_model", type=str, default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Replace these placeholders with your real split loading logic.
    train_rows = []
    val_rows = []

    dummy_backbone = MultimodalQualityModel(
        image_model_name=args.image_model,
        text_model_name=args.text_model,
    )
    data_config = resolve_model_data_config(dummy_backbone.image_encoder.backbone)
    train_transform = create_transform(**data_config, is_training=True)
    eval_transform = create_transform(**data_config, is_training=False)
    del dummy_backbone

    train_dataset = MultimodalReviewDataset(train_rows, image_root="data/raw/images", image_transform=train_transform)
    val_dataset = MultimodalReviewDataset(val_rows, image_root="data/raw/images", image_transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    model = MultimodalQualityModel(
        image_model_name=args.image_model,
        text_model_name=args.text_model,
        use_cross_attention=False,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(total_steps // 10, 1),
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=device.type == "cuda")
    early_stopper = EarlyStopper(patience=4)

    history = []
    best_val_mae = float("inf")

    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, scaler, device, train=False)
        scheduler.step()

        summary = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
        }
        history.append(summary)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1}, output_dir / "best.pt")

        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1}, output_dir / "last.pt")

        if early_stopper.step(val_metrics["mae"]):
            break

    with open(output_dir / "history.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)


if __name__ == "__main__":
    main()
```

## 4.6 `evaluate.py`

```python
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support


def bucketize_scores(scores: np.ndarray) -> np.ndarray:
    # Example: low < 4, medium < 7, high >= 7
    return np.digitize(scores, bins=[4.0, 7.0])


@torch.no_grad()
def predict_loop(model, loader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    overall_true, overall_pred = [], []
    factors_true, factors_pred = [], []

    for batch in loader:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        outputs = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        overall_true.append(batch["overall_score"].cpu().numpy())
        overall_pred.append(outputs["overall_score"].cpu().numpy())
        factors_true.append(batch["factor_scores"].cpu().numpy())
        factors_pred.append(outputs["factor_scores"].cpu().numpy())

    return {
        "overall_true": np.concatenate(overall_true),
        "overall_pred": np.concatenate(overall_pred),
        "factors_true": np.concatenate(factors_true),
        "factors_pred": np.concatenate(factors_pred),
    }


def build_metrics(predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    overall_true = predictions["overall_true"]
    overall_pred = predictions["overall_pred"]

    y_true_bucket = bucketize_scores(overall_true)
    y_pred_bucket = bucketize_scores(overall_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_bucket,
        y_pred_bucket,
        average="macro",
        zero_division=0,
    )

    mse = mean_squared_error(overall_true, overall_pred)
    return {
        "mae": float(mean_absolute_error(overall_true, overall_pred)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
```

## 4.7 `api.py`

```python
from __future__ import annotations

import io
from functools import lru_cache

import torch
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from PIL import Image
from timm.data import create_transform, resolve_model_data_config

from src.data.dataset import clean_review_text
from src.models.model_fusion import MultimodalQualityModel
from transformers import AutoTokenizer


class PredictResponse(BaseModel):
    overall_score: float
    factor_scores: dict[str, float]
    explanation: str


app = FastAPI(title="Product Quality Assessment API")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_artifacts():
    model = MultimodalQualityModel(
        image_model_name="convnext_tiny",
        text_model_name="xlm-roberta-base",
        use_cross_attention=False,
    )
    checkpoint = torch.load("artifacts/checkpoints/run_001/best.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
    data_config = resolve_model_data_config(model.image_encoder.backbone)
    transform = create_transform(**data_config, is_training=False)
    return model, tokenizer, transform


def build_explanation(overall: float, factors: dict[str, float]) -> str:
    # Start with a deterministic template before adding an LLM-backed layer.
    strongest = max(factors, key=factors.get)
    weakest = min(factors, key=factors.get)
    return (
        f"Overall score is {overall:.2f}. "
        f"Strongest dimension is {strongest}, while {weakest} is the weakest dimension."
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(review_text: str = Form(...), image: UploadFile = File(...)) -> PredictResponse:
    model, tokenizer, transform = load_artifacts()

    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixel_values = transform(pil_image).unsqueeze(0).to(DEVICE)

    cleaned_text = clean_review_text(review_text)
    encoded = tokenizer(
        cleaned_text,
        padding="max_length",
        truncation=True,
        max_length=96,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    overall_score = float(outputs["overall_score"].cpu().item())
    factor_scores_tensor = outputs["factor_scores"].cpu().squeeze(0)
    factor_scores = {
        "quality": float(factor_scores_tensor[0].item()),
        "price": float(factor_scores_tensor[1].item()),
        "appearance": float(factor_scores_tensor[2].item()),
    }

    return PredictResponse(
        overall_score=overall_score,
        factor_scores=factor_scores,
        explanation=build_explanation(overall_score, factor_scores),
    )
```

## 4.8 Notes On The Skeleton

The starter code intentionally omits configuration loading, logging, Grad-CAM implementation, and experiment tracking to keep the skeleton readable. Those should be added as supporting modules rather than embedded directly into the training or API files.
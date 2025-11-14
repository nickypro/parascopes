# trainer_probe.py

import os
import gc
import pickle
from collections import defaultdict
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
import einops

from normalizers import compute_normalizers, Normalizer, SAMPLES_PER_CHUNK
from utils_load_data import load_residuals, load_embeds


config = {
    # data / paths
    "hf_repo_residuals": "nickypro/fineweb-llama3b-residuals",
    "hf_repo_embeds":    "yulia-volkova/llama-3b-outlines-embeddings_new",
    "local_residuals_dir": None,   # or "/path/to/res_chunks"
    "local_embeds_dir":    None,   # or "/path/to/embed_chunks"

    # chunks
    "start_chunk": 0,
    "end_chunk": 4,       # inclusive
    "norm_chunks": 2,     # how many initial chunks to use for normalization
    "val_last_k": 1,       # how many last chunks to hold out for validation


    "batch_size": 256,
    "num_epochs": 3,
    "d_sonar": 1024,
    "lr": 1e-4,
    "lr_decay": 0.8,
    "weight_decay": 1e-6,
    "limit_layers": None,  # or an int, e.g. 32

    "checkpoint_dir": "./checkpoints/probe_runs",
}


class LinearProbe(nn.Module):
    """
    Simple linear probe: flattens [batch, n_layers, d_model] -> [batch, d_res]
    and maps to d_sonar.
    """
    def __init__(self, n_layers: int, d_model: int, d_sonar: int):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_sonar = d_sonar
        self.linear = nn.Linear(n_layers * d_model, d_sonar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_layers, d_model]
        x = x.reshape(x.shape[0], -1)
        return self.linear(x)


class Trainer:
    """
    Trainer for linear probe on residuals -> SONAR embeddings.

    - Computes normalization on a subset of chunks (cfg.norm_chunks).
    - Uses last cfg.val_last_k chunks as validation, others for training.
    - Loads one chunk at a time (no fancy LRU cache).
    - Logs to wandb (assumes wandb.init(...) already called).
    """

    def __init__(self, config: dict, device: torch.device):
        self._config = dict(config)
        self.device = device

        self._c_ns = None  # cached SimpleNamespace
        # Build chunk lists
        self.all_chunks = list(range(self.c.start_chunk, self.c.end_chunk + 1))
        if not self.all_chunks:
            raise ValueError("No chunks selected: check start_chunk/end_chunk in config.")

        self.val_last_k = getattr(self.c, "val_last_k", 5)
        self.val_chunks = self.all_chunks[-self.val_last_k:] if self.val_last_k > 0 else []
        self.train_chunks = [c for c in self.all_chunks if c not in self.val_chunks]

        if not self.train_chunks:
            raise ValueError("No training chunks left after reserving validation chunks.")

        # 1) Compute normalizers on subset of chunks
        norm_chunk_ids = self.all_chunks[: min(self.c.norm_chunks, len(self.all_chunks))]
        print("Using chunks for normalization:", [f"{c:03d}" for c in norm_chunk_ids])

        res_norm, emb_norm = compute_normalizers(
            norm_chunk_ids=norm_chunk_ids,
            hf_repo_residuals=self.c.hf_repo_residuals,
            hf_repo_embeds=self.c.hf_repo_embeds,
            local_residuals_dir=self.c.local_residuals_dir,
            local_embeds_dir=self.c.local_embeds_dir,
            dtype=torch.float32,
        )

        res_norm.mean = res_norm.mean.to(device)
        res_norm.std = res_norm.std.to(device)
        emb_norm.mean = emb_norm.mean.to(device)
        emb_norm.std = emb_norm.std.to(device)

        self.normalizer_res: Normalizer = res_norm
        self.normalizer_emb: Normalizer = emb_norm

        # 2) Peek at one chunk to infer [n_layers, d_model]
        example_chunk = self.train_chunks[0]
        res_list = load_residuals(
            example_chunk,
            self.c.hf_repo_residuals,
            self.c.local_residuals_dir,
        )
        embeds_example = load_embeds(
            example_chunk,
            self.c.hf_repo_embeds,
            self.c.local_embeds_dir,
        )

        if len(res_list) == 0:
            raise RuntimeError(f"Chunk {example_chunk} has no residuals.")

        res_all = res_list[0]["res"].to(dtype=torch.float32)  # [n_layers, n_para, d_model]
        first_para = res_all[:, :1, :]                        # [n_layers, 1, d_model]
        res_tensor = einops.rearrange(first_para, "layer para dim -> layer dim")  # [n_layers, d_model]

        if self.c.limit_layers is not None:
            res_tensor = res_tensor[: self.c.limit_layers, :]

        n_layers, d_model = res_tensor.shape
        self._config["n_layers"] = int(n_layers)
        self._config["d_model"] = int(d_model)
        self._config["d_res"] = int(n_layers * d_model)

        print(
            f"\nProbe input shape:\n"
            f"  n_layers = {n_layers}, d_model = {d_model}\n"
            f"  flattened d_res = {n_layers * d_model}\n"
            f"  d_sonar = {self.c.d_sonar}"
        )

        # 3) Initialize model, optimizer, scheduler, loss
        self.model = LinearProbe(
            n_layers=n_layers,
            d_model=d_model,
            d_sonar=self.c.d_sonar,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.c.lr,
            weight_decay=self.c.weight_decay,
        )
        self.criterion_mse = nn.MSELoss()

        self.lr_decay = getattr(self.c, "lr_decay", 1.0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=self.lr_decay
        )

        self.global_step = 0

    @property
    def c(self) -> SimpleNamespace:
        # cache the namespace so we don't re-create it on every access
        if self._c_ns is None:
            self._c_ns = SimpleNamespace(**self._config)
        return self._c_ns

    # ---------------------------------------------------------------------
    # Data loading per chunk
    # ---------------------------------------------------------------------
    def _load_chunk_tensors(self, chunk_id: int):
        """
        Load a single chunk and return:
          - res_tensor: [N, n_layers, d_model]
          - embeds:     [N, d_sonar]
        Skips chunk if size mismatch with SAMPLES_PER_CHUNK.
        """
        res_list = load_residuals(
            chunk_id,
            self.c.hf_repo_residuals,
            self.c.local_residuals_dir,
        )
        embeds = load_embeds(
            chunk_id,
            self.c.hf_repo_embeds,
            self.c.local_embeds_dir,
        )

        n_res = len(res_list)
        n_emb = embeds.shape[0] if isinstance(embeds, torch.Tensor) else len(embeds)
        if n_res != SAMPLES_PER_CHUNK or n_emb != SAMPLES_PER_CHUNK:
            print(
                f"[data] skip chunk {chunk_id:03d}: "
                f"res={n_res}, emb={n_emb}, expected={SAMPLES_PER_CHUNK}"
            )
            return None, None

        res_tensors = []
        for res in res_list:
            res_all = res["res"].to(dtype=torch.float32)               # [n_layers, n_para, d_model]
            first_para = res_all[:, :1, :]                             # [n_layers, 1, d_model]
            res_tensor = einops.rearrange(first_para, "layer para dim -> layer dim")  # [n_layers, d_model]
            if self.c.limit_layers is not None:
                res_tensor = res_tensor[: self.c.limit_layers, :]
            res_tensors.append(res_tensor)

        res_tensor = torch.stack(res_tensors, dim=0)  # [N, n_layers, d_model]
        return res_tensor, embeds

    def _make_loader_for_chunk(self, chunk_id: int, shuffle: bool) -> DataLoader | None:
        res_tensor, embeds = self._load_chunk_tensors(chunk_id)
        if res_tensor is None:
            return None

        indices = torch.arange(res_tensor.shape[0])
        dataset = TensorDataset(res_tensor, embeds, indices)
        loader = DataLoader(
            dataset,
            batch_size=self.c.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )
        return loader

    # ---------------------------------------------------------------------
    # Training utilities
    # ---------------------------------------------------------------------
    def _step_batch(self, x: torch.Tensor, y: torch.Tensor):
        """
        One optimization step on a batch.
        Returns (loss_value, mse_value).
        """
        self.optimizer.zero_grad()

        # normalize
        x = self.normalizer_res.normalize(x)
        y = self.normalizer_emb.normalize(y)

        pred = self.model(x)
        loss = self.criterion_mse(pred, y)
        loss.backward()
        self.optimizer.step()

        mse = ((pred - y) ** 2).mean().item()
        return loss.item(), mse

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        n_batches = 0

        pbar_chunks = tqdm(self.train_chunks, desc=f"Epoch {epoch+1} (chunks)")
        for chunk_id in pbar_chunks:
            loader = self._make_loader_for_chunk(chunk_id, shuffle=True)
            if loader is None:
                continue

            pbar = tqdm(loader, leave=False, desc=f"Chunk {chunk_id:03d}")
            for batch_x, batch_y, batch_idx in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                loss_val, mse_val = self._step_batch(batch_x, batch_y)
                self.global_step += 1
                train_loss_sum += loss_val
                train_mse_sum += mse_val
                n_batches += 1

                avg_loss = train_loss_sum / max(n_batches, 1)
                avg_mse = train_mse_sum / max(n_batches, 1)

                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "mse": f"{avg_mse:.4f}"})

                wandb.log(
                    {
                        "global_step": self.global_step,
                        "train/loss": loss_val,
                        "train/mse": mse_val,
                        "train/epoch": epoch + 1,
                        "train/chunk": chunk_id,
                        "train/idx_start": int(batch_idx[0].item()),
                        "train/idx_end": int(batch_idx[-1].item()),
                    },
                    step=self.global_step,
                )

                # clear references ASAP
                del batch_x, batch_y

            del loader
            gc.collect()
            torch.cuda.empty_cache()

        metrics = {
            "train_loss": train_loss_sum / max(n_batches, 1),
            "train_mse": train_mse_sum / max(n_batches, 1),
        }
        return metrics

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        val_loss_sum = 0.0
        val_mse_sum = 0.0
        total_samples = 0
        n_batches = 0

        if not self.val_chunks:
            return {"val_loss": float("nan"), "val_mse": float("nan")}

        pbar_chunks = tqdm(self.val_chunks, desc="Validation (chunks)")
        for chunk_id in pbar_chunks:
            loader = self._make_loader_for_chunk(chunk_id, shuffle=False)
            if loader is None:
                continue

            pbar = tqdm(loader, leave=False, desc=f"Val chunk {chunk_id:03d}")
            for batch_x, batch_y, batch_idx in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # normalize
                x = self.normalizer_res.normalize(batch_x)
                y = self.normalizer_emb.normalize(batch_y)

                pred = self.model(x)
                loss = self.criterion_mse(pred, y)
                mse = ((pred - y) ** 2).mean().item()

                bs = batch_x.shape[0]
                val_loss_sum += loss.item()
                val_mse_sum += mse * bs
                total_samples += bs
                n_batches += 1

                avg_loss = val_loss_sum / max(n_batches, 1)
                avg_mse = val_mse_sum / max(total_samples, 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "mse": f"{avg_mse:.4f}"})

                del batch_x, batch_y, x, y, pred, loss

            del loader
            gc.collect()
            torch.cuda.empty_cache()

        avg_loss = val_loss_sum / max(n_batches, 1)
        avg_mse = val_mse_sum / max(total_samples, 1)

        metrics = {
            "val_loss": avg_loss,
            "val_mse": avg_mse,
        }
        return metrics

    # ---------------------------------------------------------------------
    # Top-level train loop
    # ---------------------------------------------------------------------
    def train(self):
        torch.set_grad_enabled(True)

        for epoch in range(self.c.num_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            wandb.log(
                {
                    "epoch": epoch + 1,
                    **{f"epoch/{k}": v for k, v in train_metrics.items()},
                    **{f"epoch/{k}": v for k, v in val_metrics.items()},
                    "epoch/lr": self.scheduler.get_last_lr()[0],
                },
                step=self.global_step,
            )

            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_metrics['train_loss']:.4f}, "
                f"val_loss={val_metrics['val_loss']:.4f}, "
                f"lr={self.scheduler.get_last_lr()[0]:.2e}"
            )

            self.scheduler.step()

        return self.model

    # ---------------------------------------------------------------------
    # Checkpointing
    # ---------------------------------------------------------------------
    def save_checkpoint(self, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "model": self.model.state_dict(),
                    "config": self._config,
                    "res_norm_mean": self.normalizer_res.mean.cpu(),
                    "res_norm_std": self.normalizer_res.std.cpu(),
                    "emb_norm_mean": self.normalizer_emb.mean.cpu(),
                    "emb_norm_std": self.normalizer_emb.std.cpu(),
                },
                f,
            )

    @classmethod
    def load_checkpoint(cls, filename: str, device: torch.device):
        with open(filename, "rb") as f:
            ckpt = pickle.load(f)

        config = ckpt["config"]
        trainer = cls(config, device)

        trainer.model.load_state_dict(ckpt["model"])
        trainer.normalizer_res = Normalizer(
            mean=ckpt["res_norm_mean"].to(device),
            std=ckpt["res_norm_std"].to(device),
        )
        trainer.normalizer_emb = Normalizer(
            mean=ckpt["emb_norm_mean"].to(device),
            std=ckpt["emb_norm_std"].to(device),
        )

        return trainer

# utils_train.py

import os
import gc
import pickle
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

from utils_normalizers import compute_normalizers, Normalizer, SAMPLES_PER_CHUNK
from utils_load_data import load_residuals, load_embeds


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

    - Uses residual PRE diffs: Δ[l] = resid_pre[l+1] - resid_pre[l]
    - Handles Gemma vs Llama layouts automatically (via model_family).
    - Computes normalization on a subset of chunks (cfg.norm_chunks).
    - Uses last cfg.val_last_k chunks as validation, others for training.
    - Loads one chunk at a time (no fancy LRU cache).
    - Logs to wandb (assumes wandb.init(...) already called).
    """

    def __init__(self, config: dict, device: torch.device):
        self._config = dict(config)
        self.device = device

        # --- infer model_family if not provided explicitly ---
        if "model_family" not in self._config:
            self._config["model_family"] = self._infer_model_family()
            print(f"[Trainer] Inferred model_family = {self._config['model_family']}")

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

        # NOTE: compute_normalizers still uses raw residuals as saved.
        # If you want normalization on PRE DIFFS specifically, you can
        # update compute_normalizers to call the same _residual_pre_diffs
        # logic used below.
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

        # 2) Peek at one chunk to infer [n_layers, d_model] after diffs
        example_chunk = self.train_chunks[0]
        res_list = load_residuals(
            example_chunk,
            self.c.local_residuals_dir,
            self.c.hf_repo_residuals,
        )
        _ = load_embeds(
            example_chunk,
            self.c.local_embeds_dir,
            self.c.hf_repo_embeds,
        )

        if len(res_list) == 0:
            raise RuntimeError(f"Chunk {example_chunk} has no residuals.")

        # res_all: [n_layers_raw, n_para, d_model]
        res_all = res_list[0]["res"]
        res_tensor = self._residual_pre_diffs(res_all)  # [n_layers_eff, d_model]

        n_layers, d_model = res_tensor.shape
        self._config["n_layers"] = int(n_layers)
        self._config["d_model"] = int(d_model)
        self._config["d_res"] = int(n_layers * d_model)

        print(
            f"\nProbe input shape (using residual PRE diffs):\n"
            f"  n_layers = {n_layers}, d_model = {d_model}\n"
            f"  flattened d_res = {n_layers * d_model}\n"
            f"  d_sonar = {self.c.d_sonar}\n"
            f"  model_family = {self.c.model_family}"
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
    # Model-family detection
    # ---------------------------------------------------------------------
    def _infer_model_family(self) -> str:
        """
        Infer model_family ('gemma', 'llama', or 'unknown') from the paths / repo names
        if the user didn't specify it explicitly.
        """
        candidates = []
        for key in [
            "hf_repo_residuals",
            "hf_repo_embeds",
            "local_residuals_dir",
            "local_embeds_dir",
        ]:
            val = self._config.get(key, None)
            if val is not None:
                candidates.append(str(val).lower())

        joined = " ".join(candidates)
        if "llama" in joined:
            return "llama"
        if "gemma" in joined:
            return "gemma"
        return "unknown"

    # ---------------------------------------------------------------------
    # residual PRE-diff transform
    # ---------------------------------------------------------------------
    def _residual_pre_diffs(self, res_all: torch.Tensor) -> torch.Tensor:
        """
        Convert raw residual dump [n_layers_raw, n_para, d_model] into
        layer-wise residual diffs:

            Δ[l] = resid_pre[l+1] - resid_pre[l]

        for all model families.

        Returns: [n_layers_eff, d_model]
        """
        # 1) Take first paragraph / token
        #    res_all: [n_layers_raw, n_para, d_model]
        #    states:  [n_layers_raw, d_model]
        states = res_all[:, 0, :].to(dtype=torch.float32)

        family = getattr(self.c, "model_family", "unknown").lower()

        # 2) Extract only resid_pre states, depending on model family
        if "llama" in family:
            # Pattern is [pre0, mid0, post0, pre1, mid1, post1, ...]
            # Keep only pre's at layer boundaries.
            states = states[::2, :]
        elif "gemma" in family:
            # Gemma resids are pre/post only; assume first is pre.
            # Assume states are [pre0, pre1, ..., preL].
            pass
        else:
            pass

        if states.shape[0] < 2:
            raise ValueError(
                f"Not enough residual states ({states.shape[0]}) to compute diffs."
            )

        # 3) Compute diffs always TRUE:
        #    Δ[l] = pre[l+1] - pre[l]
        diffs = states[1:, :] - states[:-1, :]  # [n_layers_eff, d_model]

        # 4) Optional layer limit
        if self.c.limit_layers is not None:
            diffs = diffs[: self.c.limit_layers, :]

        return diffs

    # ---------------------------------------------------------------------
    # Data loading per chunk
    # ---------------------------------------------------------------------
    def _load_chunk_tensors(self, chunk_id: int):
        """
        Load a single chunk and return:
          - res_tensor: [N, n_layers, d_model]  (here n_layers = #diffs)
          - embeds:     [N, d_sonar]
        Skips chunk if size mismatch with SAMPLES_PER_CHUNK.
        """
        res_list = load_residuals(
            chunk_id,
            self.c.local_residuals_dir,
            self.c.hf_repo_residuals,
        )
        embeds = load_embeds(
            chunk_id,
            self.c.local_embeds_dir,
            self.c.hf_repo_embeds,
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
            # res["res"]: [n_layers_raw, n_para, d_model]
            res_all = res["res"]
            res_tensor = self._residual_pre_diffs(res_all)  # [n_layers_eff, d_model]
            res_tensors.append(res_tensor)

        res_tensor = torch.stack(res_tensors, dim=0)  # [N, n_layers_eff, d_model]
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

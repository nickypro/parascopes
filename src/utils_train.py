# %%
import os
import gc
import pickle
from tqdm import tqdm
import torch
import wandb
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset
from utils_models  import Linear, MLP, ScaledLinear
from utils_welford import load_or_compute_welford_stats, Normalizer
from utils_load_data import load_res_data, load_embeds, load_split_paragraphs
from utils_sonar import SonarDecoderCELoss
from collections import defaultdict

class Trainer:
    def __init__(self, config, device):
        self._config = config
        self._validate_config()
        self.device = device

        self._config["group_operation"] = "cat" if "group_operation" not in self._config else self._config["group_operation"]
        welford_data = load_or_compute_welford_stats(
            groups_to_load=self.c.groups_to_load,
            group_size=self.c.group_size,
            group_operation=self.c.group_operation,
            do_diff_data=self.c.do_diff_data,
            model_path=self.c.model_path,
        )
        self.normalizer_emb: Normalizer = welford_data.norm_emb
        self.normalizer_res: Normalizer = welford_data.norm_res

        # d_res = load_res_data(0, self.c.group_size, self.c.groups_to_load, self.c.group_operation).shape[-1]
        res_data, text_data, shapes = load_res_data(
            0,
            group_size=self.c.group_size,
            groups_to_load=self.c.groups_to_load,
            group_operation=self.c.group_operation,
            do_diff_data=self.c.do_diff_data,
            model_path=self.c.model_path,
        )
        d_res = res_data.shape[-1]
        self._config["d_res"] = d_res

        # Initialize model
        self.model = self._init_model().to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.c.lr,
            weight_decay=self.c.weight_decay
        )
        self.criterion_mse = torch.nn.MSELoss()

        # Initialize learning rate scheduler: reduce LR by a factor of 0.9 every
        # epoch
        self.lr_decay = self.c.lr_decay if "lr_decay" in self._config else 1.0
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.lr_decay)

        self.use_decoder = self.c.use_decoder if "use_decoder" in self._config else False
        self.decoder_max_tokens = self.c.decoder_max_tokens if "decoder_max_tokens" in self._config else None
        self.decoder_coeff = self.c.decoder_coeff if "decoder_coeff" in self._config else 1.0

        if self.use_decoder:
            self.decoder_loss = SonarDecoderCELoss(max_tokens=self.decoder_max_tokens)
            self.decoder_loss.model.eval()

    def _validate_config(self):
        assert "model_path" in self._config, "model_path is required"
        assert self._config["model_path"] is not None, "model_path is required"
        assert "groups_to_load" in self._config, "groups_to_load is required"
        assert self._config["groups_to_load"] is not None, "groups_to_load is required"
        assert "group_size" in self._config, "group_size is required"
        assert self._config["group_size"] is not None, "group_size is required"
        assert "group_operation" in self._config, "group_operation is required"
        assert self._config["group_operation"] is not None, "group_operation is required"
        assert "do_diff_data" in self._config, "do_diff_data is required"
        assert self._config["do_diff_data"] is not None, "do_diff_data is required"
        assert "model_path" in self._config, "model_path is required"
        verbose = False
        if verbose:
            print(f"{self._config=}")

    @property
    def c(self):
        return SimpleNamespace(**self._config)

    def _init_model(self):
        if self.c.model_type == 'scaled_linear':
            return ScaledLinear(self.c)
        elif self.c.model_type == 'linear':
            return Linear(self.c)
        elif self.c.model_type == 'mlp':
            return MLP(self.c)
        else:
            raise ValueError(f"Unknown model type: {self.c.model_type}")

    @torch.no_grad()
    def create_data_loader(self, file_idx, shuffle=False):
        res_data, text_data, shapes = load_res_data(
            file_idx, 
            group_size=self.c.group_size,
            groups_to_load=self.c.groups_to_load, 
            group_operation=self.c.group_operation,
            do_diff_data=self.c.do_diff_data,
            model_path=self.c.model_path,
        )
        embeds = load_embeds(file_idx, shapes, model_path=self.c.model_path)
        paragraphs = []
        for idx, (p, shape) in enumerate(zip(text_data, shapes)):
            curr_paras = p[1:1+shape]
            paragraphs.extend(curr_paras)
            if len(curr_paras) != shape:
                print(f"idx {idx}: Paragraph length {len(curr_paras)} != shape {shape} for index {file_idx}")
            
        indices = torch.arange(len(paragraphs))
        print(f"{res_data.shape=}, {embeds.shape=}, {indices.shape=}")
        dataset = TensorDataset(res_data, embeds, indices)
        __data_loader = DataLoader(dataset, batch_size=self.c.batch_size, shuffle=shuffle)

        def get_batch(__data_loader):
            # we shuffle the [res_data, embeds, paragraphs] together
            # but use tensor dataset so we need to manually get the texts here 
            for x, y, idxs in __data_loader:
                texts = [paragraphs[i] for i in idxs.cpu().numpy()]
                yield x, y, texts

        return get_batch(__data_loader), len(__data_loader)

    def calculate_total_loss(self, input_embeds, target_embeds, target_text=None, use_decoder=None):
        if use_decoder is None:
            use_decoder = self.use_decoder

        # Initialise "loss" variables
        loss_mse = torch.tensor(0.0, device=self.device)
        loss_ce = torch.tensor(0.0, device=self.device)

        # get the (noarmalized) approximate output
        outputs = self.model(input_embeds)

        # calculate MSE loss
        loss_mse = self.criterion_mse(outputs, target_embeds)

        if use_decoder: # get CE loss from the decoder (need to unnormalize y)
            y_unnormed = self.normalizer_emb.restore(outputs)
            loss_ce = self.decoder_coeff * self.decoder_loss(y_unnormed, target_text)

        loss = loss_mse + loss_ce
        loss_data = {"loss": loss.item(), "loss_mse": loss_mse.item(), "loss_ce": loss_ce.item()}
        return loss, loss_data

    def train_epoch(self, epoch):
        self.model.train()
        train_losses = defaultdict(float)
        train_batches = 0

        for file_idx in (pbar := tqdm(range(self.c.num_files), desc=f"Epoch {epoch+1}")):
            train_loader, num_batches = self.create_data_loader(file_idx, shuffle=True)

            for curr_batch, (x, y, texts) in enumerate(train_loader):
                x = self.normalizer_res.normalize(x.to(self.device))
                y = self.normalizer_emb.normalize(y.to(self.device))

                loss, loss_data = self.calculate_total_loss(x, y, texts, use_decoder=self.use_decoder)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                for key, value in loss_data.items():
                    train_losses[key] += value
                train_batches += 1
                pbar.set_postfix({
                    "batch": f"{curr_batch}/{num_batches}",
                    **{k:(v/train_batches) for k,v in train_losses.items()},
                })

                if hasattr(self.model, "normalize"):
                    self.model.normalize()

                del x, y, texts, loss, loss_data

            del train_loader
            gc.collect()
            torch.cuda.empty_cache()

        return {f"train_{key}": (value/train_batches) for key, value in train_losses.items()}

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_losses = defaultdict(float)

        VALIDATION_FILE_INDEX = 99
        val_batches = 0
        with torch.no_grad():
            test_loader, num_batches = self.create_data_loader(VALIDATION_FILE_INDEX, shuffle=False)

            for curr_batch, (x, y, texts) in enumerate(pbar := tqdm(test_loader, desc="Validating")):
                x = self.normalizer_res.normalize(x.to(self.device))
                y = self.normalizer_emb.normalize(y.to(self.device))

                loss, loss_data = self.calculate_total_loss(x, y, texts, use_decoder=self.use_decoder)
                for key, value in loss_data.items():
                    val_losses[key] += value

                # clear cache
                del x, y, texts, loss
                gc.collect()
                torch.cuda.empty_cache()
                val_batches += 1
                pbar.set_postfix({
                    "batch": f"{curr_batch}/{num_batches}",
                    **{k:(v/val_batches) for k,v in val_losses.items()},
                })
            del test_loader
            gc.collect()
            torch.cuda.empty_cache()

        return {f"val_{key}": (value/val_batches) for key, value in val_losses.items()}

    def train(self):
        torch.set_grad_enabled(True)

        for epoch in range(self.c.num_epochs):
            train_losses = self.train_epoch(epoch)
            val_losses = self.validate()  # Validate on next file

            wandb.log({
                "epoch": epoch + 1,
                **train_losses,
                **val_losses,
            })

            print(f"Epoch {epoch+1}: Train Loss: {train_losses['train_loss']:.4f}, Val Loss: {val_losses['val_loss']:.4f}")

            # Step the learning rate scheduler
            self.scheduler.step()

        return self.model

    def save_checkpoint(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model.state_dict(),
                'config': self.c,
                'welford_emb': self.normalizer_emb.welford,
                'welford_res': self.normalizer_res.welford,
            }, f)

    @classmethod
    def load_checkpoint(cls, filename, device):
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)

        config = checkpoint['config'].__dict__
        normalizer_emb = Normalizer(checkpoint['welford_emb'])
        normalizer_res = Normalizer(checkpoint['welford_res'])

        print(config)
        trainer = cls(config, device)
        trainer.model.load_state_dict(checkpoint['model'])
        return trainer

    @classmethod
    def load_from_wandb(cls, run_name, device=None):
        """Load model directly from W&B run name."""
        run = wandb.Api().run(run_name)
        run_id = run.id
        print(run, run.config)
        model_type = run.config['model_type']
        operations = run.config['group_operation']
        filename = f"./checkpoints/sweeps/{run_id}_{model_type}_{operations}.pkl"

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Checkpoint file not found: {filename}")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return cls.load_checkpoint(filename, device=device)

# %%

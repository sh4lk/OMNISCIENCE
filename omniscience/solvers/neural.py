"""Neural Cryptanalysis Module.

Trains a deep neural network (Transformer or MLP) to approximate the
decryption function directly from (plaintext, ciphertext) pairs.

This is the "last resort" solver — when all mathematical approaches fail,
we treat decryption as a sequence-to-sequence learning problem and leverage
GPU acceleration to learn the mapping.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from omniscience.core.config import NeuralConfig
from omniscience.core.types import CryptoInstance, ReconResult, SolverResult, SolverStatus

log = logging.getLogger(__name__)


# ====================================================================== #
#  PyTorch Models                                                         #
# ====================================================================== #

def _build_mlp(input_dim: int, output_dim: int, cfg: NeuralConfig):
    """Build a deep MLP for decryption approximation."""
    import torch
    import torch.nn as nn

    layers: list[nn.Module] = []
    dim = input_dim
    for i in range(cfg.num_layers):
        out = cfg.hidden_dim
        layers.append(nn.Linear(dim, out))
        layers.append(nn.GELU())
        layers.append(nn.LayerNorm(out))
        if i > 0 and i % 2 == 0:
            layers.append(nn.Dropout(0.1))
        dim = out
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class CryptoTransformer:
    """Transformer-based decryption approximator.

    Treats each byte of input as a token, uses positional encoding,
    and predicts output bytes autoregressively or in parallel.
    """

    def __init__(self, vocab_size: int, seq_len: int, cfg: NeuralConfig):
        import torch
        import torch.nn as nn

        self.cfg = cfg
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._build(vocab_size, seq_len, cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def _build(vocab_size: int, seq_len: int, cfg: NeuralConfig):
        import torch
        import torch.nn as nn

        class TransformerDecryptor(nn.Module):
            def __init__(self):
                super().__init__()
                d_model = cfg.hidden_dim
                self.embed = nn.Embedding(vocab_size, d_model)
                self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=cfg.num_heads,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
                self.output_proj = nn.Linear(d_model, vocab_size)
                self.norm = nn.LayerNorm(d_model)

            def forward(self, x):
                # x: (batch, seq_len) of int token IDs
                h = self.embed(x) + self.pos_enc[:, : x.size(1), :]
                h = self.transformer(h)
                h = self.norm(h)
                return self.output_proj(h)  # (batch, seq_len, vocab_size)

        return TransformerDecryptor()

    def train(
        self,
        ct_data: NDArray[np.int64],
        pt_data: NDArray[np.int64],
        max_epochs: int | None = None,
    ) -> dict[str, Any]:
        """Train on (ciphertext → plaintext) pairs.

        Args:
            ct_data: (N, seq_len) array of ciphertext token IDs.
            pt_data: (N, seq_len) array of plaintext token IDs.
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        epochs = max_epochs or self.cfg.max_epochs
        bs = self.cfg.batch_size

        ct_t = torch.tensor(ct_data, dtype=torch.long, device=self.device)
        pt_t = torch.tensor(pt_data, dtype=torch.long, device=self.device)
        dataset = TensorDataset(ct_t, pt_t)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)

        best_loss = float("inf")
        patience_counter = 0
        history: list[float] = []

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            for ct_batch, pt_batch in loader:
                logits = self.model(ct_batch)  # (B, L, V)
                loss = self.criterion(
                    logits.view(-1, self.vocab_size),
                    pt_batch.view(-1),
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            history.append(avg_loss)

            if epoch % 10 == 0:
                log.info("[Neural/Transformer] Epoch %d/%d  loss=%.6f", epoch, epochs, avg_loss)

            # Early stopping
            if avg_loss < best_loss - 1e-5:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stop_patience:
                    log.info("[Neural/Transformer] Early stop at epoch %d", epoch)
                    break

            # Perfect convergence check
            if avg_loss < 1e-6:
                log.info("[Neural/Transformer] Perfect convergence at epoch %d", epoch)
                break

        return {"final_loss": best_loss, "epochs_trained": epoch + 1, "history": history}

    def predict(self, ct_data: NDArray[np.int64]) -> NDArray[np.int64]:
        """Predict plaintext from ciphertext."""
        import torch

        self.model.eval()
        with torch.no_grad():
            ct_t = torch.tensor(ct_data, dtype=torch.long, device=self.device)
            logits = self.model(ct_t)
            preds = logits.argmax(dim=-1)
        return preds.cpu().numpy()


# ====================================================================== #
#  Solver Wrapper                                                         #
# ====================================================================== #

class NeuralCryptanalysisSolver:
    """Neural network-based decryption solver."""

    NAME = "neural"

    def __init__(self, cfg: NeuralConfig | None = None):
        self.cfg = cfg or NeuralConfig()

    def solve(
        self,
        instance: CryptoInstance,
        recon: ReconResult,
        timeout: float = 3600.0,
    ) -> SolverResult:
        t0 = time.perf_counter()
        try:
            import torch
        except ImportError:
            return self._fail("PyTorch not installed", t0)

        try:
            pt = np.array(instance.pt_as_int_list(), dtype=np.int64)
            ct = np.array(instance.ct_known_as_int_list(), dtype=np.int64)
            ct_target = np.array(instance.ct_target_as_int_list(), dtype=np.int64)

            n_samples = min(len(pt), len(ct))
            if n_samples < 10:
                return self._fail("Not enough training pairs for neural solver", t0)

            seq_len = max(len(pt), len(ct))
            # Pad to equal length
            if len(pt) < seq_len:
                pt = np.pad(pt, (0, seq_len - len(pt)))
            if len(ct) < seq_len:
                ct = np.pad(ct, (0, seq_len - len(ct)))
            if len(ct_target) < seq_len:
                ct_target = np.pad(ct_target, (0, seq_len - len(ct_target)))

            # Determine vocabulary size
            vocab = int(max(np.max(pt), np.max(ct), np.max(ct_target))) + 1
            vocab = max(vocab, 256)  # at least byte-level

            # Reshape to (N_samples, seq_len) — split into chunks
            chunk_size = min(seq_len, 64)
            n_chunks = seq_len // chunk_size
            if n_chunks == 0:
                chunk_size = seq_len
                n_chunks = 1

            pt_chunks = pt[:n_chunks * chunk_size].reshape(n_chunks, chunk_size)
            ct_chunks = ct[:n_chunks * chunk_size].reshape(n_chunks, chunk_size)

            # Augment with random shifts if we have too few samples
            if n_chunks < 100:
                aug_ct, aug_pt = self._augment(ct_chunks, pt_chunks, target=500)
            else:
                aug_ct, aug_pt = ct_chunks, pt_chunks

            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(
                "[Neural] Training on %d samples, seq_len=%d, vocab=%d, device=%s",
                len(aug_ct), chunk_size, vocab, device,
            )

            if self.cfg.use_transformer:
                model = CryptoTransformer(vocab, chunk_size, self.cfg)
            else:
                model = self._build_mlp_wrapper(chunk_size, vocab, device)

            stats = model.train(aug_ct, aug_pt)
            log.info("[Neural] Training complete: %s", {k: v for k, v in stats.items() if k != "history"})

            # Predict
            ct_target_chunks = ct_target[:n_chunks * chunk_size].reshape(n_chunks, chunk_size)
            predictions = model.predict(ct_target_chunks).flatten()

            # Validate on known pairs
            known_preds = model.predict(ct_chunks).flatten()
            accuracy = float(np.mean(known_preds[:len(pt_chunks.flatten())] == pt_chunks.flatten()))
            log.info("[Neural] Validation accuracy: %.2f%%", accuracy * 100)

            if accuracy > 0.95:
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    decrypted=predictions.tolist(),
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=accuracy,
                    details={
                        "method": "transformer" if self.cfg.use_transformer else "mlp",
                        "accuracy": accuracy,
                        "training_stats": {k: v for k, v in stats.items() if k != "history"},
                    },
                )
            else:
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.FAILED,
                    decrypted=predictions.tolist(),
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=accuracy,
                    details={"reason": f"Accuracy {accuracy:.2%} below threshold", "method": "neural"},
                )

        except Exception as exc:
            log.exception("Neural solver crashed")
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.FAILED,
                elapsed_seconds=time.perf_counter() - t0,
                details={"error": str(exc)},
            )

    def _build_mlp_wrapper(self, seq_len: int, vocab: int, device: str):
        """Wrapper to use MLP with the same train/predict interface."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        cfg = self.cfg
        _device = torch.device(device)

        class MLPWrapper:
            def __init__(self):
                self.model = _build_mlp(seq_len * vocab, seq_len * vocab, cfg).to(_device)
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate)
                self.criterion = nn.CrossEntropyLoss()
                self.seq_len = seq_len
                self.vocab = vocab

            def _one_hot(self, data):
                t = torch.tensor(data, dtype=torch.long, device=_device)
                return nn.functional.one_hot(t, self.vocab).float().view(t.size(0), -1)

            def train(self, ct_data, pt_data):
                ct_oh = self._one_hot(ct_data)
                pt_t = torch.tensor(pt_data, dtype=torch.long, device=_device)
                dataset = TensorDataset(ct_oh, pt_t)
                loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
                best_loss = float("inf")
                patience = 0
                for epoch in range(cfg.max_epochs):
                    total = 0.0
                    nb = 0
                    self.model.train()
                    for x, y in loader:
                        logits = self.model(x).view(-1, self.seq_len, self.vocab)
                        loss = self.criterion(logits.reshape(-1, self.vocab), y.reshape(-1))
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        total += loss.item()
                        nb += 1
                    avg = total / max(nb, 1)
                    if avg < best_loss - 1e-5:
                        best_loss = avg
                        patience = 0
                    else:
                        patience += 1
                        if patience >= cfg.early_stop_patience:
                            break
                    if avg < 1e-6:
                        break
                return {"final_loss": best_loss, "epochs_trained": epoch + 1}

            def predict(self, ct_data):
                self.model.eval()
                with torch.no_grad():
                    ct_oh = self._one_hot(ct_data)
                    logits = self.model(ct_oh).view(-1, self.seq_len, self.vocab)
                    return logits.argmax(dim=-1).cpu().numpy()

        return MLPWrapper()

    @staticmethod
    def _augment(
        ct: NDArray, pt: NDArray, target: int = 500
    ) -> tuple[NDArray, NDArray]:
        """Augment training data by circular shifts and noise."""
        n = len(ct)
        if n >= target:
            return ct, pt
        reps = (target // n) + 1
        ct_aug = np.tile(ct, (reps, 1))[:target]
        pt_aug = np.tile(pt, (reps, 1))[:target]
        # Add circular shifts
        for i in range(n, len(ct_aug)):
            shift = np.random.randint(1, ct.shape[1])
            ct_aug[i] = np.roll(ct_aug[i], shift)
            pt_aug[i] = np.roll(pt_aug[i], shift)
        return ct_aug, pt_aug

    def _fail(self, reason: str, t0: float) -> SolverResult:
        log.debug("[Neural] %s", reason)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.FAILED,
            elapsed_seconds=time.perf_counter() - t0,
            details={"reason": reason},
        )

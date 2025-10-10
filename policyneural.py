import os
import math
from typing import Dict, Tuple, Optional

import numpy as np

from board import Board

# ----- PyTorch implementation appended below (overrides NumPy version) -----
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicyValueTorch:  # type: ignore[override]
    """
    PyTorch MLP with the same public API as the previous NumPy version:
      - forward(X) -> (policy_logits, value, cache)
      - backward(cache, target_pi, target_v, l2, c_v) -> (loss, grads)
      - step(grads, lr)

    We keep the training workflow unchanged by returning explicit grads and
    applying them in step(), even though PyTorch could update internally.
    We persist weights as .pt files (generic and size-specific) for compatibility.
    """

    def __init__(self, hidden_size: int = 256, seed: int = 42, device: str = "cpu") -> None:
        self.hidden_size = int(hidden_size)
        self.rng = np.random.RandomState(seed)
        self._device = torch.device(device)
        self._use_cuda_graphs: bool = False
        torch.manual_seed(int(seed))
        # Prefer high-throughput matmul on CUDA (Ampere supports TF32)
        try:
            if self._device.type == "cuda":
                torch.set_float32_matmul_precision("high")
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
                torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
        except Exception:
            pass

        self.input_dim: int = 0
        self.policy_dim: int = 0
        self.fc1: nn.Linear = None  # type: ignore
        self.policy_head: nn.Linear = None  # type: ignore
        self.value_head: nn.Linear = None  # type: ignore

        self._loaded_once = False
        # Load if possible (will defer layer creation until dims are known)
        self._maybe_load()

        # Runtime batching helpers
        self._copy_stream: Optional[torch.cuda.Stream] = None  # type: ignore[name-defined]
        self._pinned_batch: Optional[torch.Tensor] = None
        self._gpu_input_batch: Optional[torch.Tensor] = None
        self._batch_feature_dim: int = 0
        self._batch_capacity: int = 0
        self._graph: Optional[torch.cuda.CUDAGraph] = None  # type: ignore[name-defined]
        self._graph_logits: Optional[torch.Tensor] = None
        self._graph_values: Optional[torch.Tensor] = None

    # ----- persistence helpers -----
    def _weights_path_pt(self) -> str:
        # Allow override via env var so worker processes can load from a snapshot
        override = os.getenv("GOMCTS_WEIGHTS_PATH")
        if override and isinstance(override, str) and len(override) > 0:
            return override
        return os.path.join(os.path.dirname(__file__), "weights.pt")

    def _weights_path_pt_for_policy_dim(self, policy_dim: int) -> str:
        # policy_dim = N*N + 1 => estimate N for readability
        n_est = int(round(math.sqrt(max(0, int(policy_dim) - 1))))
        return os.path.join(os.path.dirname(__file__), f"weights_{n_est}.pt")

    def _weights_path_npz(self) -> str:
        return os.path.join(os.path.dirname(__file__), "weights.npz")

    def _weights_path_npz_for_policy_dim(self, policy_dim: int) -> str:
        n_est = int(round(math.sqrt(max(0, int(policy_dim) - 1))))
        return os.path.join(os.path.dirname(__file__), f"weights_{n_est}.npz")

    def _state_to_np(self) -> Dict[str, np.ndarray]:
        if self.fc1 is None:
            return {}
        # Convert torch shapes to match previous NumPy orientation
        return {
            "W1": self.fc1.weight.detach().cpu().numpy().T.astype(np.float32),
            "b1": self.fc1.bias.detach().cpu().numpy().astype(np.float32),
            "W_policy": self.policy_head.weight.detach().cpu().numpy().T.astype(np.float32),
            "b_policy": self.policy_head.bias.detach().cpu().numpy().astype(np.float32),
            "W_value": self.value_head.weight.detach().cpu().numpy().T.astype(np.float32),
            "b_value": self.value_head.bias.detach().cpu().numpy().astype(np.float32),
        }

    def _np_to_state(self, params: Dict[str, np.ndarray]) -> None:
        W1 = torch.tensor(params["W1"].T, dtype=torch.float32)
        b1 = torch.tensor(params["b1"], dtype=torch.float32)
        Wp = torch.tensor(params["W_policy"].T, dtype=torch.float32)
        bp = torch.tensor(params["b_policy"], dtype=torch.float32)
        Wv = torch.tensor(params["W_value"].T, dtype=torch.float32)
        bv = torch.tensor(params["b_value"], dtype=torch.float32)
        in_dim = W1.shape[1]
        hid = W1.shape[0]
        pol_dim = Wp.shape[0]
        self._ensure_initialized(input_dim=int(in_dim), policy_dim=int(pol_dim))
        with torch.no_grad():
            self.fc1.weight.copy_(W1)
            self.fc1.bias.copy_(b1)
            self.policy_head.weight.copy_(Wp)
            self.policy_head.bias.copy_(bp)
            self.value_head.weight.copy_(Wv)
            self.value_head.bias.copy_(bv)

    def _save(self) -> None:
        # Save .pt
        if self.fc1 is not None:
            payload = {
                "input_dim": self.input_dim,
                "policy_dim": self.policy_dim,
                "state_dict": {
                    "fc1.weight": self.fc1.weight.detach().cpu(),
                    "fc1.bias": self.fc1.bias.detach().cpu(),
                    "policy_head.weight": self.policy_head.weight.detach().cpu(),
                    "policy_head.bias": self.policy_head.bias.detach().cpu(),
                    "value_head.weight": self.value_head.weight.detach().cpu(),
                    "value_head.bias": self.value_head.bias.detach().cpu(),
                }
            }
            # Generic path
            torch.save(payload, self._weights_path_pt())
            # Note: disable legacy NPZ saves to reduce I/O

    def _maybe_load(self) -> None:
        # Prefer .pt; fall back to .npz
        pt_path = self._weights_path_pt()
        npz_path = self._weights_path_npz()
        if os.path.exists(pt_path):
            try:
                data = torch.load(pt_path, map_location=self._device)
                in_dim = int(data.get("input_dim", 0))
                pol_dim = int(data.get("policy_dim", 0))
                if in_dim > 0 and pol_dim > 0:
                    self._ensure_initialized(in_dim, pol_dim)
                    sd = data.get("state_dict", {})
                    with torch.no_grad():
                        self.fc1.weight.copy_(sd["fc1.weight"])  # type: ignore[index]
                        self.fc1.bias.copy_(sd["fc1.bias"])    # type: ignore[index]
                        self.policy_head.weight.copy_(sd["policy_head.weight"])  # type: ignore[index]
                        self.policy_head.bias.copy_(sd["policy_head.bias"])      # type: ignore[index]
                        self.value_head.weight.copy_(sd["value_head.weight"])    # type: ignore[index]
                        self.value_head.bias.copy_(sd["value_head.bias"])        # type: ignore[index]
                    self._loaded_once = True
                    return
            except Exception:
                pass
        if os.path.exists(npz_path):
            try:
                arrs = np.load(npz_path, allow_pickle=False)
                params = {k: arrs[k] for k in arrs.files}
                self._np_to_state(params)
                self._loaded_once = True
            except Exception:
                pass

    def _maybe_load_for_policy_dim(self, policy_dim: int) -> None:
        """Load from generic weights.pt when compatible; otherwise skip loading."""
        # 1) Try generic weights.pt if its saved policy_dim matches the requested one
        pt_generic = self._weights_path_pt()
        if os.path.exists(pt_generic):
            try:
                data = torch.load(pt_generic, map_location=self._device)
                in_dim = int(data.get("input_dim", 0))
                pol_dim = int(data.get("policy_dim", 0))
                if pol_dim == int(policy_dim) and pol_dim > 0:
                    self._ensure_initialized(input_dim=max(1, in_dim), policy_dim=pol_dim)
                    sd = data.get("state_dict", {})
                    with torch.no_grad():
                        self.fc1.weight.copy_(sd["fc1.weight"])  # type: ignore[index]
                        self.fc1.bias.copy_(sd["fc1.bias"])    # type: ignore[index]
                        self.policy_head.weight.copy_(sd["policy_head.weight"])  # type: ignore[index]
                        self.policy_head.bias.copy_(sd["policy_head.bias"])      # type: ignore[index]
                        self.value_head.weight.copy_(sd["value_head.weight"])    # type: ignore[index]
                        self.value_head.bias.copy_(sd["value_head.bias"])        # type: ignore[index]
                    self._loaded_once = True
                    return
            except Exception:
                pass
        # If incompatible, do not load anything further

    # ----- network shape mgmt -----
    def _ensure_initialized(self, input_dim: int, policy_dim: int) -> None:
        need_rebuild = (
            self.fc1 is None or
            self.input_dim != int(input_dim) or
            self.policy_dim != int(policy_dim)
        )
        if not need_rebuild:
            return
        self.input_dim = int(input_dim)
        self.policy_dim = int(policy_dim)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_size, bias=True).to(self._device)
        self.policy_head = nn.Linear(self.hidden_size, self.policy_dim, bias=True).to(self._device)
        self.value_head = nn.Linear(self.hidden_size, 1, bias=True).to(self._device)
        # Kaiming initialization is already used by default for Linear; keep defaults

    # ----- batched runtime setup -----
    def ensure_batch_runtime(self, batch_size: int, feature_dim: int, use_cuda_graph: bool = False) -> None:
        if self._device.type != "cuda":
            # CPU path: nothing to set up beyond dims; fall back to normal batched forward
            self._batch_capacity = int(max(self._batch_capacity, batch_size))
            self._batch_feature_dim = int(feature_dim)
            return
        import torch.cuda
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream()
        need_alloc = (
            self._pinned_batch is None
            or self._gpu_input_batch is None
            or int(self._batch_capacity) < int(batch_size)
            or int(self._batch_feature_dim) != int(feature_dim)
        )
        if need_alloc:
            self._batch_capacity = int(batch_size)
            self._batch_feature_dim = int(feature_dim)
            self._pinned_batch = torch.empty((self._batch_capacity, self._batch_feature_dim), dtype=torch.float32).pin_memory()
            self._gpu_input_batch = torch.empty((self._batch_capacity, self._batch_feature_dim), dtype=torch.float32, device=self._device)
            # Rebuild graph on reallocation
            self._graph = None
            self._graph_logits = None
            self._graph_values = None
        if use_cuda_graph and self._graph is None:
            # Warmup and capture static forward graph on fixed shapes
            torch.cuda.synchronize()
            self.fc1.eval(); self.policy_head.eval(); self.value_head.eval()
            # Warmup allocate kernels
            with torch.no_grad():
                h = F.relu(self.fc1(self._gpu_input_batch))  # type: ignore[arg-type]
                logits = self.policy_head(h)
                values_t = torch.tanh(self.value_head(h)).reshape(-1)
            torch.cuda.synchronize()
            # Capture
            g = torch.cuda.CUDAGraph()
            self._graph_logits = torch.empty_like(logits)
            self._graph_values = torch.empty_like(values_t)
            with torch.cuda.graph(g):
                h2 = F.relu(self.fc1(self._gpu_input_batch))  # type: ignore[arg-type]
                l2 = self.policy_head(h2)
                v2 = torch.tanh(self.value_head(h2)).reshape(-1)
                self._graph_logits.copy_(l2)
                self._graph_values.copy_(v2)
            self._graph = g

    def forward_batch_runtime(self, feats_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        B, Fdim = int(feats_batch.shape[0]), int(feats_batch.shape[1])
        self._ensure_initialized(input_dim=Fdim, policy_dim=int(self.policy_dim))
        if self._device.type == "cuda" and self._gpu_input_batch is not None and self._pinned_batch is not None:
            import torch.cuda
            src = torch.from_numpy(feats_batch.astype(np.float32, copy=False))
            # Stage into pinned
            self._pinned_batch[:B].copy_(src, non_blocking=True)
            # Copy to device on copy stream
            assert self._copy_stream is not None
            with torch.cuda.stream(self._copy_stream):
                self._gpu_input_batch[:B].copy_(self._pinned_batch[:B], non_blocking=True)
            torch.cuda.current_stream().wait_stream(self._copy_stream)
            # Forward (graph if available)
            self.fc1.eval(); self.policy_head.eval(); self.value_head.eval()
            with torch.no_grad():
                if self._graph is not None and self._graph_logits is not None and self._graph_values is not None:
                    self._graph.replay()
                    logits_t = self._graph_logits[:B]
                    values_t = self._graph_values[:B]
                else:
                    h = F.relu(self.fc1(self._gpu_input_batch[:B]))
                    logits_t = self.policy_head(h)
                    values_t = torch.tanh(self.value_head(h)).reshape(-1)
                priors_t = torch.softmax(logits_t, dim=1)
            priors = priors_t.detach().cpu().numpy().astype(np.float32)
            values = np.clip(values_t.detach().cpu().numpy().astype(np.float32), -1.0, 1.0)
            return priors, values.reshape(-1)
        # CPU fallback
        # Reuse existing single-call batched helper logic
        return infer_policy_value_from_features_batch_torch(self, feats_batch, int(self.policy_dim))

    # ----- API compatible forward/backward/step -----
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:  # type: ignore[override]
        assert X.ndim == 2
        B, Fdim = X.shape
        # Require policy_dim to be set by the inference wrapper before direct forward
        assert self.policy_dim > 0, "policy_dim is not initialized. Call through the inference wrapper to set it."
        self._ensure_initialized(input_dim=int(Fdim), policy_dim=int(self.policy_dim))
        x = torch.from_numpy(X.astype(np.float32)).to(self._device)
        h = F.relu(self.fc1(x))
        logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h))
        cache = {"X": X}  # kept for API; we recompute in backward
        return logits.detach().cpu().numpy().astype(np.float32), value.detach().cpu().numpy().astype(np.float32), cache

    def backward(self,
                 cache: Dict[str, np.ndarray],
                 target_pi: np.ndarray,
                 target_v: np.ndarray,
                 l2: float = 1e-4,
                 c_v: float = 1.0,
                 max_grad_norm: Optional[float] = None) -> Tuple[float, Dict[str, np.ndarray]]:  # type: ignore[override]
        X = cache.get("X")
        assert X is not None
        # Ensure training mode for modules
        self.fc1.train(); self.policy_head.train(); self.value_head.train()
        x = torch.from_numpy(X.astype(np.float32)).to(self._device)
        t_pi = torch.from_numpy(target_pi.astype(np.float32)).to(self._device)
        t_v = torch.from_numpy(target_v.astype(np.float32)).to(self._device)

        # Zero grads
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

        h = F.relu(self.fc1(x))
        logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h))

        logp = F.log_softmax(logits, dim=1)
        ce = -(t_pi * logp).sum(dim=1).mean()
        mse = F.mse_loss(value, t_v)
        # L2 regularization on weights only (no biases) to match NumPy path
        reg = (
            self.fc1.weight.pow(2).sum()
            + self.policy_head.weight.pow(2).sum()
            + self.value_head.weight.pow(2).sum()
        )
        loss = ce + float(c_v) * mse + float(l2) * reg

        loss.backward()
        # Optional gradient clipping for stability
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), float(max_grad_norm))

        # Collect grads in NumPy with the same orientation as old NumPy version
        grads = {
            "W1": self.fc1.weight.grad.detach().cpu().numpy().T.astype(np.float32),
            "b1": self.fc1.bias.grad.detach().cpu().numpy().astype(np.float32),
            "W_policy": self.policy_head.weight.grad.detach().cpu().numpy().T.astype(np.float32),
            "b_policy": self.policy_head.bias.grad.detach().cpu().numpy().astype(np.float32),
            "W_value": self.value_head.weight.grad.detach().cpu().numpy().T.astype(np.float32),
            "b_value": self.value_head.bias.grad.detach().cpu().numpy().astype(np.float32),
        }
        return float(loss.detach().cpu().item()), grads

    def step(self, grads: Dict[str, np.ndarray], lr: float = 3e-3, save_every: int = 1) -> None:  # type: ignore[override]
        # Manual SGD to keep the same workflow
        with torch.no_grad():
            self.fc1.weight -= float(lr) * torch.from_numpy(grads["W1"].T).to(self.fc1.weight.dtype).to(self._device)
            self.fc1.bias   -= float(lr) * torch.from_numpy(grads["b1"]).to(self.fc1.bias.dtype).to(self._device)
            self.policy_head.weight -= float(lr) * torch.from_numpy(grads["W_policy"].T).to(self.policy_head.weight.dtype).to(self._device)
            self.policy_head.bias   -= float(lr) * torch.from_numpy(grads["b_policy"]).to(self.policy_head.bias.dtype).to(self._device)
            self.value_head.weight  -= float(lr) * torch.from_numpy(grads["W_value"].T).to(self.value_head.weight.dtype).to(self._device)
            self.value_head.bias    -= float(lr) * torch.from_numpy(grads["b_value"]).to(self.value_head.bias.dtype).to(self._device)
        # Save based on frequency (tracked via an internal counter)
        c = getattr(self, "_update_counter", 0) + 1
        self._update_counter = c
        if int(save_every) <= 1 or (c % int(save_every) == 0):
            self._save()

    # Helper to iterate parameters
    def parameters(self):
        return [
            self.fc1.weight, self.fc1.bias,
            self.policy_head.weight, self.policy_head.bias,
            self.value_head.weight, self.value_head.bias,
        ]


def infer_policy_value_torch(net: MLPPolicyValueTorch, board: Board) -> Tuple[np.ndarray, float]:  # type: ignore[override]
    size = int(getattr(board, "size", 9))
    num_actions = size * size + 1
    feats = board.to_features().astype(np.float32).reshape(1, -1)
    net._ensure_initialized(input_dim=feats.shape[1], policy_dim=num_actions)
    # Try to load generic weights.pt if compatible with this policy dimension
    net._maybe_load_for_policy_dim(num_actions)

    # Forward (no grad, proper device, eval mode)
    net.fc1.eval(); net.policy_head.eval(); net.value_head.eval()
    with torch.no_grad():
        # Use autocast on CUDA to leverage Tensor Cores / FP16 on Jetson
        use_amp = (net._device.type == "cuda")
        amp_ctx = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast
        with amp_ctx(enabled=use_amp):
            x = torch.from_numpy(feats).to(torch.float32).to(net._device)
            h = F.relu(net.fc1(x))
            logits = net.policy_head(h)
            value_t = torch.tanh(net.value_head(h))
        value = value_t.detach().cpu().numpy().reshape(-1)[0]

    logits_np = logits.detach().cpu().numpy().reshape(-1).astype(np.float32)
    # Softmax over LEGAL actions only for less compute
    legal = np.asarray(board.legal_moves(), dtype=np.int64)
    lgl = logits_np[legal]
    lgl = lgl - float(lgl.max())
    e = np.exp(lgl)
    s = float(e.sum())
    priors = np.zeros((num_actions,), dtype=np.float32)
    if s > 0.0 and np.isfinite(s):
        priors[legal] = (e / s).astype(np.float32)
    else:
        priors[legal] = 1.0 / max(1, len(legal))

    # Komi-aware heuristic blend for early-game guidance
    v_net = float(np.clip(value, -1.0, 1.0))
    try:
        black = int((board.grid == 1).sum())
        white = int((board.grid == -1).sum())
        score = float(black - white)
        # Mild shaping; scale keeps heuristic in a small range
        v_heur = float(np.tanh(score / 10.0))
        stones = float(black + white)
        N = int(getattr(board, "size", 9))
        # Heavier weighting when few stones are on board; cap at 0.5
        alpha = float(min(0.5, max(0.0, 1.0 - stones / float(max(1, N * N)))))
        v_blend = (1.0 - alpha) * v_net + alpha * v_heur
    except Exception:
        v_blend = v_net

    return priors, float(np.clip(v_blend, -1.0, 1.0))
def infer_policy_value_from_features_batch_torch(net: MLPPolicyValueTorch,
                                                feats_batch: np.ndarray,
                                                num_actions: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batched policy/value inference from precomputed features.
    feats_batch: (B, F)
    returns: (priors: (B, A), values: (B,))
    """
    assert feats_batch.ndim == 2
    B, Fdim = feats_batch.shape
    net._ensure_initialized(input_dim=int(Fdim), policy_dim=int(num_actions))
    # Try persistent runtime path
    try:
        net.ensure_batch_runtime(
            batch_size=int(B),
            feature_dim=int(Fdim),
            use_cuda_graph=bool(getattr(net, "_use_cuda_graphs", False)),
        )
        return net.forward_batch_runtime(feats_batch)
    except Exception:
        pass

    net.fc1.eval(); net.policy_head.eval(); net.value_head.eval()
    with torch.no_grad():
        use_amp = (net._device.type == "cuda")
        amp_ctx = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast
        with amp_ctx(enabled=use_amp):
            x_cpu = torch.from_numpy(feats_batch.astype(np.float32, copy=False))
            if net._device.type == "cuda":
                x_cpu = x_cpu.pin_memory()
                x = x_cpu.to(net._device, non_blocking=True)
            else:
                x = x_cpu
            h = F.relu(net.fc1(x))
            logits = net.policy_head(h)
            values_t = torch.tanh(net.value_head(h)).reshape(-1)
            priors_t = torch.softmax(logits, dim=1)
    priors = priors_t.detach().cpu().numpy().astype(np.float32)
    values = np.clip(values_t.detach().cpu().numpy().astype(np.float32), -1.0, 1.0)
    return priors, values.reshape(-1)




def _he_uniform(fan_in: int, fan_out: int, rng: np.random.RandomState) -> np.ndarray:
    limit = np.sqrt(6.0 / float(max(1, fan_in)))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)


class MLPPolicyValueNumpy:
    """
    Lightweight NumPy MLP that maps board feature vectors to
    (policy logits over actions, scalar value in [-1,1]).

    - Input: flatten features from Board.to_features()  =>  4 * N * N
    - Output policy head: N*N + 1 (pass)
    - Output value head: scalar tanh

    The network auto-adapts its first layer on first forward() call for a new
    board size by reinitializing W1/b1 with appropriate input dimension.
    Weights are saved/loaded from weights.npz automatically on step().
    """

    def __init__(self, hidden_size: int = 256, seed: int = 42) -> None:
        self.hidden_size = int(hidden_size)
        self.rng = np.random.RandomState(seed)
        # Parameters are initialized lazily based on first input shape
        self.params: Dict[str, np.ndarray] = {}
        self._loaded_once = False
        self._maybe_load()

    # ----- persistence -----
    def _weights_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "weights.npz")

    def _maybe_load(self) -> None:
        path = self._weights_path()
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=False)
                self.params = {k: data[k] for k in data.files}
                self._loaded_once = True
            except Exception:
                # Ignore corrupt file; reinitialize on demand
                self.params = {}

    def _save(self) -> None:
        path = self._weights_path()
        # Write to a temp file then atomically replace if possible
        tmp_path = path + ".tmp.npz"
        try:
            np.savez(tmp_path, **self.params)
            try:
                os.replace(tmp_path, path)
            except Exception:
                # Fallback for environments without atomic replace
                os.remove(path) if os.path.exists(path) else None
                os.rename(tmp_path, path)
        except Exception:
            # Best-effort; leave weights unsaved on failure
            pass

    # ----- model definition -----
    def _ensure_initialized(self, input_dim: int, policy_dim: int) -> None:
        reinit = False
        if not self.params:
            reinit = True
        else:
            w1 = self.params.get("W1")
            w_p = self.params.get("W_policy")
            if w1 is None or w1.shape[0] != input_dim or w_p is None or w_p.shape[1] != policy_dim:
                reinit = True

        if reinit:
            self.params = {}
            # Input -> hidden
            self.params["W1"] = _he_uniform(input_dim, self.hidden_size, self.rng)
            self.params["b1"] = np.zeros((self.hidden_size,), dtype=np.float32)
            # Hidden -> policy
            self.params["W_policy"] = _he_uniform(self.hidden_size, policy_dim, self.rng)
            self.params["b_policy"] = np.zeros((policy_dim,), dtype=np.float32)
            # Hidden -> value
            self.params["W_value"] = _he_uniform(self.hidden_size, 1, self.rng)
            self.params["b_value"] = np.zeros((1,), dtype=np.float32)

    # ----- forward / backward -----
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        X: (B, F)
        returns (policy_logits: (B, A), value: (B, 1), cache)
        """
        assert X.ndim == 2
        B, F = X.shape
        # Require policy head to be initialized by inference wrapper with correct dim
        if "W_policy" not in self.params:
            raise AssertionError("policy_dim is uninitialized. Call through infer_policy_value_numpy() first.")
        policy_dim = self.params.get("W_policy").shape[1]
        self._ensure_initialized(input_dim=F, policy_dim=int(policy_dim))

        W1 = self.params["W1"]; b1 = self.params["b1"]
        Wp = self.params["W_policy"]; bp = self.params["b_policy"]
        Wv = self.params["W_value"]; bv = self.params["b_value"]

        H_pre = X @ W1 + b1
        H = np.maximum(0.0, H_pre)
        P_logits = H @ Wp + bp
        V = np.tanh(H @ Wv + bv)

        cache = {
            "X": X, "H": H, "H_pre": H_pre,
        }
        return P_logits.astype(np.float32), V.astype(np.float32), cache

    def backward(self,
                 cache: Dict[str, np.ndarray],
                 target_pi: np.ndarray,
                 target_v: np.ndarray,
                 l2: float = 1e-4,
                 c_v: float = 1.0) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Compute combined loss and gradients.
        Loss = cross_entropy(softmax(P_logits), target_pi) + c_v * mse(V, target_v) + l2 * ||W||^2
        """
        X = cache["X"]
        H = cache["H"]
        B = X.shape[0]

        W1 = self.params["W1"]; Wp = self.params["W_policy"]; Wv = self.params["W_value"]

        # Forward again to get current outputs (lightweight)
        P_logits, V, _ = self.forward(X)

        # Softmax and losses
        max_logit = np.max(P_logits, axis=1, keepdims=True)
        exp = np.exp(P_logits - max_logit)
        sum_exp = np.sum(exp, axis=1, keepdims=True)
        P = exp / np.clip(sum_exp, 1e-8, None)

        # Cross-entropy with provided target distribution target_pi
        ce = -np.sum(target_pi * np.log(np.clip(P, 1e-8, 1.0)), axis=1)
        # Value MSE
        mse = np.mean((V - target_v) ** 2, axis=1)
        loss = np.mean(ce + c_v * mse) + l2 * (
            np.sum(W1 * W1) + np.sum(Wp * Wp) + np.sum(Wv * Wv)
        )

        # Gradients
        dP_logits = (P - target_pi) / float(B)
        dV = (2.0 * (V - target_v) / float(B)) * (1.0 - V ** 2) * c_v

        dWp = H.T @ dP_logits + 2.0 * l2 * Wp
        dbp = np.sum(dP_logits, axis=0)

        dHv = dV @ self.params["W_value"].T
        dWv = H.T @ dV + 2.0 * l2 * self.params["W_value"]
        dbv = np.sum(dV, axis=0)

        dH = dP_logits @ self.params["W_policy"].T + dHv
        dH_pre = dH * (cache["H_pre"] > 0)

        dW1 = X.T @ dH_pre + 2.0 * l2 * W1
        db1 = np.sum(dH_pre, axis=0)

        grads = {
            "W1": dW1.astype(np.float32),
            "b1": db1.astype(np.float32),
            "W_policy": dWp.astype(np.float32),
            "b_policy": dbp.astype(np.float32),
            "W_value": dWv.astype(np.float32),
            "b_value": dbv.astype(np.float32),
        }

        return float(loss), grads

    def step(self, grads: Dict[str, np.ndarray], lr: float = 3e-3) -> None:
        for k, g in grads.items():
            if k not in self.params:
                continue
            self.params[k] = (self.params[k] - float(lr) * g).astype(np.float32)
        # Save after every update for simplicity
        self._save()


def infer_policy_value_numpy(net: MLPPolicyValueNumpy, board: Board) -> Tuple[np.ndarray, float]:
    """
    Convenience wrapper to produce a legal policy distribution and value for MCTS.
    Returns (priors: (A,), value: float), where A = N*N + 1.
    """
    size = int(getattr(board, "size", 9))
    num_actions = size * size + 1
    feats = board.to_features().astype(np.float32).reshape(1, -1)

    # Ensure network is initialized with correct policy dimension
    net._ensure_initialized(input_dim=feats.shape[1], policy_dim=num_actions)
    logits, v, _ = net.forward(feats)
    logits = logits.reshape(-1)
    # Softmax over ALL actions; MCTS expansion will filter to legal actions.
    x = logits - np.max(logits)
    exp = np.exp(x)
    s = float(np.sum(exp))
    if np.isfinite(s) and s > 0.0:
        priors = (exp / s).astype(np.float32)
    else:
        priors = np.ones((num_actions,), dtype=np.float32) / float(max(1, num_actions))
    value = float(np.clip(v.reshape(-1)[0], -1.0, 1.0))
    return priors, value
# ----- Backward-compatible aliases (default to Torch implementation) -----
MLPPolicyValue = MLPPolicyValueTorch
infer_policy_value = infer_policy_value_torch



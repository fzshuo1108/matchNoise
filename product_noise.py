#
#
# import numpy as np
# import math
# from typing import List, Optional, Sequence, Tuple
#
# import torch
#
# from opacus.accountants.accountant import IAccountant
# from opacus.optimizers import DPOptimizer
# from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed
# from opacus.privacy_engine import PrivacyEngine
#
#
# # ============================================================
# # Fixed constants
# # ============================================================
#
# FIXED_K = 300000
# FIXED_DELTA0 = 1e-12
#
#
# # ============================================================
# # Basic math utilities
# # ============================================================
#
# def _phi(x: float) -> float:
#     return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
#
#
# def _log_2phi(x: float) -> float:
#     return math.log1p(math.erf(x / math.sqrt(2.0)))
#
#
# def _log_add(logx: float, logy: float) -> float:
#     if logx == -math.inf:
#         return logy
#     if logy == -math.inf:
#         return logx
#     if logx > logy:
#         return logx + math.log1p(math.exp(logy - logx))
#     return logy + math.log1p(math.exp(logx - logy))
#
#
# def _log_comb(n: int, k: int) -> float:
#     if k < 0 or k > n:
#         return -math.inf
#     return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)
#
#
# # ============================================================
# # Product-noise mechanism parameters
# # ============================================================
#
# def product_t_squared(
#     *,
#     M: int,
#     k: int = FIXED_K,
# ) -> float:
#     """
#     Original theorem-based t^2(M, k):
#
#         t^2 = 2 * k^(4/M) * ((M/4 + 3/2)^(1 + 4/M)) / exp(1 + 2/M)
#     """
#     if M <= 0:
#         raise ValueError("M must be positive.")
#     if k <= 0:
#         raise ValueError("k must be positive.")
#
#     return (
#         2.0
#         * (k ** (4.0 / M))
#         * (((M / 4.0) + 1.5) ** (1.0 + 4.0 / M))
#         / (math.e ** (1.0 + 2.0 / M))
#     )
#
#
# def sigma_M_from_epsilon_step(
#     *,
#     epsilon_step: float,
#     clipping_norm: float,
#     M: int,
#     k: int = FIXED_K,
# ) -> float:
#     """
#     Original theorem-based calibration:
#
#         sigma_M = (Delta_2 f / epsilon_step) * sqrt(t^2(M, k))
#
#     with Delta_2 f = 2C.
#     """
#     if epsilon_step <= 0:
#         raise ValueError("epsilon_step must be positive.")
#
#     delta2f = 2.0 * clipping_norm
#     return (delta2f / epsilon_step) * math.sqrt(product_t_squared(M=M, k=k))
#
#
# def product_lambda_from_sigma_M(
#     *,
#     clipping_norm: float,
#     sigma_M: float,
# ) -> float:
#     """
#     For gradient perturbation with clipped gradient sum:
#         Delta_2 f = 2C
#     so
#         lambda = Delta_2 f / sigma_M = (2C) / sigma_M
#     """
#     if sigma_M <= 0:
#         raise ValueError("sigma_M must be positive.")
#     return (2.0 * clipping_norm) / sigma_M
#
#
# def product_C_M(
#     *,
#     M: int,
#     delta0: float,
# ) -> float:
#     """
#     C_M = 1 / cos(theta_0)
#         = ( sqrt(pi) * Gamma(M/2) / (delta0 * Gamma((M-1)/2)) )^(1/(M-2))
#     """
#     if M <= 2:
#         raise ValueError("This formula requires M > 2.")
#     if not (0.0 < delta0 < 1.0):
#         raise ValueError("delta0 must be in (0, 1).")
#
#     log_num = 0.5 * math.log(math.pi) + math.lgamma(M / 2.0)
#     log_den = math.log(delta0) + math.lgamma((M - 1.0) / 2.0)
#     log_ratio = log_num - log_den
#     return math.exp(log_ratio / (M - 2.0))
#
#
# def product_conditional_rdp_single_alpha(
#     *,
#     alpha: float,
#     clipping_norm: float,
#     sigma_M: float,
#     M: int,
#     delta0: float,
# ) -> float:
#     """
#     Paper's conditional single-step RDP formula:
#
#     eps_alpha =
#       1/(alpha-1) * [ alpha * lambda^2 * C_M
#                       + (alpha * lambda * C_M)^2 / 2
#                       + log(2 Phi(alpha * lambda * C_M)) ]
#     """
#     if alpha <= 1.0:
#         raise ValueError("alpha must be > 1.")
#
#     lam = product_lambda_from_sigma_M(
#         clipping_norm=clipping_norm,
#         sigma_M=sigma_M,
#     )
#     c_m = product_C_M(M=M, delta0=delta0)
#
#     a = alpha * lam * c_m
#     return (
#         alpha * (lam ** 2) * c_m
#         + 0.5 * (a ** 2)
#         + _log_2phi(a)
#     ) / (alpha - 1.0)
#
#
# # ============================================================
# # Standard-RDP-style Poisson subsampling workflow
# # ============================================================
#
# def subsampled_rdp_like_standard_workflow(
#     *,
#     alpha: float,
#     sample_rate: float,
#     base_rdp_fn,
# ) -> float:
#     """
#     Keep the same engineering workflow as standard RDP-based accounting.
#
#     For integer alpha >= 2, use a generic log-sum style upper bound:
#         eps_sub(alpha) <= 1/(alpha-1) * log(
#             1 + sum_{j=2}^{alpha} C(alpha, j) q^j (1-q)^(alpha-j) exp((j-1) eps_base(j))
#         )
#
#     For non-integer alpha, linearly interpolate between neighboring integers.
#     """
#     q = sample_rate
#     if not (0.0 < q <= 1.0):
#         raise ValueError("sample_rate must be in (0, 1].")
#
#     if q == 1.0:
#         return base_rdp_fn(alpha)
#
#     if alpha <= 1.0:
#         raise ValueError("alpha must be > 1.")
#
#     def _integer_alpha_bound(a_int: int) -> float:
#         if a_int <= 1:
#             raise ValueError("Integer alpha must be >= 2")
#
#         # j = 0 and j = 1 terms, without privacy penalty
#         # C(a,0)(1-q)^a + C(a,1)q(1-q)^(a-1)
#         logA = _log_add(
#             a_int * math.log(1.0 - q),
#             math.log(a_int) + math.log(q) + (a_int - 1) * math.log(1.0 - q),
#         )
#
#         # j >= 2 terms with privacy penalty
#         for j in range(2, a_int + 1):
#             eps_j = base_rdp_fn(float(j))
#             term = (
#                     _log_comb(a_int, j)
#                     + j * math.log(q)
#                     + (a_int - j) * math.log(1.0 - q)
#                     + (j - 1.0) * eps_j
#             )
#             logA = _log_add(logA, term)
#
#         return logA / (a_int - 1.0)
#
#     if float(alpha).is_integer():
#         return _integer_alpha_bound(int(alpha))
#
#     lo = max(2, int(math.floor(alpha)))
#     hi = int(math.ceil(alpha))
#     eps_lo = _integer_alpha_bound(lo)
#     eps_hi = _integer_alpha_bound(hi)
#
#     weight = alpha - lo
#     return (1.0 - weight) * eps_lo + weight * eps_hi
#
#
# # ============================================================
# # Accountant
# # ============================================================
#
# class ProductConditionalRDPAccountant(IAccountant):
#     """
#     Product-noise conditional RDP accountant.
#
#     Workflow:
#       1) base single-step RDP: paper's conditional RDP formula
#       2) subsampling: RDP-style Poisson-subsampling workflow
#       3) composition: additive in RDP
#       4) final DP conversion:
#              delta_rdp = target_delta - T * delta0
#          epsilon(delta) = min_alpha [ total_rdp(alpha) + log(1/delta_rdp)/(alpha-1) ]
#
#     Interpretation:
#       noise_multiplier is reinterpreted as sigma_M.
#     """
#
#     DEFAULT_ALPHAS = (
#         [1.25, 1.5, 1.75]
#         + [2 + i for i in range(0, 64)]
#         + [128, 256]
#     )
#
#     def __init__(
#         self,
#         *,
#         clipping_norm: float,
#         M: int,
#         delta0: float,
#         alphas: Optional[Sequence[float]] = None,
#     ):
#         super().__init__()
#         self.clipping_norm = float(clipping_norm)
#         self.M = int(M)
#         self.delta0 = float(delta0)
#         self.alphas = list(alphas) if alphas is not None else list(self.DEFAULT_ALPHAS)
#
#     @classmethod
#     def mechanism(cls) -> str:
#         return "product_conditional_rdp"
#
#     def __len__(self) -> int:
#         return len(self.history)
#
#     def step(self, *, noise_multiplier: float, sample_rate: float):
#         sigma_M = float(noise_multiplier)
#         if sigma_M <= 0:
#             raise ValueError("sigma_M must be positive.")
#
#         if len(self.history) > 0:
#             prev_sigma, prev_q, prev_steps = self.history[-1]
#             if prev_sigma == sigma_M and prev_q == sample_rate:
#                 self.history[-1] = (prev_sigma, prev_q, prev_steps + 1)
#                 return
#
#         self.history.append((sigma_M, sample_rate, 1))
#
#     def _step_rdp(self, alpha: float, sigma_M: float, sample_rate: float) -> float:
#         def base_rdp_fn(a: float) -> float:
#             return product_conditional_rdp_single_alpha(
#                 alpha=a,
#                 clipping_norm=self.clipping_norm,
#                 sigma_M=sigma_M,
#                 M=self.M,
#                 delta0=self.delta0,
#             )
#
#         return subsampled_rdp_like_standard_workflow(
#             alpha=alpha,
#             sample_rate=sample_rate,
#             base_rdp_fn=base_rdp_fn,
#         )
#
#     def get_privacy_spent(
#         self,
#         *,
#         delta: float,
#         alphas: Optional[Sequence[float]] = None,
#     ) -> Tuple[float, float]:
#         if not self.history:
#             return 0.0, 0.0
#
#         alphas = list(alphas) if alphas is not None else self.alphas
#         total_steps = sum(steps for _, _, steps in self.history)
#
#         delta_rdp = delta - total_steps * self.delta0
#         if delta_rdp <= 0.0:
#             return float("inf"), float("nan")
#
#         best_eps = float("inf")
#         best_alpha = float("nan")
#
#         for alpha in alphas:
#             if alpha <= 1.0:
#                 continue
#
#             total_rdp = 0.0
#             for sigma_M, q, steps in self.history:
#                 total_rdp += steps * self._step_rdp(alpha, sigma_M, q)
#
#             eps = total_rdp + math.log(1.0 / delta_rdp) / (alpha - 1.0)
#             if eps < best_eps:
#                 best_eps = eps
#                 best_alpha = float(alpha)
#
#         return float(best_eps), float(best_alpha)
#
#     def get_epsilon(self, delta: float, **kwargs) -> float:
#         eps, _ = self.get_privacy_spent(delta=delta, **kwargs)
#         return float(eps)
#
#
# # ============================================================
# # Optimizer
# # ============================================================
#
# class ProductDPOptimizer(DPOptimizer):
#     """
#     Product-noise version of Opacus DPOptimizer.
#
#     In this implementation, `noise_multiplier` is interpreted as sigma_M.
#     Product noise:
#         n = sigma_M * R * h
#     where
#         R ~ chi_1 = |N(0,1)|
#         h ~ Uniform(S^{M-1})
#     """
#
#     def __init__(
#         self,
#         optimizer,
#         *,
#         noise_multiplier: float,
#         max_grad_norm: float,
#         expected_batch_size: Optional[int],
#         loss_reduction: str = "mean",
#         generator=None,
#         secure_mode: bool = False,
#         M: Optional[int] = None,
#         **kwargs,
#     ):
#         super().__init__(
#             optimizer=optimizer,
#             noise_multiplier=noise_multiplier,
#             max_grad_norm=max_grad_norm,
#             expected_batch_size=expected_batch_size,
#             loss_reduction=loss_reduction,
#             generator=generator,
#             secure_mode=secure_mode,
#             **kwargs,
#         )
#         self.M = M
#
#     @property
#     def sigma_M(self) -> float:
#         return float(self.noise_multiplier)
#
#     def _infer_M(self) -> int:
#         if self.M is not None:
#             return int(self.M)
#         return sum(p.numel() for p in self.params if p.requires_grad)
#
#     def _generate_product_noise(self, *, active_params: List[torch.nn.Parameter]):
#         raw_noises = []
#         for p in active_params:
#             ref = p.summed_grad
#             z = torch.normal(
#                 mean=0.0,
#                 std=1.0,
#                 size=ref.shape,
#                 device=ref.device,
#                 dtype=ref.dtype,
#                 generator=self.generator,
#             )
#             raw_noises.append(z)
#
#         eps = torch.finfo(raw_noises[0].dtype).eps
#         magnitude = torch.sqrt(sum(torch.sum(z ** 2) for z in raw_noises)).clamp_min(eps)
#         h_list = [z / magnitude for z in raw_noises]
#
#         ref0 = active_params[0].summed_grad
#         R = torch.abs(
#             torch.normal(
#                 mean=0.0,
#                 std=1.0,
#                 size=(1,),
#                 device=ref0.device,
#                 dtype=ref0.dtype,
#                 generator=self.generator,
#             )
#         )
#
#         sigma_tensor = torch.tensor(self.sigma_M, device=ref0.device, dtype=ref0.dtype)
#         return [sigma_tensor * R * h for h in h_list]
#
#     def add_noise(self):
#         if not self.params:
#             return
#
#         active_params = [p for p in self.params if p.summed_grad is not None]
#         if not active_params:
#             return
#
#         for p in active_params:
#             _check_processed_flag(p.summed_grad)
#
#         _ = self._infer_M()
#
#         noises = self._generate_product_noise(active_params=active_params)
#
#         for p, noise in zip(active_params, noises):
#             p.grad = (p.summed_grad + noise).view_as(p)
#             _mark_as_processed(p.summed_grad)
#
#
# # ============================================================
# # Debug helpers
# # ============================================================
#
# def debug_sigma_curve(
#     *,
#     accountant: ProductConditionalRDPAccountant,
#     target_delta: float,
#     sample_rate: float,
#     steps: int,
#     sigma_list: Optional[Sequence[float]] = None,
# ):
#     if sigma_list is None:
#         sigma_list = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12]
#
#     saved_history = list(accountant.history)
#     try:
#         print("\n[DEBUG] sigma_M -> total epsilon")
#         print(
#             f"[DEBUG] delta0={accountant.delta0:.3e}, "
#             f"M={accountant.M}, C={accountant.clipping_norm}, "
#             f"sample_rate={sample_rate:.6g}, steps={steps}, target_delta={target_delta:.3e}"
#         )
#         for s in sigma_list:
#             try:
#                 accountant.history = [(float(s), sample_rate, steps)]
#                 eps = accountant.get_epsilon(delta=target_delta)
#                 print(f"[DEBUG] sigma_M={s:.3e}, total_epsilon={eps}")
#             except Exception as e:
#                 print(f"[DEBUG] sigma_M={s:.3e}, total_epsilon=ERROR: {e}")
#     finally:
#         accountant.history = saved_history
#
#
# def debug_step_rdp_curve(
#     *,
#     accountant: ProductConditionalRDPAccountant,
#     sample_rate: float,
#     sigma_list: Optional[Sequence[float]] = None,
#     alphas_to_check: Optional[Sequence[float]] = None,
# ):
#     if sigma_list is None:
#         sigma_list = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12]
#     if alphas_to_check is None:
#         alphas_to_check = [2.0, 8.0, 64.0, 256.0]
#
#     print("\n[DEBUG] sigma_M -> one-step subsampled RDP")
#     print(
#         f"[DEBUG] delta0={accountant.delta0:.3e}, "
#         f"M={accountant.M}, C={accountant.clipping_norm}, sample_rate={sample_rate:.6g}"
#     )
#     for s in sigma_list:
#         for a in alphas_to_check:
#             try:
#                 val = accountant._step_rdp(alpha=float(a), sigma_M=float(s), sample_rate=sample_rate)
#                 print(f"[DEBUG] sigma_M={s:.3e}, alpha={a:.2f}, step_rdp={val}")
#             except Exception as e:
#                 print(f"[DEBUG] sigma_M={s:.3e}, alpha={a:.2f}, step_rdp=ERROR: {e}")
#
#
# def debug_base_rdp_curve(
#     *,
#     accountant: ProductConditionalRDPAccountant,
#     sigma_list: Optional[Sequence[float]] = None,
#     alphas_to_check: Optional[Sequence[float]] = None,
# ):
#     if sigma_list is None:
#         sigma_list = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12]
#     if alphas_to_check is None:
#         alphas_to_check = [2.0, 8.0, 64.0, 256.0]
#
#     print("\n[DEBUG] sigma_M -> base conditional RDP (no subsampling)")
#     print(
#         f"[DEBUG] delta0={accountant.delta0:.3e}, "
#         f"M={accountant.M}, C={accountant.clipping_norm}"
#     )
#     for s in sigma_list:
#         for a in alphas_to_check:
#             try:
#                 val = product_conditional_rdp_single_alpha(
#                     alpha=float(a),
#                     clipping_norm=accountant.clipping_norm,
#                     sigma_M=float(s),
#                     M=accountant.M,
#                     delta0=accountant.delta0,
#                 )
#                 print(f"[DEBUG] sigma_M={s:.3e}, alpha={a:.2f}, base_rdp={val}")
#             except Exception as e:
#                 print(f"[DEBUG] sigma_M={s:.3e}, alpha={a:.2f}, base_rdp=ERROR: {e}")
#
#
# def debug_conversion_floor(
#     *,
#     accountant: ProductConditionalRDPAccountant,
#     target_delta: float,
#     steps: int,
#     alphas_to_check: Optional[Sequence[float]] = None,
# ):
#     if alphas_to_check is None:
#         alphas_to_check = [2.0, 8.0, 64.0, 256.0]
#
#     delta_rdp = target_delta - steps * accountant.delta0
#     print("\n[DEBUG] conversion floor")
#     print(
#         f"[DEBUG] target_delta={target_delta:.3e}, "
#         f"delta0={accountant.delta0:.3e}, steps={steps}, "
#         f"delta_rdp={delta_rdp:.3e}"
#     )
#     if delta_rdp <= 0:
#         print("[DEBUG] delta_rdp <= 0, conversion invalid")
#         return
#
#     for a in alphas_to_check:
#         if a <= 1:
#             continue
#         floor = math.log(1.0 / delta_rdp) / (a - 1.0)
#         print(f"[DEBUG] alpha={a:.2f}, conversion_floor={floor}")
#
#
# # ============================================================
# # Search sigma_M directly (delta0 fixed)
# # ============================================================
#
# def get_product_sigma_M_for_fixed_delta0(
#     *,
#     target_epsilon: float,
#     target_delta: float,
#     sample_rate: float,
#     steps: int,
#     accountant: ProductConditionalRDPAccountant,
#     tol: float = 1e-4,
#     max_sigma_M: float = 1e12,
#     init_hi: Optional[float] = None,
# ) -> float:
#     """
#     For a fixed delta0, find the smallest sigma_M such that
#         accountant.get_epsilon(delta=target_delta) <= target_epsilon
#
#     Returns:
#         sigma_M_star
#     """
#     if steps <= 0:
#         raise ValueError("steps must be positive.")
#
#     saved_history = list(accountant.history)
#     eps_cache = {}
#
#     def eps_for_sigma(sigma_M: float) -> float:
#         key = float(sigma_M)
#         if key in eps_cache:
#             return eps_cache[key]
#
#         if (not math.isfinite(key)) or key <= 0.0:
#             eps_cache[key] = float("inf")
#             return eps_cache[key]
#
#         try:
#             accountant.history = [(key, sample_rate, steps)]
#             eps_val = accountant.get_epsilon(delta=target_delta)
#             if not math.isfinite(eps_val):
#                 eps_val = float("inf")
#             eps_cache[key] = eps_val
#             return eps_val
#         except Exception:
#             eps_cache[key] = float("inf")
#             return eps_cache[key]
#
#     try:
#         lo = 1e-12
#         hi = float(init_hi) if (init_hi is not None and init_hi > lo) else 1.0
#
#         eps_hi = eps_for_sigma(hi)
#         while (not math.isfinite(eps_hi) or eps_hi > target_epsilon) and hi < max_sigma_M:
#             hi *= 2.0
#             eps_hi = eps_for_sigma(hi)
#
#         if not math.isfinite(eps_hi) or eps_hi > target_epsilon:
#             raise RuntimeError(
#                 f"No feasible sigma_M found in [1e-12, {max_sigma_M:.1e}] "
#                 f"for fixed delta0={accountant.delta0:.3e}."
#             )
#
#         while hi - lo > tol:
#             mid = 0.5 * (lo + hi)
#             eps_mid = eps_for_sigma(mid)
#
#             if math.isfinite(eps_mid) and eps_mid <= target_epsilon:
#                 hi = mid
#             else:
#                 lo = mid
#
#         return float(hi)
#
#     finally:
#         accountant.history = saved_history
#
#
# # ============================================================
# # PrivacyEngine
# # ============================================================
#
# class ProductPrivacyEngine(PrivacyEngine):
#     """
#     PrivacyEngine for product noise.
#
#     Reinterpretation:
#         noise_multiplier == sigma_M
#
#     New behavior:
#         make_private_with_epsilon() fixes a tiny delta0 and searches sigma_M directly.
#     """
#
#     def __init__(
#         self,
#         *,
#         clipping_norm: float,
#         delta0: float = FIXED_DELTA0,
#         M: Optional[int] = None,
#         alphas: Optional[Sequence[float]] = None,
#         secure_mode: bool = False,
#         k: int = FIXED_K,
#         max_sigma_M: float = 1e12,
#         sigma_search_tol: float = 1e-4,
#         debug_when_target_epsilon_is_3: bool = True,
#     ):
#         super().__init__(accountant="rdp", secure_mode=secure_mode)
#
#         self.accountant = ProductConditionalRDPAccountant(
#             clipping_norm=clipping_norm,
#             M=M if M is not None else 3,
#             delta0=delta0,
#             alphas=alphas,
#         )
#
#         self._product_M = M
#         self._fixed_k = int(k)  # kept for compatibility / theory reference
#         self._fixed_delta0 = float(delta0)
#         self._max_sigma_M = float(max_sigma_M)
#         self._sigma_search_tol = float(sigma_search_tol)
#         self._debug_when_target_epsilon_is_3 = bool(debug_when_target_epsilon_is_3)
#
#     def _prepare_optimizer(
#         self,
#         *,
#         optimizer,
#         noise_multiplier,
#         max_grad_norm,
#         expected_batch_size,
#         loss_reduction="mean",
#         distributed=False,
#         clipping="flat",
#         noise_generator=None,
#         grad_sample_mode="hooks",
#         **kwargs,
#     ):
#         if clipping != "flat":
#             raise NotImplementedError(
#                 "Current ProductDPOptimizer supports only flat clipping."
#             )
#         if distributed:
#             raise NotImplementedError(
#                 "Distributed mode is not implemented in this version."
#             )
#         if grad_sample_mode != "hooks":
#             raise NotImplementedError(
#                 "Current ProductDPOptimizer supports only hooks mode."
#             )
#
#         if isinstance(optimizer, DPOptimizer):
#             optimizer = optimizer.original_optimizer
#
#         generator = None
#         if self.secure_mode:
#             generator = self.secure_rng
#         elif noise_generator is not None:
#             generator = noise_generator
#
#         return ProductDPOptimizer(
#             optimizer=optimizer,
#             noise_multiplier=noise_multiplier,
#             max_grad_norm=max_grad_norm,
#             expected_batch_size=expected_batch_size,
#             loss_reduction=loss_reduction,
#             generator=generator,
#             secure_mode=self.secure_mode,
#             M=self._product_M,
#             **kwargs,
#         )
#
#     def make_private(
#         self,
#         *,
#         module,
#         optimizer,
#         data_loader,
#         noise_multiplier,
#         max_grad_norm,
#         batch_first=True,
#         loss_reduction="mean",
#         poisson_sampling=True,
#         clipping="flat",
#         noise_generator=None,
#         grad_sample_mode="hooks",
#         **kwargs,
#     ):
#         if self._product_M is None:
#             self._product_M = sum(
#                 p.numel() for p in module.parameters() if p.requires_grad
#             )
#
#         self.accountant.M = self._product_M
#         self.accountant.clipping_norm = float(max_grad_norm)
#         self.accountant.delta0 = float(self._fixed_delta0)
#
#         return super().make_private(
#             module=module,
#             optimizer=optimizer,
#             data_loader=data_loader,
#             noise_multiplier=noise_multiplier,
#             max_grad_norm=max_grad_norm,
#             batch_first=batch_first,
#             loss_reduction=loss_reduction,
#             poisson_sampling=poisson_sampling,
#             clipping=clipping,
#             noise_generator=noise_generator,
#             grad_sample_mode=grad_sample_mode,
#             **kwargs,
#         )
#
#     def make_private_with_epsilon(
#         self,
#         *,
#         module,
#         optimizer,
#         data_loader,
#         target_epsilon,
#         target_delta,
#         epochs,
#         max_grad_norm,
#         batch_first=True,
#         loss_reduction="mean",
#         poisson_sampling=True,
#         clipping="flat",
#         noise_generator=None,
#         grad_sample_mode="hooks",
#         **kwargs,
#     ):
#         if not poisson_sampling:
#             raise NotImplementedError(
#                 "This implementation follows the standard Poisson-subsampling "
#                 "RDP workflow. Set poisson_sampling=True."
#             )
#
#         if self._product_M is None:
#             self._product_M = sum(
#                 p.numel() for p in module.parameters() if p.requires_grad
#             )
#
#         self.accountant.M = self._product_M
#         self.accountant.clipping_norm = float(max_grad_norm)
#         self.accountant.delta0 = float(self._fixed_delta0)
#
#         sample_rate = 1.0 / len(data_loader)
#         steps = int(epochs * len(data_loader))
#
#         if steps * self._fixed_delta0 >= target_delta:
#             raise ValueError(
#                 f"Fixed delta0={self._fixed_delta0:.3e} is too large for "
#                 f"target_delta={target_delta:.3e} and steps={steps}. "
#                 f"Need steps * delta0 < target_delta."
#             )
#
#         if self._debug_when_target_epsilon_is_3 and abs(float(target_epsilon) - 3.0) < 1e-12:
#             debug_base_rdp_curve(
#                 accountant=self.accountant,
#                 sigma_list=[1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12],
#                 alphas_to_check=[2.0, 8.0, 64.0, 256.0],
#             )
#             debug_step_rdp_curve(
#                 accountant=self.accountant,
#                 sample_rate=sample_rate,
#                 sigma_list=[1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12],
#                 alphas_to_check=[2.0, 8.0, 64.0, 256.0],
#             )
#             debug_conversion_floor(
#                 accountant=self.accountant,
#                 target_delta=target_delta,
#                 steps=steps,
#                 alphas_to_check=[2.0, 8.0, 64.0, 256.0],
#             )
#             debug_sigma_curve(
#                 accountant=self.accountant,
#                 target_delta=target_delta,
#                 sample_rate=sample_rate,
#                 steps=steps,
#                 sigma_list=[1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12],
#             )
#
#         sigma_M = get_product_sigma_M_for_fixed_delta0(
#             target_epsilon=target_epsilon,
#             target_delta=target_delta,
#             sample_rate=sample_rate,
#             steps=steps,
#             accountant=self.accountant,
#             tol=self._sigma_search_tol,
#             max_sigma_M=self._max_sigma_M,
#         )
#
#         # for reference only: convert searched sigma_M back to theorem-style epsilon_step
#         epsilon_step = (2.0 * max_grad_norm) * math.sqrt(
#             product_t_squared(M=self.accountant.M, k=self._fixed_k)
#         ) / sigma_M
#
#         print(
#             f"[product privacy engine] fixed_delta0={self._fixed_delta0:.3e}, "
#             f"sigma_M={sigma_M:.6g}, implied_epsilon_step={epsilon_step:.6g}"
#         )
#
#         return self.make_private(
#             module=module,
#             optimizer=optimizer,
#             data_loader=data_loader,
#             noise_multiplier=sigma_M,
#             max_grad_norm=max_grad_norm,
#             batch_first=batch_first,
#             loss_reduction=loss_reduction,
#             poisson_sampling=poisson_sampling,
#             clipping=clipping,
#             noise_generator=noise_generator,
#             grad_sample_mode=grad_sample_mode,
#             **kwargs,
#         )

import numpy as np
import math
from typing import List, Optional, Sequence, Tuple

import torch

from opacus.accountants.accountant import IAccountant
from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed
from opacus.privacy_engine import PrivacyEngine


# ============================================================
# Default constants
# ============================================================

FIXED_DELTA0 = 1e-12


# ============================================================
# Basic math utilities
# ============================================================

def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _log_2phi(x: float) -> float:
    return math.log1p(math.erf(x / math.sqrt(2.0)))


def _log_add(logx: float, logy: float) -> float:
    if logx == -math.inf:
        return logy
    if logy == -math.inf:
        return logx
    if logx > logy:
        return logx + math.log1p(math.exp(logy - logx))
    return logy + math.log1p(math.exp(logx - logy))


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


# ============================================================
# Product-noise mechanism parameters
# ============================================================

def product_t_squared(
    *,
    M: int,
    k: int,
) -> float:
    """
    Original theorem-based t^2(M, k):

        t^2 = 2 * k^(4/M) * ((M/4 + 3/2)^(1 + 4/M)) / exp(1 + 2/M)
    """
    if M <= 0:
        raise ValueError("M must be positive.")
    if k <= 0:
        raise ValueError("k must be positive.")

    return (
        2.0
        * (k ** (4.0 / M))
        * (((M / 4.0) + 1.5) ** (1.0 + 4.0 / M))
        / (math.e ** (1.0 + 2.0 / M))
    )


def sigma_M_from_epsilon_step(
    *,
    epsilon_step: float,
    clipping_norm: float,
    M: int,
    k: int,
) -> float:
    """
    Original theorem-based calibration:

        sigma_M = (Delta_2 f / epsilon_step) * sqrt(t^2(M, k))

    with Delta_2 f = 2C.
    """
    if epsilon_step <= 0:
        raise ValueError("epsilon_step must be positive.")

    delta2f = 2.0 * clipping_norm
    return (delta2f / epsilon_step) * math.sqrt(product_t_squared(M=M, k=k))


def product_lambda_from_sigma_M(
    *,
    clipping_norm: float,
    sigma_M: float,
) -> float:
    """
    For gradient perturbation with clipped gradient sum:
        Delta_2 f = 2C
    so
        lambda = Delta_2 f / sigma_M = (2C) / sigma_M
    """
    if sigma_M <= 0:
        raise ValueError("sigma_M must be positive.")
    return (2.0 * clipping_norm) / sigma_M


def product_C_M(
    *,
    M: int,
    delta0: float,
) -> float:
    """
    C_M = 1 / cos(theta_0)
        = ( sqrt(pi) * Gamma(M/2) / (delta0 * Gamma((M-1)/2)) )^(1/(M-2))
    """
    if M <= 2:
        raise ValueError("This formula requires M > 2.")
    if not (0.0 < delta0 < 1.0):
        raise ValueError("delta0 must be in (0, 1).")

    log_num = 0.5 * math.log(math.pi) + math.lgamma(M / 2.0)
    log_den = math.log(delta0) + math.lgamma((M - 1.0) / 2.0)
    log_ratio = log_num - log_den
    return math.exp(log_ratio / (M - 2.0))


def product_conditional_rdp_single_alpha(
    *,
    alpha: float,
    clipping_norm: float,
    sigma_M: float,
    M: int,
    delta0: float,
) -> float:
    """
    Paper's conditional single-step RDP formula:

    eps_alpha =
      1/(alpha-1) * [ alpha * lambda^2 * C_M
                      + (alpha * lambda * C_M)^2 / 2
                      + log(2 Phi(alpha * lambda * C_M)) ]
    """
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1.")

    lam = product_lambda_from_sigma_M(
        clipping_norm=clipping_norm,
        sigma_M=sigma_M,
    )
    c_m = product_C_M(M=M, delta0=delta0)

    a = alpha * lam * c_m
    return (
        alpha * (lam ** 2) * c_m
        + 0.5 * (a ** 2)
        + _log_2phi(a)
    ) / (alpha - 1.0)


# ============================================================
# Standard-RDP-style Poisson subsampling workflow
# ============================================================

def subsampled_rdp_like_standard_workflow(
    *,
    alpha: float,
    sample_rate: float,
    base_rdp_fn,
) -> float:
    """
    Keep the same engineering workflow as standard RDP-based accounting.

    For integer alpha >= 2, use a generic log-sum style upper bound:
        eps_sub(alpha) <= 1/(alpha-1) * log(
            1 + sum_{j=2}^{alpha} C(alpha, j) q^j (1-q)^(alpha-j) exp((j-1) eps_base(j))
        )

    For non-integer alpha, linearly interpolate between neighboring integers.
    """
    q = sample_rate
    if not (0.0 < q <= 1.0):
        raise ValueError("sample_rate must be in (0, 1].")

    if q == 1.0:
        return base_rdp_fn(alpha)

    if alpha <= 1.0:
        raise ValueError("alpha must be > 1.")

    def _integer_alpha_bound(a_int: int) -> float:
        if a_int <= 1:
            raise ValueError("Integer alpha must be >= 2")

        # j = 0 and j = 1 terms, without privacy penalty
        logA = _log_add(
            a_int * math.log(1.0 - q),
            math.log(a_int) + math.log(q) + (a_int - 1) * math.log(1.0 - q),
        )

        # j >= 2 terms with privacy penalty
        for j in range(2, a_int + 1):
            eps_j = base_rdp_fn(float(j))
            term = (
                _log_comb(a_int, j)
                + j * math.log(q)
                + (a_int - j) * math.log(1.0 - q)
                + (j - 1.0) * eps_j
            )
            logA = _log_add(logA, term)

        return logA / (a_int - 1.0)

    if float(alpha).is_integer():
        return _integer_alpha_bound(int(alpha))

    lo = max(2, int(math.floor(alpha)))
    hi = int(math.ceil(alpha))
    eps_lo = _integer_alpha_bound(lo)
    eps_hi = _integer_alpha_bound(hi)

    weight = alpha - lo
    return (1.0 - weight) * eps_lo + weight * eps_hi


# ============================================================
# Accountant
# ============================================================

class ProductConditionalRDPAccountant(IAccountant):
    """
    Product-noise conditional RDP accountant.

    Workflow:
      1) base single-step RDP: paper's conditional RDP formula
      2) subsampling: RDP-style Poisson-subsampling workflow
      3) composition: additive in RDP
      4) final DP conversion:
             delta_rdp = target_delta - T * delta0
         epsilon(delta) = min_alpha [ total_rdp(alpha) + log(1/delta_rdp)/(alpha-1) ]

    Interpretation:
      noise_multiplier is reinterpreted as sigma_M.
    """

    DEFAULT_ALPHAS = (
        [1.25, 1.5, 1.75]
        + [2 + i for i in range(0, 64)]
        + [128, 256]
    )

    def __init__(
        self,
        *,
        clipping_norm: float,
        M: int,
        delta0: float,
        alphas: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.clipping_norm = float(clipping_norm)
        self.M = int(M)
        self.delta0 = float(delta0)
        self.alphas = list(alphas) if alphas is not None else list(self.DEFAULT_ALPHAS)

    @classmethod
    def mechanism(cls) -> str:
        return "product_conditional_rdp"

    def __len__(self) -> int:
        return len(self.history)

    def step(self, *, noise_multiplier: float, sample_rate: float):
        sigma_M = float(noise_multiplier)
        if sigma_M <= 0:
            raise ValueError("sigma_M must be positive.")

        if len(self.history) > 0:
            prev_sigma, prev_q, prev_steps = self.history[-1]
            if prev_sigma == sigma_M and prev_q == sample_rate:
                self.history[-1] = (prev_sigma, prev_q, prev_steps + 1)
                return

        self.history.append((sigma_M, sample_rate, 1))

    def _step_rdp(self, alpha: float, sigma_M: float, sample_rate: float) -> float:
        def base_rdp_fn(a: float) -> float:
            return product_conditional_rdp_single_alpha(
                alpha=a,
                clipping_norm=self.clipping_norm,
                sigma_M=sigma_M,
                M=self.M,
                delta0=self.delta0,
            )

        return subsampled_rdp_like_standard_workflow(
            alpha=alpha,
            sample_rate=sample_rate,
            base_rdp_fn=base_rdp_fn,
        )

    def get_privacy_spent(
        self,
        *,
        delta: float,
        alphas: Optional[Sequence[float]] = None,
    ) -> Tuple[float, float]:
        if not self.history:
            return 0.0, 0.0

        alphas = list(alphas) if alphas is not None else self.alphas
        total_steps = sum(steps for _, _, steps in self.history)

        delta_rdp = delta - total_steps * self.delta0
        if delta_rdp <= 0.0:
            return float("inf"), float("nan")

        best_eps = float("inf")
        best_alpha = float("nan")

        for alpha in alphas:
            if alpha <= 1.0:
                continue

            total_rdp = 0.0
            for sigma_M, q, steps in self.history:
                total_rdp += steps * self._step_rdp(alpha, sigma_M, q)

            eps = total_rdp + math.log(1.0 / delta_rdp) / (alpha - 1.0)
            if eps < best_eps:
                best_eps = eps
                best_alpha = float(alpha)

        return float(best_eps), float(best_alpha)

    def get_epsilon(self, delta: float, **kwargs) -> float:
        eps, _ = self.get_privacy_spent(delta=delta, **kwargs)
        return float(eps)


# ============================================================
# Optimizer
# ============================================================

class ProductDPOptimizer(DPOptimizer):
    """
    Product-noise version of Opacus DPOptimizer.

    In this implementation, `noise_multiplier` is interpreted as sigma_M.
    Product noise:
        n = sigma_M * R * h
    where
        R ~ chi_1 = |N(0,1)|
        h ~ Uniform(S^{M-1})
    """

    def __init__(
        self,
        optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        M: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            **kwargs,
        )
        self.M = M

    @property
    def sigma_M(self) -> float:
        return float(self.noise_multiplier)

    def _infer_M(self) -> int:
        if self.M is not None:
            return int(self.M)
        return sum(p.numel() for p in self.params if p.requires_grad)

    def _generate_product_noise(self, *, active_params: List[torch.nn.Parameter]):
        raw_noises = []
        for p in active_params:
            ref = p.summed_grad
            z = torch.normal(
                mean=0.0,
                std=1.0,
                size=ref.shape,
                device=ref.device,
                dtype=ref.dtype,
                generator=self.generator,
            )
            raw_noises.append(z)

        eps = torch.finfo(raw_noises[0].dtype).eps
        magnitude = torch.sqrt(sum(torch.sum(z ** 2) for z in raw_noises)).clamp_min(eps)
        h_list = [z / magnitude for z in raw_noises]

        ref0 = active_params[0].summed_grad
        R = torch.abs(
            torch.normal(
                mean=0.0,
                std=1.0,
                size=(1,),
                device=ref0.device,
                dtype=ref0.dtype,
                generator=self.generator,
            )
        )

        sigma_tensor = torch.tensor(self.sigma_M, device=ref0.device, dtype=ref0.dtype)
        return [sigma_tensor * R * h for h in h_list]

    def add_noise(self):
        if not self.params:
            return

        active_params = [p for p in self.params if p.summed_grad is not None]
        if not active_params:
            return

        for p in active_params:
            _check_processed_flag(p.summed_grad)

        _ = self._infer_M()

        noises = self._generate_product_noise(active_params=active_params)

        for p, noise in zip(active_params, noises):
            p.grad = (p.summed_grad + noise).view_as(p)
            _mark_as_processed(p.summed_grad)


# ============================================================
# Debug helpers
# ============================================================

def debug_sigma_curve(
    *,
    accountant: ProductConditionalRDPAccountant,
    target_delta: float,
    sample_rate: float,
    steps: int,
    sigma_list: Optional[Sequence[float]] = None,
):
    if sigma_list is None:
        sigma_list = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12]

    saved_history = list(accountant.history)
    try:
        print("\n[DEBUG] sigma_M -> total epsilon")
        print(
            f"[DEBUG] delta0={accountant.delta0:.3e}, "
            f"M={accountant.M}, C={accountant.clipping_norm}, "
            f"sample_rate={sample_rate:.6g}, steps={steps}, target_delta={target_delta:.3e}"
        )
        for s in sigma_list:
            try:
                accountant.history = [(float(s), sample_rate, steps)]
                eps = accountant.get_epsilon(delta=target_delta)
                print(f"[DEBUG] sigma_M={s:.3e}, total_epsilon={eps}")
            except Exception as e:
                print(f"[DEBUG] sigma_M={s:.3e}, total_epsilon=ERROR: {e}")
    finally:
        accountant.history = saved_history


def debug_step_rdp_curve(
    *,
    accountant: ProductConditionalRDPAccountant,
    sample_rate: float,
    sigma_list: Optional[Sequence[float]] = None,
    alphas_to_check: Optional[Sequence[float]] = None,
):
    if sigma_list is None:
        sigma_list = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12]
    if alphas_to_check is None:
        alphas_to_check = [2.0, 8.0, 64.0, 256.0]

    print("\n[DEBUG] sigma_M -> one-step subsampled RDP")
    print(
        f"[DEBUG] delta0={accountant.delta0:.3e}, "
        f"M={accountant.M}, C={accountant.clipping_norm}, sample_rate={sample_rate:.6g}"
    )
    for s in sigma_list:
        for a in alphas_to_check:
            try:
                val = accountant._step_rdp(alpha=float(a), sigma_M=float(s), sample_rate=sample_rate)
                print(f"[DEBUG] sigma_M={s:.3e}, alpha={a:.2f}, step_rdp={val}")
            except Exception as e:
                print(f"[DEBUG] sigma_M={s:.3e}, alpha={a:.2f}, step_rdp=ERROR: {e}")


def debug_base_rdp_curve(
    *,
    accountant: ProductConditionalRDPAccountant,
    sigma_list: Optional[Sequence[float]] = None,
    alphas_to_check: Optional[Sequence[float]] = None,
):
    if sigma_list is None:
        sigma_list = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12]
    if alphas_to_check is None:
        alphas_to_check = [2.0, 8.0, 64.0, 256.0]

    print("\n[DEBUG] sigma_M -> base conditional RDP (no subsampling)")
    print(
        f"[DEBUG] delta0={accountant.delta0:.3e}, "
        f"M={accountant.M}, C={accountant.clipping_norm}"
    )
    for s in sigma_list:
        for a in alphas_to_check:
            try:
                val = product_conditional_rdp_single_alpha(
                    alpha=float(a),
                    clipping_norm=accountant.clipping_norm,
                    sigma_M=float(s),
                    M=accountant.M,
                    delta0=accountant.delta0,
                )
                print(f"[DEBUG] sigma_M={s:.3e}, alpha={a:.2f}, base_rdp={val}")
            except Exception as e:
                print(f"[DEBUG] sigma_M={s:.3e}, alpha={a:.2f}, base_rdp=ERROR: {e}")


def debug_conversion_floor(
    *,
    accountant: ProductConditionalRDPAccountant,
    target_delta: float,
    steps: int,
    alphas_to_check: Optional[Sequence[float]] = None,
):
    if alphas_to_check is None:
        alphas_to_check = [2.0, 8.0, 64.0, 256.0]

    delta_rdp = target_delta - steps * accountant.delta0
    print("\n[DEBUG] conversion floor")
    print(
        f"[DEBUG] target_delta={target_delta:.3e}, "
        f"delta0={accountant.delta0:.3e}, steps={steps}, "
        f"delta_rdp={delta_rdp:.3e}"
    )
    if delta_rdp <= 0:
        print("[DEBUG] delta_rdp <= 0, conversion invalid")
        return

    for a in alphas_to_check:
        if a <= 1:
            continue
        floor = math.log(1.0 / delta_rdp) / (a - 1.0)
        print(f"[DEBUG] alpha={a:.2f}, conversion_floor={floor}")


# ============================================================
# Search sigma_M directly (delta0 fixed)
# ============================================================

def get_product_sigma_M_for_fixed_delta0(
    *,
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    steps: int,
    accountant: ProductConditionalRDPAccountant,
    tol: float = 1e-4,
    max_sigma_M: float = 1e12,
    init_hi: Optional[float] = None,
) -> float:
    """
    For a fixed delta0, find the smallest sigma_M such that
        accountant.get_epsilon(delta=target_delta) <= target_epsilon

    Returns:
        sigma_M_star
    """
    if steps <= 0:
        raise ValueError("steps must be positive.")

    saved_history = list(accountant.history)
    eps_cache = {}

    def eps_for_sigma(sigma_M: float) -> float:
        key = float(sigma_M)
        if key in eps_cache:
            return eps_cache[key]

        if (not math.isfinite(key)) or key <= 0.0:
            eps_cache[key] = float("inf")
            return eps_cache[key]

        try:
            accountant.history = [(key, sample_rate, steps)]
            eps_val = accountant.get_epsilon(delta=target_delta)
            if not math.isfinite(eps_val):
                eps_val = float("inf")
            eps_cache[key] = eps_val
            return eps_val
        except Exception:
            eps_cache[key] = float("inf")
            return eps_cache[key]

    try:
        lo = 1e-12
        hi = float(init_hi) if (init_hi is not None and init_hi > lo) else 1.0

        eps_hi = eps_for_sigma(hi)
        while (not math.isfinite(eps_hi) or eps_hi > target_epsilon) and hi < max_sigma_M:
            hi *= 2.0
            eps_hi = eps_for_sigma(hi)

        if not math.isfinite(eps_hi) or eps_hi > target_epsilon:
            raise RuntimeError(
                f"No feasible sigma_M found in [1e-12, {max_sigma_M:.1e}] "
                f"for fixed delta0={accountant.delta0:.3e}."
            )

        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            eps_mid = eps_for_sigma(mid)

            if math.isfinite(eps_mid) and eps_mid <= target_epsilon:
                hi = mid
            else:
                lo = mid

        return float(hi)

    finally:
        accountant.history = saved_history


# ============================================================
# PrivacyEngine
# ============================================================

class ProductPrivacyEngine(PrivacyEngine):
    """
    PrivacyEngine for product noise.

    Reinterpretation:
        noise_multiplier == sigma_M

    New behavior:
        make_private_with_epsilon() fixes a tiny delta0 and searches sigma_M directly.
    """

    def __init__(
        self,
        *,
        clipping_norm: float,
        delta0: float = FIXED_DELTA0,
        M: Optional[int] = None,
        alphas: Optional[Sequence[float]] = None,
        secure_mode: bool = False,
        k: int,
        max_sigma_M: float = 1e12,
        sigma_search_tol: float = 1e-4,
        debug_when_target_epsilon_is_3: bool = True,
    ):
        super().__init__(accountant="rdp", secure_mode=secure_mode)

        if k <= 0:
            raise ValueError("k must be positive.")

        self.accountant = ProductConditionalRDPAccountant(
            clipping_norm=clipping_norm,
            M=M if M is not None else 3,
            delta0=delta0,
            alphas=alphas,
        )

        self._product_M = M
        self._k = int(k)
        self._fixed_delta0 = float(delta0)
        self._max_sigma_M = float(max_sigma_M)
        self._sigma_search_tol = float(sigma_search_tol)
        self._debug_when_target_epsilon_is_3 = bool(debug_when_target_epsilon_is_3)

    def _prepare_optimizer(
        self,
        *,
        optimizer,
        noise_multiplier,
        max_grad_norm,
        expected_batch_size,
        loss_reduction="mean",
        distributed=False,
        clipping="flat",
        noise_generator=None,
        grad_sample_mode="hooks",
        **kwargs,
    ):
        if clipping != "flat":
            raise NotImplementedError(
                "Current ProductDPOptimizer supports only flat clipping."
            )
        if distributed:
            raise NotImplementedError(
                "Distributed mode is not implemented in this version."
            )
        if grad_sample_mode != "hooks":
            raise NotImplementedError(
                "Current ProductDPOptimizer supports only hooks mode."
            )

        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        return ProductDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
            M=self._product_M,
            **kwargs,
        )

    def make_private(
        self,
        *,
        module,
        optimizer,
        data_loader,
        noise_multiplier,
        max_grad_norm,
        batch_first=True,
        loss_reduction="mean",
        poisson_sampling=True,
        clipping="flat",
        noise_generator=None,
        grad_sample_mode="hooks",
        **kwargs,
    ):
        if self._product_M is None:
            self._product_M = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )

        self.accountant.M = self._product_M
        self.accountant.clipping_norm = float(max_grad_norm)
        self.accountant.delta0 = float(self._fixed_delta0)

        return super().make_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            poisson_sampling=poisson_sampling,
            clipping=clipping,
            noise_generator=noise_generator,
            grad_sample_mode=grad_sample_mode,
            **kwargs,
        )

    def make_private_with_epsilon(
        self,
        *,
        module,
        optimizer,
        data_loader,
        target_epsilon,
        target_delta,
        epochs,
        max_grad_norm,
        batch_first=True,
        loss_reduction="mean",
        poisson_sampling=True,
        clipping="flat",
        noise_generator=None,
        grad_sample_mode="hooks",
        **kwargs,
    ):
        if not poisson_sampling:
            raise NotImplementedError(
                "This implementation follows the standard Poisson-subsampling "
                "RDP workflow. Set poisson_sampling=True."
            )

        if self._product_M is None:
            self._product_M = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )

        self.accountant.M = self._product_M
        self.accountant.clipping_norm = float(max_grad_norm)
        self.accountant.delta0 = float(self._fixed_delta0)

        sample_rate = 1.0 / len(data_loader)
        steps = int(epochs * len(data_loader))

        if steps * self._fixed_delta0 >= target_delta:
            raise ValueError(
                f"Fixed delta0={self._fixed_delta0:.3e} is too large for "
                f"target_delta={target_delta:.3e} and steps={steps}. "
                f"Need steps * delta0 < target_delta."
            )

        if self._debug_when_target_epsilon_is_3 and abs(float(target_epsilon) - 3.0) < 1e-12:
            debug_base_rdp_curve(
                accountant=self.accountant,
                sigma_list=[1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12],
                alphas_to_check=[2.0, 8.0, 64.0, 256.0],
            )
            debug_step_rdp_curve(
                accountant=self.accountant,
                sample_rate=sample_rate,
                sigma_list=[1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12],
                alphas_to_check=[2.0, 8.0, 64.0, 256.0],
            )
            debug_conversion_floor(
                accountant=self.accountant,
                target_delta=target_delta,
                steps=steps,
                alphas_to_check=[2.0, 8.0, 64.0, 256.0],
            )
            debug_sigma_curve(
                accountant=self.accountant,
                target_delta=target_delta,
                sample_rate=sample_rate,
                steps=steps,
                sigma_list=[1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12],
            )

        sigma_M = get_product_sigma_M_for_fixed_delta0(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=steps,
            accountant=self.accountant,
            tol=self._sigma_search_tol,
            max_sigma_M=self._max_sigma_M,
        )

        # for reference only: convert searched sigma_M back to theorem-style epsilon_step
        epsilon_step = (2.0 * max_grad_norm) * math.sqrt(
            product_t_squared(M=self.accountant.M, k=self._k)
        ) / sigma_M

        print(
            f"[product privacy engine] fixed_delta0={self._fixed_delta0:.3e}, "
            f"k={self._k}, sigma_M={sigma_M:.6g}, implied_epsilon_step={epsilon_step:.6g}"
        )

        return self.make_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=sigma_M,
            max_grad_norm=max_grad_norm,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            poisson_sampling=poisson_sampling,
            clipping=clipping,
            noise_generator=noise_generator,
            grad_sample_mode=grad_sample_mode,
            **kwargs,
        )
"""
quantum_opt.py

QAOA-based portfolio optimization. Formulates asset selection as a QUBO via
Qiskit's PortfolioOptimization application, then solves it with QAOA on
either a local Aer simulator or real IBM Quantum hardware.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from src.utils import OptimizationResult, portfolio_performance

load_dotenv()


def _build_sampler_and_transpiler(backend: str):
    """
    Returns (sampler, transpiler) for the requested backend.

    V2 primitives expect circuits already decomposed into basis gates —
    QAOA's high-level ansatz instruction isn't natively understood by Aer
    or real hardware, so we build a preset pass manager and hand it to
    QAOA's `transpiler` argument.

    backend:
        "simulator"    -> local Aer simulator (fast, free, default)
        "ibm_hardware" -> real IBM Quantum device via qiskit-ibm-runtime
                           (requires IBM_QUANTUM_TOKEN in .env)
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    if backend == "simulator":
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

        backend_obj = AerSimulator()
        sampler = AerSamplerV2()
        transpiler = generate_preset_pass_manager(optimization_level=1, backend=backend_obj)
        return sampler, transpiler

    if backend == "ibm_hardware":
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit_ibm_runtime import SamplerV2 as IBMSampler

        token = os.getenv("IBM_QUANTUM_TOKEN")
        instance = os.getenv("IBM_QUANTUM_INSTANCE", "ibm-q/open/main")
        if not token:
            raise RuntimeError(
                "IBM_QUANTUM_TOKEN not set. Copy .env.example to .env and add "
                "your token from https://quantum.ibm.com/ before using "
                "--backend ibm_hardware."
            )

        service = QiskitRuntimeService(channel="ibm_quantum", token=token, instance=instance)
        least_busy = service.least_busy(operational=True, simulator=False)
        sampler = IBMSampler(mode=least_busy)
        transpiler = generate_preset_pass_manager(optimization_level=1, backend=least_busy)
        return sampler, transpiler

    raise ValueError(f"Unknown backend: {backend!r}. Use 'simulator' or 'ibm_hardware'.")


def qaoa_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    budget: int,
    risk_factor: float = 0.5,
    reps: int = 2,
    max_iter: int = 200,
    backend: str = "simulator",
    seed: int = 42,
) -> OptimizationResult:
    """
    Solve the cardinality-constrained portfolio selection problem with QAOA.
    """
    start = time.perf_counter()
    algorithm_globals.random_seed = seed

    portfolio = PortfolioOptimization(
        expected_returns=mean_returns.values,
        covariances=cov_matrix.values,
        risk_factor=risk_factor,
        budget=budget,
    )
    qp = portfolio.to_quadratic_program()

    sampler, transpiler = _build_sampler_and_transpiler(backend)
    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=max_iter),
        reps=reps,
        transpiler=transpiler,
    )
    optimizer = MinimumEigenOptimizer(qaoa)

    result = optimizer.solve(qp)
    runtime = time.perf_counter() - start

    selection = np.array(result.x)  # binary selection vector
    n_selected = selection.sum()
    weights = selection / n_selected if n_selected > 0 else selection

    exp_ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)

    backend_label = "aer_simulator" if backend == "simulator" else "ibm_hardware"

    return OptimizationResult(
        method="quantum (QAOA)",
        weights=dict(zip(mean_returns.index, weights)),
        expected_return=exp_ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        runtime_seconds=runtime,
        backend=backend_label,
    )
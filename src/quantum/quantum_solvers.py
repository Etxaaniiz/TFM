import os
# Force JAX to use CPU only to bypass incompatible PJRT CUDA plugins in Colab
os.environ["JAX_PLATFORMS"] = "cpu"

import time
import numpy as np
from typing import Dict, Any
from qrisp import QuantumVariable, dicke_state
from qrisp.qaoa import (
    QAOAProblem, 
    create_QUBO_cost_operator, 
    create_QUBO_cl_cost_function, 
    RX_mixer, 
    XY_mixer
)
from qrisp.jasp import jaspify
from src.utils.utils import get_memory_usage
from src.metrics.metrics import calculate_portfolio_metrics, calculate_qubo_energy
from src.quantum.regularized_qaoa import RegularizedQAOAProblem

def solve_qaoa(
    instance: Dict[str, Any], 
    p: int = 1, 
    maxiter: int = 100, 
    shots: int = 1024
) -> Dict[str, Any]:
    """
    Solves the QUBO formulation using standard QAOA (X-mixer, uniform superposition).
    
    Parameters:
    instance (dict): Problem instance containing N, K, mu, Sigma, Q, lambda_val.
    p (int): QAOA depth/steps (default: 1).
    maxiter (int): Maximum optimization iterations (default: 100).
    shots (int): Number of measurement shots (default: 1024).
    
    Returns:
    dict: Solver results including portfolio metrics and runtime.
    """
    N = instance['N']
    K = instance['K']
    mu = instance['mu']
    Sigma = instance['Sigma']
    Q = instance['Q']
    lambda_val = instance.get('lambda_val', 0.5)
    offset = instance.get('offset', 0.0)
    
    mem_before = get_memory_usage()
    start_time = time.perf_counter()
    
    # 1. Setup QuantumVariable
    qv = QuantumVariable(N)
    
    # 2. Instantiate standard QAOA Problem
    qaoa_prob = QAOAProblem(
        cost_operator=create_QUBO_cost_operator(Q),
        mixer=RX_mixer,
        cl_cost_function=create_QUBO_cl_cost_function(Q)
    )
    
    # 3. Run optimization
    results_dict = qaoa_prob.run(
        qarg=qv,
        depth=p,
        max_iter=maxiter,
        mes_kwargs={"shots": shots}
    )
    
    end_time = time.perf_counter()
    mem_after = get_memory_usage()
    
    # 4. Extract solution
    # Find bitstring with the highest probability/frequency
    best_bitstring = max(results_dict, key=results_dict.get)
    x_sol = np.array([int(bit) for bit in best_bitstring])
    
    runtime = end_time - start_time
    mem_used = mem_after
    
    # 5. Compute metrics
    metrics = calculate_portfolio_metrics(x_sol, mu, Sigma, K, lambda_val)
    energy = calculate_qubo_energy(x_sol, Q) + offset
    
    tickers_list = instance.get('tickers', [])
    selected_tickers = ",".join([tickers_list[i] for i in range(N) if x_sol[i] == 1]) if tickers_list else ""
    num_selected = int(np.sum(x_sol))
    
    results = {
        "dataset": instance['dataset'],
        "solver": "qaoa",
        "N": N,
        "K": K,
        "instance_id": instance['instance_id'],
        "seed": instance['seed'],
        "p": p,
        "objective": metrics['objective'],
        "energy": energy,
        "gap": 0.0, # Will be computed post-solve relative to Gurobi
        "sharpe": metrics['sharpe'],
        "expected_return": metrics['expected_return'],
        "volatility": metrics['volatility'],
        "feasible": metrics['feasible'],
        "runtime_seconds": runtime,
        "memory_mb": mem_used,
        "selected_tickers": selected_tickers,
        "num_selected": num_selected,
        "solution": x_sol,
        "counts": results_dict
    }
    
    return results

def solve_xy(
    instance: Dict[str, Any], 
    p: int = 1, 
    maxiter: int = 100, 
    shots: int = 1024
) -> Dict[str, Any]:
    """
    Solves the QUBO formulation using XY-QAOA (XY-mixer, Dicke state initialization).
    
    Parameters:
    instance (dict): Problem instance containing N, K, mu, Sigma, Q, lambda_val.
    p (int): QAOA depth/steps (default: 1).
    maxiter (int): Maximum optimization iterations (default: 100).
    shots (int): Number of measurement shots (default: 1024).
    
    Returns:
    dict: Solver results including portfolio metrics and runtime.
    """
    N = instance['N']
    K = instance['K']
    mu = instance['mu']
    Sigma = instance['Sigma']
    Q = instance['Q']
    lambda_val = instance.get('lambda_val', 0.5)
    offset = instance.get('offset', 0.0)
    
    mem_before = get_memory_usage()
    start_time = time.perf_counter()
    
    # 1. Setup QuantumVariable
    qv = QuantumVariable(N)
    
    # 2. Define Dicke state initialization function
    def init_dicke(q_var):
        from qrisp import x
        for i in range(K):
            x(q_var[i])
        dicke_state(q_var, K)
        
    # 3. Instantiate XY-QAOA Problem
    xy_qaoa_prob = QAOAProblem(
        cost_operator=create_QUBO_cost_operator(Q),
        mixer=XY_mixer,
        cl_cost_function=create_QUBO_cl_cost_function(Q),
        init_function=init_dicke
    )
    
    # 4. Run optimization
    results_dict = xy_qaoa_prob.run(
        qarg=qv,
        depth=p,
        max_iter=maxiter,
        mes_kwargs={"shots": shots}
    )
    
    end_time = time.perf_counter()
    mem_after = get_memory_usage()
    
    # 5. Extract solution
    best_bitstring = max(results_dict, key=results_dict.get)
    x_sol = np.array([int(bit) for bit in best_bitstring])
    
    runtime = end_time - start_time
    mem_used = mem_after
    
    # 6. Compute metrics
    metrics = calculate_portfolio_metrics(x_sol, mu, Sigma, K, lambda_val)
    energy = calculate_qubo_energy(x_sol, Q) + offset
    
    tickers_list = instance.get('tickers', [])
    selected_tickers = ",".join([tickers_list[i] for i in range(N) if x_sol[i] == 1]) if tickers_list else ""
    num_selected = int(np.sum(x_sol))
    
    results = {
        "dataset": instance['dataset'],
        "solver": "xy_qaoa",
        "N": N,
        "K": K,
        "instance_id": instance['instance_id'],
        "seed": instance['seed'],
        "p": p,
        "objective": metrics['objective'],
        "energy": energy,
        "gap": 0.0,
        "sharpe": metrics['sharpe'],
        "expected_return": metrics['expected_return'],
        "volatility": metrics['volatility'],
        "feasible": metrics['feasible'],
        "runtime_seconds": runtime,
        "memory_mb": mem_used,
        "selected_tickers": selected_tickers,
        "num_selected": num_selected,
        "solution": x_sol,
        "counts": results_dict
    }
    
    return results

def create_QUBO_cl_cost_function_jax(Q):
    """
    Creates a classical cost function for QUBO compatible with both standard Python dicts
    and JAX tracers (under jaspify).
    """
    import jax.numpy as jnp
    def cl_cost_function(counts):
        if isinstance(counts, dict):
            # Standard Python dictionary evaluation
            def QUBO_obj(bitstring, Q):
                x = np.array([int(b) for b in bitstring], dtype=int)
                return float(x.T @ Q @ x)
            energy = 0.0
            for meas, prob in counts.items():
                energy += QUBO_obj(meas, Q) * prob
            return energy
        else:
            # JAX array tracing evaluation (for Jaspify terminal_sampling)
            Q_jax = jnp.array(Q, dtype=jnp.float64)
            N = Q_jax.shape[0]
            powers = 2 ** jnp.arange(N - 1, -1, -1, dtype=jnp.int32)
            
            # Cast counts to int32 to match powers and perform integer division and modulo
            counts_int = counts.astype(jnp.int32)
            X = (counts_int[:, None] // powers) % 2
            X_float = X.astype(jnp.float64)
            
            # Compute x^T Q x for all shots in parallel
            costs = jnp.einsum('si,ij,sj->s', X_float, Q_jax, X_float)
            return jnp.mean(costs)
            
    return cl_cost_function

def solve_jasp(
    instance: Dict[str, Any], 
    p: int = 2, 
    maxiter: int = 100, 
    shots: int = 1024
) -> Dict[str, Any]:
    """
    Solves the QUBO formulation using JaspQAOA (JITcompiled via JAX and terminal sampling).
    
    Parameters:
    instance (dict): Problem instance containing N, K, mu, Sigma, Q, lambda_val.
    p (int): QAOA depth/steps (default: 2).
    maxiter (int): Maximum optimization iterations (default: 100).
    shots (int): Number of measurement shots (default: 1024).
    
    Returns:
    dict: Solver results including portfolio metrics and runtime.
    """
    N = instance['N']
    K = instance['K']
    mu = instance['mu']
    Sigma = instance['Sigma']
    Q = instance['Q']
    lambda_val = instance.get('lambda_val', 0.5)
    offset = instance.get('offset', 0.0)
    
    mem_before = get_memory_usage()
    start_time = time.perf_counter()
    
    # Jasp optimization routine compiled via JAX and terminal sampling
    @jaspify(terminal_sampling=True)
    def execute_jasp_qaoa():
        q_var = QuantumVariable(N)
        qaoa_prob = QAOAProblem(
            cost_operator=create_QUBO_cost_operator(Q),
            mixer=RX_mixer,
            cl_cost_function=create_QUBO_cl_cost_function_jax(Q)
        )
        return qaoa_prob.run(
            qarg=q_var,
            depth=p,
            max_iter=maxiter,
            mes_kwargs={"shots": shots}
        )
        
    # Execute the jaspified function
    results_dict = execute_jasp_qaoa()
    
    end_time = time.perf_counter()
    mem_after = get_memory_usage()
    
    # Extract solution from NumPy array returned by Jasp
    # Find the most frequent state index (mode)
    state_indices, counts = np.unique(results_dict, return_counts=True)
    best_state_index = state_indices[np.argmax(counts)]
    
    # Decode best state index to binary string of length N
    best_bitstring = f"{best_state_index:0{N}b}"
    x_sol = np.array([int(bit) for bit in best_bitstring])
    
    runtime = end_time - start_time
    mem_used = mem_after
    
    # Compute metrics
    metrics = calculate_portfolio_metrics(x_sol, mu, Sigma, K, lambda_val)
    energy = calculate_qubo_energy(x_sol, Q) + offset
    
    tickers_list = instance.get('tickers', [])
    selected_tickers = ",".join([tickers_list[i] for i in range(N) if x_sol[i] == 1]) if tickers_list else ""
    num_selected = int(np.sum(x_sol))
    
    results = {
        "dataset": instance['dataset'],
        "solver": "jasp_qaoa",
        "N": N,
        "K": K,
        "instance_id": instance['instance_id'],
        "seed": instance['seed'],
        "p": p,
        "objective": metrics['objective'],
        "energy": energy,
        "gap": 0.0,
        "sharpe": metrics['sharpe'],
        "expected_return": metrics['expected_return'],
        "volatility": metrics['volatility'],
        "feasible": metrics['feasible'],
        "runtime_seconds": runtime,
        "memory_mb": mem_used,
        "selected_tickers": selected_tickers,
        "num_selected": num_selected,
        "solution": x_sol
    }
    
    return results

def solve_xy_regularized(
    instance: Dict[str, Any], 
    p: int = 2, 
    maxiter: int = 100, 
    shots: int = 1024,
    alpha: float = 0.1
) -> Dict[str, Any]:
    """
    Solves the QUBO formulation using XY-QAOA with TQA initialization and Ridge regularization.
    
    Parameters:
    instance (dict): Problem instance containing N, K, mu, Sigma, Q, lambda_val.
    p (int): QAOA depth/steps (default: 2).
    maxiter (int): Maximum optimization iterations (default: 100).
    shots (int): Number of measurement shots (default: 1024).
    alpha (float): Ridge regularization parameter (default: 0.1).
    
    Returns:
    dict: Solver results including portfolio metrics and runtime.
    """
    N = instance['N']
    K = instance['K']
    mu = instance['mu']
    Sigma = instance['Sigma']
    Q = instance['Q']
    lambda_val = instance.get('lambda_val', 0.5)
    offset = instance.get('offset', 0.0)
    
    mem_before = get_memory_usage()
    start_time = time.perf_counter()
    
    # 1. Setup QuantumVariable
    qv = QuantumVariable(N)
    
    # 2. Define Dicke state initialization function
    def init_dicke(q_var):
        from qrisp import x
        for i in range(K):
            x(q_var[i])
        dicke_state(q_var, K)
        
    # 3. Instantiate Regularized XY-QAOA Problem
    xy_qaoa_prob = RegularizedQAOAProblem(
        cost_operator=create_QUBO_cost_operator(Q),
        mixer=XY_mixer,
        cl_cost_function=create_QUBO_cl_cost_function(Q),
        init_function=init_dicke,
        alpha=alpha
    )
    
    # 4. Run optimization using TQA initialization
    results_dict = xy_qaoa_prob.run(
        qarg=qv,
        depth=p,
        max_iter=maxiter,
        init_type="tqa",
        mes_kwargs={"shots": shots}
    )
    
    end_time = time.perf_counter()
    mem_after = get_memory_usage()
    
    # 5. Extract solution
    best_bitstring = max(results_dict, key=results_dict.get)
    x_sol = np.array([int(bit) for bit in best_bitstring])
    
    runtime = end_time - start_time
    mem_used = mem_after
    
    # 6. Compute metrics
    metrics = calculate_portfolio_metrics(x_sol, mu, Sigma, K, lambda_val)
    energy = calculate_qubo_energy(x_sol, Q) + offset
    
    tickers_list = instance.get('tickers', [])
    selected_tickers = ",".join([tickers_list[i] for i in range(N) if x_sol[i] == 1]) if tickers_list else ""
    num_selected = int(np.sum(x_sol))
    
    results = {
        "dataset": instance['dataset'],
        "solver": "xy_qaoa_regularized",
        "N": N,
        "K": K,
        "instance_id": instance['instance_id'],
        "seed": instance['seed'],
        "p": p,
        "objective": metrics['objective'],
        "energy": energy,
        "gap": 0.0,
        "sharpe": metrics['sharpe'],
        "expected_return": metrics['expected_return'],
        "volatility": metrics['volatility'],
        "feasible": metrics['feasible'],
        "runtime_seconds": runtime,
        "memory_mb": mem_used,
        "selected_tickers": selected_tickers,
        "num_selected": num_selected,
        "solution": x_sol,
        "counts": results_dict
    }
    
    return results

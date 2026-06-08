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
        "solution": x_sol
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
        "solution": x_sol
    }
    
    return results

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
            cl_cost_function=create_QUBO_cl_cost_function(Q)
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
    
    # Extract solution
    best_bitstring = max(results_dict, key=results_dict.get)
    x_sol = np.array([int(bit) for bit in best_bitstring])
    
    runtime = end_time - start_time
    mem_used = mem_after
    
    # Compute metrics
    metrics = calculate_portfolio_metrics(x_sol, mu, Sigma, K, lambda_val)
    energy = calculate_qubo_energy(x_sol, Q) + offset
    
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
        "solution": x_sol
    }
    
    return results

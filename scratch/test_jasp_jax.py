import os
import numpy as np
import jax
import jax.numpy as jnp
from qrisp import QuantumVariable, h
from qrisp.jasp import jaspify
from qrisp.qaoa import QAOAProblem, RX_mixer, create_QUBO_cost_operator

Q = np.array([
    [-1.0, 0.5, 0.2],
    [0.5, -1.0, 0.1],
    [0.2, 0.1, -1.0]
])

def create_QUBO_cl_cost_function_jax(Q):
    def cl_cost_function(counts):
        if isinstance(counts, dict):
            def QUBO_obj(bitstring, Q):
                x = np.array([int(b) for b in bitstring], dtype=int)
                return float(x.T @ Q @ x)
            energy = 0.0
            for meas, prob in counts.items():
                energy += QUBO_obj(meas, Q) * prob
            return energy
        else:
            Q_jax = jnp.array(Q, dtype=jnp.float64)
            N = Q_jax.shape[0]
            powers = 2 ** jnp.arange(N - 1, -1, -1, dtype=jnp.int32)
            counts_int = counts.astype(jnp.int32)
            X = (counts_int[:, None] // powers) % 2
            X_float = X.astype(jnp.float64)
            costs = jnp.einsum('si,ij,sj->s', X_float, Q_jax, X_float)
            return jnp.mean(costs)
    return cl_cost_function

def solve_jasp_test(instance, p=1, maxiter=10, shots=10):
    N = instance['N']
    Q = instance['Q']
    
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
        
    results_array = execute_jasp_qaoa()
    print("execute_jasp_qaoa returned array:", results_array)
    
    # Extract solution using unique modes
    state_indices, counts = np.unique(results_array, return_counts=True)
    best_state_index = state_indices[np.argmax(counts)]
    best_bitstring = f"{best_state_index:0{N}b}"
    x_sol = np.array([int(bit) for bit in best_bitstring])
    
    return x_sol

instance = {
    "N": 3,
    "Q": Q
}

print("Running solve_jasp_test...")
try:
    x_sol = solve_jasp_test(instance, p=1, maxiter=5, shots=10)
    print("Decoded solution vector:", x_sol)
except Exception as e:
    import traceback
    traceback.print_exc()

from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.algorithms.adaptive.qaoa.var_form import QAOAVarForm
from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.finance.ising import portfolio
from qiskit import execute, Aer
import numpy as np

from zquantum.core.interfaces.optimizer import Optimizer
from scipy.optimize import OptimizeResult

class CVAROptimizer(Optimizer):

    def __init__(self, options):
        
    def minimize(self, cost_function, initial_params):
        
        # Optimization Results Object
        history = []
        
        maxiter = 100
        optimizer = COBYLA(maxiter=maxiter)
        var_form = QAOAVarForm(H, 1)  # use this ansatz for CVaR-QAOA
        m = var_form.num_parameters
        backend = Aer.get_backend('statevector_simulator')  

        def compute_cvar(probabilities, values, alpha):
            sorted_indices = np.argsort(values)
            probs = np.array(probabilities)[sorted_indices]
            vals = np.array(values)[sorted_indices]
            cvar = 0
            total_prob = 0
            for i, (p, v) in enumerate(zip(probs, vals)):
                done = False
                if p >= alpha - total_prob:
                    p = alpha - total_prob
                    done = True
                total_prob += p
                cvar += p * v
            cvar /= total_prob
            return cvar
                
        def eval_bitstring(H, x):
            # invert bitstring for convenience and translate to +/-1
            x = x[::-1]    
            spins = np.array([(-1)**(b == '1') for b in x])
            value = 0
            # loop over pauli terms and add contribution to objective
            for p in H.paulis:
                weight = np.real(p[0])
                indices = np.where(p[1].z)
                value += weight * np.prod(spins[indices])
            return value

class Objective:
    
    def __init__(self, var_form, H, offset, alpha, backend, optimal=None):
        self.history = []
        self.var_form = var_form
        self.H = H
        self.offset = offset
        self.alpha = alpha
        self.backend = backend
        self.optimal = optimal
        self.opt_history = []
    
    def evaluate(self, thetas):
    
        # create and run circuit
        qc = self.var_form.construct_circuit(thetas)
        job = execute(qc, self.backend)
        result = job.result()
        counts = result.get_counts()

        # evaluate counts
        probabilities = np.zeros(len(counts))
        values = np.zeros(len(counts))
        for i, (x, p) in enumerate(counts.items()):
            values[i] = eval_bitstring(self.H, x) + self.offset
            probabilities[i] = p
            
        # track optimal probability
        if self.optimal:
            indices = np.where(values <= self.optimal + 1e-8)
            self.opt_history += [sum(probabilities[indices])]
        
        # evaluate cvar
        cvar = compute_cvar(probabilities, values, self.alpha)
        self.history += [cvar]
        return cvar

    optimization_results = {}
    optimization_results['opt_value'] = evaluate
    optimization_results['opt_params'] = opt_history
    optimization_results['history'] = history

    return OptimizeResult(optimization_results)
import warnings
import timeit
import pennylane as qml # type: ignore
from pennylane import numpy as np # type: ignore
warnings.filterwarnings("ignore")

 # type: ignore


symbols = ['O', 'H', 'H', 'H']
coords = np.array([
    [ 0.000000,  0.000000,  0.128500],  # Oxygen (Top of pyramid)
    [ 0.000000,  0.937600, -0.165400],  # H1
    [ 0.812000, -0.468800, -0.165400],  # H2
    [-0.812000, -0.468800, -0.165400]   # H3
], requires_grad=False)

molecule = qml.qchem.Molecule(
    symbols,
    coords,
    charge=1,
    mult=1,
    basis_name='6-31G',
    name='hydronium',
    unit='angstrom'
)

H, qubits = qml.qchem.molecular_hamiltonian(
    molecule,
    mapping='jordan_wigner',
    method='pyscf',
    active_electrons=8,
    active_orbitals=8
)

coeffs, obs = H.terms()

print(f"Using {qubits} qubits to encode the hydronium ion.")
print(f"Molecular Hamiltonian has {len(obs)} operators.")

H_sparse = qml.SparseHamiltonian(H.sparse_matrix(), wires=range(qubits))
dev = qml.device("lightning.qubit", wires=qubits)

hf_state = qml.qchem.hf_state(8, qubits)
print(f"Hartree-Fock state: {hf_state}")

singles, doubles = qml.qchem.excitations(8, qubits)
operator_pool = doubles + singles  # Prioritize doubles usually, but mix is fine
print(f"Operator Pool Size: {len(operator_pool)}")

dev = qml.device("lightning.qubit", wires=range(qubits))

# Storage for the circuit we are building
current_ops = []     
current_params = []  

best_energy = float('inf')
best_params = []
best_ops = []

# --- 2. ENERGY FUNCTION (For the 'Optimization' phase) ---
@qml.qnode(dev, diff_method="adjoint")
def energy_fn(params):
    qml.BasisState(hf_state, wires=range(qubits))
    
    # Apply the grown circuit
    for i, op_wires in enumerate(current_ops):
        if len(op_wires) == 2:
            qml.SingleExcitation(params[i], wires=op_wires)
        elif len(op_wires) == 4:
            qml.DoubleExcitation(params[i], wires=op_wires)
            
    return qml.expval(H_sparse)

start_time = timeit.default_timer()
# --- 3. THE SPEED TRICK: Vectorized Gradient Scanner ---
# Calculates gradients for ALL 350 pool candidates in ONE execution
@qml.qnode(dev, diff_method="adjoint")
def super_gradient_circuit(pool_params, fixed_params):
    qml.BasisState(hf_state, wires=range(qubits))
    
    # A. Apply the EXISTING circuit (Fixed - these don't change during scan)
    for i, wires in enumerate(current_ops):
        if len(wires) == 2:
            qml.SingleExcitation(fixed_params[i], wires=wires)
        elif len(wires) == 4:
            qml.DoubleExcitation(fixed_params[i], wires=wires)
            
    # B. Apply the ENTIRE POOL (Candidates)
    # We apply all 350 gates with parameters 'pool_params'
    # We will pass 0.0 for all of them. The gradient will tell us 
    # "How much does the energy drop if I turn this gate on?"
    for i, wires in enumerate(operator_pool):
        if len(wires) == 2:
            qml.SingleExcitation(pool_params[i], wires=wires)
        elif len(wires) == 4:
            qml.DoubleExcitation(pool_params[i], wires=wires)
            
    return qml.expval(H_sparse)
end_time = timeit.default_timer()
print(f"Time taken to calculate the gradient: {end_time - start_time} seconds")
# --- 4. THE FAST LOOP ---

max_steps = 50
threshold = 5e-5

print(f"--- Starting Fast ADAPT-VQE ---")
print(f"Goal: < -76.30 Ha")

for step in range(max_steps):
    print(f"\nStep {step+1}: Scanning pool...", end="")
    
    # 1. PREPARE INPUTS
    # We want gradients at zero for the pool
    pool_zeros = np.zeros(len(operator_pool), requires_grad=True)
    # The current circuit params are fixed during the scan
    fixed_params = np.array(current_params, requires_grad=False)
    
    # 2. THE ONE-SHOT GRADIENT CALCULATION
    # This runs ONCE instead of 350 times
    grads = qml.grad(super_gradient_circuit, argnum=0)(pool_zeros, fixed_params)
    
    # 3. SELECT THE WINNER
    best_idx = np.argmax(abs(grads))
    max_grad = abs(grads[best_idx])
    
    print(f" Done. Max Grad: {max_grad:.5f}")
    
    if max_grad < threshold:
        print(f"--> Convergence! No more operators needed.")
        break
    
    selected_op = operator_pool[best_idx]
    print(f"  Adding Op: {selected_op}")
    
    # 4. GROW & OPTIMIZE
    current_ops.append(selected_op)
    current_params.append(0.0)
    
    # Retrain the new circuit
    # (Since it's only growing by 1 gate at a time, this is fast)
    opt = qml.AdamOptimizer(stepsize=0.05)
    params_tensor = np.array(current_params, requires_grad=True)
    
    for k in range(30):
        params_tensor, E = opt.step_and_cost(energy_fn, params_tensor)
        
    current_params = params_tensor.tolist()
    print(f"  Current Energy: {E:.6f} Ha")
    
    if E < best_energy:
        best_energy = E
        # need to copy the params and ops to avoid reference issues
        best_params = current_params.copy()
        best_ops = current_ops.copy()
        
    if E < -76.30:
        print("  --> STRONG correlation captured!")
        
print("-" * 30)
print(f"Best Energy: {best_energy:.6f} Ha")
print(f"Circuit Depth: {len(best_ops)}")

core_indices = [0]
active_indices = [1, 2, 3, 4, 5, 6, 7, 8]

print("Building Dipole Operators...")
# This returns a list of 3 Observables: [D_x, D_y, D_z]
dipole_func = qml.qchem.dipole_moment(
    molecule, 
    core=core_indices, 
    active=active_indices,
    mapping='jordan_wigner'
)

dipole_obs = dipole_func()

# --- 2. MEASURE FUNCTION ---
@qml.qnode(dev, diff_method="adjoint")
def measure_dipole(params):
    qml.BasisState(hf_state, wires=range(qubits))
    
    # Reconstruct your optimized ADAPT-VQE circuit
    for i, op_wires in enumerate(best_ops):
        if len(op_wires) == 2:
            qml.SingleExcitation(params[i], wires=op_wires)
        elif len(op_wires) == 4:
            qml.DoubleExcitation(params[i], wires=op_wires)
            
    # Measure all 3 components at once
    return [qml.expval(dipole_obs[0]), 
            qml.expval(dipole_obs[1]), 
            qml.expval(dipole_obs[2])]

# --- 3. EXECUTE ---
# Use the parameters from your BEST step (Step 20 is likely safer than Step 40)
# If you want to use the current state, just pass 'current_params'
dx, dy, dz = measure_dipole(best_params)

total_dipole = np.sqrt(dx**2 + dy**2 + dz**2)

print(f"\n--- Final Results ---")
print(f"Dipole Vector (X, Y, Z): [{dx:.4f}, {dy:.4f}, {dz:.4f}]")
print(f"Total Dipole Moment:     {total_dipole:.4f} Debye")
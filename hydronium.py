import warnings

import pennylane as qml # type: ignore
from pennylane import numpy as np # type: ignore
import torch # type: ignore
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

hf_energy = qml.qchem.hf_energy(molecule)
print(hf_energy())

H_sparse = qml.SparseHamiltonian(H.sparse_matrix(), wires=range(qubits))
dev = qml.device("lightning.qubit", wires=qubits)

hf_state = qml.qchem.hf_state(8, qubits)
print(f"Hartree-Fock state: {hf_state}")

singles, doubles = qml.qchem.excitations(8, qubits)
operator_pool = doubles + singles  # Prioritize doubles usually, but mix is fine
print(f"Operator Pool Size: {len(operator_pool)}")

dev = qml.device("lightning.qubit", wires=range(qubits))

# Storage for the circuit we are building
current_ops = []     # List of selected excitations (wires)
current_params = []  # List of current optimal angles

# --- 2. HELPER: The Circuit Builder ---
def circuit(params, ops_list):
    qml.BasisState(hf_state, wires=range(qubits))
    
    # Apply selected operators in order
    for i, op_wires in enumerate(ops_list):
        # We use SingleExcitation/DoubleExcitation dynamically
        if len(op_wires) == 2:
            qml.SingleExcitation(params[i], wires=op_wires)
        elif len(op_wires) == 4:
            qml.DoubleExcitation(params[i], wires=op_wires)

@qml.qnode(dev, diff_method="adjoint")
def energy_fn(params):
    circuit(params, current_ops)
    return qml.expval(H_sparse)

# --- 3. HELPER: The Gradient Checker (The "Scanner") ---
# Calculates the gradient of adding ONE candidate gate (with angle=0)
@qml.qnode(dev, diff_method="adjoint")
def gradient_scanner(params_existing, candidate_wires):
    qml.BasisState(hf_state, wires=range(qubits))
    
    # 1. Apply existing circuit (fixed)
    for i, op_wires in enumerate(current_ops):
        if len(op_wires) == 2:
            qml.SingleExcitation(params_existing[i], wires=op_wires)
        elif len(op_wires) == 4:
            qml.DoubleExcitation(params_existing[i], wires=op_wires)
            
    # 2. Apply the CANDIDATE gate with parameter 0.0
    # We want the gradient of THIS gate at 0.0
    # (Note: In PennyLane, we must pass it as a tracked variable to get grad)
    theta = np.array(0.0, requires_grad=True)
    if len(candidate_wires) == 2:
        qml.SingleExcitation(theta, wires=candidate_wires)
    elif len(candidate_wires) == 4:
        qml.DoubleExcitation(theta, wires=candidate_wires)
    return qml.expval(H_sparse)

# --- 4. THE MAIN ADAPT-VQE LOOP ---
max_steps = 20
threshold = 1e-4 # Stop if no gradient is larger than this

print(f"--- Starting ADAPT-VQE ---")
print(f"Goal: Grow a circuit that beats -76.266 Ha")

for step in range(max_steps):
    print(f"\nStep {step+1}: Scanning pool...")
    
    # Scan the pool (Find the "Most Wanted" Gate)
    gradients = []
    
    for candidate in operator_pool:
        # Check gradient of adding this candidate
        # We wrap it in qml.grad to get the derivative w.r.t the new angle 'theta'
        g = qml.grad(gradient_scanner)(np.array(current_params), candidate)
        gradients.append(abs(g))
    
    # Select the Best
    best_idx = np.argmax(gradients)
    max_grad = gradients[best_idx]
    
    if max_grad < threshold:
        print(f"--> Convergence! Max gradient ({max_grad:.1e}) < Threshold.")
        break
        
    selected_op = operator_pool[best_idx]
    print(f"  Selected: {selected_op} (Grad: {max_grad:.5f})")
    
    # Grow the Circuit
    current_ops.append(selected_op)
    current_params.append(0.0) # Initialize new param at 0
    
    # Optimize the NEW Circuit (Retrain everything)
    # The new parameter needs to be optimized along with the old ones
    opt = qml.AdamOptimizer(stepsize=0.05)
    params_tensor = np.array(current_params, requires_grad=True)
    
    for k in range(30):
        params_tensor, E = opt.step_and_cost(energy_fn, params_tensor)
    
    current_params = params_tensor.tolist()
    print(f"  New Energy: {E:.6f} Ha")
    if E < -76.28:
        print("  --> STRONG correlation captured!")

print("-" * 30)
print(f"Final Circuit Depth: {len(current_ops)} gates")
print(f"Final Energy: {E:.6f} Ha")

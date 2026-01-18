import warnings
import timeit
import json
import os
import pennylane as qml # type: ignore
from pennylane import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Suppress warnings
warnings.filterwarnings("ignore")

symbols = ['O', 'H', 'H', 'H']
base_coords = np.array([
    [ 0.000000,  0.000000,  0.128500],  # Oxygen (Top of pyramid)
    [ 0.000000,  0.937600, -0.165400],  # H1
    [ 0.812000, -0.468800, -0.165400],  # H2
    [-0.812000, -0.468800, -0.165400]   # H3
], requires_grad=False)

molecule = qml.qchem.Molecule(
    symbols,
    base_coords,
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

H_sparse = qml.SparseHamiltonian(H.sparse_matrix(), wires=range(qubits))
dev = qml.device("lightning.qubit", wires=qubits)

hf_state = qml.qchem.hf_state(8, qubits)
print(f"System: {qubits} qubits")
print(f"Hartree-Fock state: {hf_state}")

singles, doubles = qml.qchem.excitations(8, qubits)
operator_pool = doubles + singles 

current_ops = []     
current_params = []  
best_energy = float('inf')
best_params = []
best_ops = []

CHECKPOINT_FILE = "hydronium_checkpoint.json"

@qml.qnode(dev, diff_method="adjoint")
def energy_fn(params):
    qml.BasisState(hf_state, wires=range(qubits))
    for i, op_wires in enumerate(current_ops):
        if len(op_wires) == 2:
            qml.SingleExcitation(params[i], wires=op_wires)
        elif len(op_wires) == 4:
            qml.DoubleExcitation(params[i], wires=op_wires)
    return qml.expval(H_sparse)

@qml.qnode(dev, diff_method="adjoint")
def super_gradient_circuit(pool_params, fixed_params):
    qml.BasisState(hf_state, wires=range(qubits))
    for i, wires in enumerate(current_ops):
        if len(wires) == 2:
            qml.SingleExcitation(fixed_params[i], wires=wires)
        elif len(wires) == 4:
            qml.DoubleExcitation(fixed_params[i], wires=wires)
    for i, wires in enumerate(operator_pool):
        if len(wires) == 2:
            qml.SingleExcitation(pool_params[i], wires=wires)
        elif len(wires) == 4:
            qml.DoubleExcitation(pool_params[i], wires=wires)
    return qml.expval(H_sparse)


if os.path.exists(CHECKPOINT_FILE):
    print(f"\n[INFO] Found checkpoint file: {CHECKPOINT_FILE}")
    print("Loading saved parameters (Skipping Training)...")
    
    with open(CHECKPOINT_FILE, 'r') as f:
        data = json.load(f)
        best_ops = data["ops"]
        best_params = data["params"]
        best_energy = data["energy"]
        
    print(f"Loaded Circuit Depth: {len(best_ops)}")
    print(f"Loaded Best Energy:   {best_energy:.6f} Ha")
    
    current_ops = best_ops
    current_params = best_params

else:
    max_steps = 40
    threshold = 5e-5

    print(f"\n--- Starting Fast ADAPT-VQE ---")
    start_time = timeit.default_timer()

    for step in range(max_steps):
        print(f"\nStep {step+1}: Scanning pool...", end="")
        
        # 1. SCAN
        pool_zeros = np.zeros(len(operator_pool), requires_grad=True)
        fixed_params = np.array(current_params, requires_grad=False)
        grads = qml.grad(super_gradient_circuit, argnum=0)(pool_zeros, fixed_params)
        
        # 2. SELECT
        best_idx = np.argmax(abs(grads))
        max_grad = abs(grads[best_idx])
        print(f" Done. Max Grad: {max_grad:.5f}")
        
        if max_grad < threshold:
            print(f"--> Convergence! Gradient below threshold.")
            break
        
        selected_op = operator_pool[best_idx]
        selected_op = [int(w) for w in selected_op] 
        print(f"  Adding Op: {selected_op}")

        current_ops.append(selected_op)
        current_params.append(0.0)
        
        opt = qml.AdamOptimizer(stepsize=0.05)
        params_tensor = np.array(current_params, requires_grad=True)
        
        for k in range(30):
            params_tensor, E = opt.step_and_cost(energy_fn, params_tensor)
            
            if E < best_energy:
                best_energy = E
                best_params = params_tensor.copy().tolist()
                best_ops = current_ops[:]
            
        current_params = params_tensor.tolist()
        print(f"  Current Energy: {E:.6f} Ha")
        
        if E < -76.30:
            print("  --> STRONG correlation captured!")

    end_time = timeit.default_timer()
    print("-" * 30)
    print(f"ADAPT-VQE Complete in {end_time - start_time:.1f}s")
    
    print(f"\n[INFO] Saving model to {CHECKPOINT_FILE}...")
    
    save_data = {
        "ops": best_ops,
        "params": best_params,
        "energy": float(best_energy)
    }
    
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(save_data, f, indent=4)
    print("Save successful.")


print("\n--- Physical Properties ---")

best_ops = best_ops[:len(best_params)]

core_indices = [0]
active_indices = [1, 2, 3, 4, 5, 6, 7, 8]

dipole_func = qml.qchem.dipole_moment(
    molecule, 
    core=core_indices, 
    active=active_indices,
    mapping='jordan_wigner'
)

dipole_obs = dipole_func()

@qml.qnode(dev, diff_method="adjoint")
def measure_dipole(params):
    qml.BasisState(hf_state, wires=range(qubits))
    for i, op_wires in enumerate(best_ops):
        if len(op_wires) == 2:
            qml.SingleExcitation(params[i], wires=op_wires)
        elif len(op_wires) == 4:
            qml.DoubleExcitation(params[i], wires=op_wires)     
    return [qml.expval(dipole_obs[0]), 
            qml.expval(dipole_obs[1]), 
            qml.expval(dipole_obs[2])]

dx, dy, dz = measure_dipole(best_params)
total_dipole = np.sqrt(dx**2 + dy**2 + dz**2)

print(f"Dipole Vector (X, Y, Z): [{dx:.4f}, {dy:.4f}, {dz:.4f}]")
print(f"Total Dipole Moment:     {total_dipole:.4f} Debye")


print("\n" + "="*40)
print("Starting PES Scan...")
print("="*40)

scales = [0.80, 0.90, 1.00, 1.10, 1.20] 
energies = []

for scale in scales:
    print(f"Scanning scale {scale:.3f} ... ", end="")
    
    new_coords = base_coords * scale
    
    mol_scan = qml.qchem.Molecule(
        symbols, new_coords, charge=1, mult=1,
        basis_name='6-31G', name=f'hydronium_{scale:.2f}',
        unit='angstrom'
    )
    H_scan, _ = qml.qchem.molecular_hamiltonian(
        mol_scan, mapping='jordan_wigner', method='pyscf',
        active_electrons=8, active_orbitals=8,
    )
    H_scan_sparse = qml.SparseHamiltonian(H_scan.sparse_matrix(), wires=range(qubits))

    @qml.qnode(dev, diff_method="adjoint")
    def scan_cost_fn(params):
        qml.BasisState(hf_state, wires=range(qubits))
        for i, op_wires in enumerate(best_ops):
            if len(op_wires) == 2:
                qml.SingleExcitation(params[i], wires=op_wires)
            elif len(op_wires) == 4:
                qml.DoubleExcitation(params[i], wires=op_wires)
        return qml.expval(H_scan_sparse)

    current_guess_params = np.array(best_params, requires_grad=True)
    
    opt = qml.AdamOptimizer(stepsize=0.05)
    local_min_E = 100.0
    
    for k in range(30):
        current_guess_params, E = opt.step_and_cost(scan_cost_fn, current_guess_params)
        if E < local_min_E: 
            local_min_E = E
            
    energies.append(local_min_E)
    print(f"Energy: {local_min_E:.6f} Ha")

try:
    import matplotlib.pyplot as plt # type: ignore
    plt.figure(figsize=(8, 6))
    plt.plot(scales, energies, 'bo-', markersize=8, linewidth=2, label='ADAPT-VQE')
    
    min_energy = min(energies)
    min_scale = scales[np.argmin(energies)]
    plt.plot(min_scale, min_energy, 'r*', markersize=15, label=f'Min ({min_scale:.2f})')
    
    plt.title(f'PES: Hydronium Dissociation ({len(best_ops)} gates)')
    plt.xlabel('Bond Length Scale')
    plt.ylabel('Energy (Ha)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('hydronium_pes.png', dpi=300)
    print(f"\n[SUCCESS] Plot saved to: hydronium_pes.png")
except ImportError:
    print("\n[WARNING] Matplotlib not installed.")
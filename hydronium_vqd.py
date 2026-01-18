import timeit
import json
import warnings
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
CHECKPOINT_FILE = "hydronium_checkpoint.json"
SCALES = [0.80, 0.90, 1.00, 1.10, 1.20]
PENALTY_BETA = 15.0  # High penalty forces orthogonality
VQD_STEPS = 60       # Steps to optimize excited state

print("--- 1. LOADING CHECKPOINT ---")
try:
    with open(CHECKPOINT_FILE, 'r') as f:
        data = json.load(f)
        # Convert lists back to standard python lists (if needed)
        loaded_ops = data["ops"]
        loaded_params = data["params"]
        print(f"Loaded {len(loaded_ops)} operators from {CHECKPOINT_FILE}")
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find {CHECKPOINT_FILE}. Run hydronium.py first!")

# --- 2. MOLECULE SETUP ---
symbols = ['O', 'H', 'H', 'H']
# Equilibrium coordinates (Angstroms)
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

_, qubits = qml.qchem.molecular_hamiltonian(
    molecule,
    mapping='jordan_wigner',
    method='pyscf',
    active_electrons=8,
    active_orbitals=8
)
dev = qml.device("lightning.qubit", wires=qubits)
hf_state = qml.qchem.hf_state(8, qubits)
print(f"System: {qubits} qubits")
print(f"Hartree-Fock state: {hf_state}")

# --- 3. HELPER FUNCTIONS ---

def get_hamiltonian(coords):
    """Builds the Hamiltonian for a specific geometry."""
    mol = qml.qchem.Molecule(
        symbols, coords, charge=1, mult=1,
        basis_name='6-31G', name='hydronium_vqd',
        unit='angstrom' # CRITICAL: Standardize units
    )
    H, _ = qml.qchem.molecular_hamiltonian(
        mol, mapping='jordan_wigner', method='pyscf',
        active_electrons=8, active_orbitals=8
    )
    return qml.SparseHamiltonian(H.sparse_matrix(), wires=range(qubits))

def ansatz_circuit(params, wires=range(qubits)):
    """Apply the optimized ansatz from the checkpoint."""
    qml.BasisState(hf_state, wires=wires)
    for i, op_wires in enumerate(loaded_ops):
        # Convert explicit list to int if needed by PennyLane
        w = [int(x) for x in op_wires]
        if len(w) == 2:
            qml.SingleExcitation(params[i], wires=w)
        elif len(w) == 4:
            qml.DoubleExcitation(params[i], wires=w)

# --- 4. COST FUNCTIONS ---

# A. Ground State Cost (Standard VQE)
@qml.qnode(dev, diff_method="adjoint")
def cost_ground(params, hamiltonian):
    ansatz_circuit(params)
    return qml.expval(hamiltonian)

# B. State Vector Retriever (For Overlap)
@qml.qnode(dev, interface="numpy") # Standard interface for state vector
def get_state_vector(params):
    ansatz_circuit(params)
    return qml.state()

# C. VQD Cost (Energy + Penalty)
# Note: We compute overlap manually outside the QNode for efficiency/clarity
@qml.qnode(dev, diff_method="adjoint")
def cost_energy_only(params, hamiltonian):
    ansatz_circuit(params)
    return qml.expval(hamiltonian)

def cost_vqd(params, hamiltonian, ground_state_vector):
    # 1. Energy Term
    E = cost_energy_only(params, hamiltonian)
    
    # 2. Overlap Term (Penalty)
    current_state = get_state_vector(params)
    # Calculate |<psi_0 | psi_1>|^2
    overlap = np.abs(np.vdot(ground_state_vector, current_state))**2
    
    return E + PENALTY_BETA * overlap

# --- 5. MAIN SCAN LOOP ---

print("\n" + "="*50)
print(f"Starting VQD Scan (Ground vs. 1st Excited)")
print(f"Scales: {SCALES}")
print("="*50)

gs_energies = []
es_energies = []

for scale in SCALES:
    print(f"\nProcessing Scale {scale:.2f}...")
    
    # 1. Setup Geometry
    current_coords = base_coords * scale
    H_current = get_hamiltonian(current_coords)
    
    # --- STEP A: Refine Ground State (GS) ---
    # We must re-find the GS for *this* geometry to punish the excited state correctly.
    # Warm-start with the JSON parameters.
    gs_params = np.array(loaded_params, requires_grad=True)
    opt_gs = qml.AdamOptimizer(stepsize=0.05)
    
    print("  > Optimizing Ground State (Hub-and-Spoke)...")
    for _ in range(25):
        gs_params, E_gs = opt_gs.step_and_cost(lambda p: cost_ground(p, H_current), gs_params)
    
    # Save GS Vector for the penalty
    gs_vector = get_state_vector(gs_params)
    gs_energies.append(E_gs)
    print(f"    GS Energy: {E_gs:.6f} Ha")
    
    # --- STEP B: Find Excited State (ES) via VQD ---
    # Random start to avoid getting stuck in the GS well
    np.random.seed(42) # Reproducibility
    es_params = np.random.uniform(low=-0.2, high=0.2, size=len(loaded_params))
    es_params = np.array(es_params, requires_grad=True)
    
    opt_vqd = qml.AdamOptimizer(stepsize=0.08) # Slightly aggressive for excited state
    
    print(f"  > Optimizing Excited State (VQD Beta={PENALTY_BETA})...")
    min_es_E = 100.0
    
    for k in range(VQD_STEPS):
        # We optimize the TOTAL cost (Energy + Penalty)
        es_params, total_cost = opt_vqd.step_and_cost(
            lambda p: cost_vqd(p, H_current, gs_vector), 
            es_params
        )
        
        # Strip penalty to check real physical energy
        current_overlap = np.abs(np.vdot(gs_vector, get_state_vector(es_params)))**2
        real_energy = total_cost - PENALTY_BETA * current_overlap
        
        if real_energy < min_es_E:
            min_es_E = real_energy
            
    es_energies.append(min_es_E)
    print(f"    ES Energy: {min_es_E:.6f} Ha (Gap: {(min_es_E - E_gs)*27.211:.2f} eV)")

# --- 6. PLOT ---
print("\nPlotting VQD Results...")

plt.figure(figsize=(9, 6))

# Ground State
plt.plot(SCALES, gs_energies, 'bo-', label='Ground State ($S_0$)', markersize=6)
# Excited State
plt.plot(SCALES, es_energies, 'r^-', label='1st Excited State ($S_1$)', markersize=6)

plt.title('VQD Spectrum: Hydronium Dissociation', fontsize=14)
plt.xlabel('Bond Length Scale', fontsize=12)
plt.ylabel('Energy (Hartree)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)

# Annotation for the Gap at Equilibrium
eq_idx = SCALES.index(1.00)
gap = es_energies[eq_idx] - gs_energies[eq_idx]
plt.annotate(
    f'Gap: {gap:.3f} Ha\n({gap*27.211:.2f} eV)',
    xy=(1.00, (es_energies[eq_idx] + gs_energies[eq_idx])/2),
    xytext=(1.05, -76.0),
    arrowprops=dict(facecolor='black', arrowstyle='->'),
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)
)

filename = 'hydronium_vqd_spectrum.png'
plt.savefig(filename, dpi=300)
print(f"[SUCCESS] Saved VQD spectrum to {filename}")    
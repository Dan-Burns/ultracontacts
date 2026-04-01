import os, sys
import MDAnalysis as mda
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

target_dir = os.path.expanduser('~/Desktop/gamma_secretase_drugging')
pdb_path = os.path.join(target_dir, '8k8e_membrane_minimized_low_pad_protein.pdb')
pq_path = os.path.join(target_dir, 'test.pq')

print(f"Checking {pdb_path}")
u = mda.Universe(pdb_path)
print('Total atoms:', len(u.atoms))
print('Hydrogen atoms:', sum(u.atoms.elements == 'H'))

from ultracontacts.topology import parse_topology
try:
    groups = parse_topology(u, 'protein', 'protein')
    print('Donors:', len(groups.donor_indices))
    print('Acceptors:', len(groups.acceptor_indices))
except Exception as e:
    import traceback
    traceback.print_exc()

print(f"\nChecking {pq_path}")
df = pd.read_parquet(pq_path)
print('User Contacts found:', len(df))
if not df.empty:
    print(df.groupby('itype').size())

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import Geometry


def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer


def transfer_conformers(scaf, mol):
    matches = mol.GetSubstructMatches(scaf)

    if len(matches) < 1:
        raise Exception('Could not find scaffold matches')
    
    match = matches[0]
    # for match in matches:
    mol_coords = mol.GetConformer().GetPositions()
    scaf_coords = mol_coords[np.array(match)]
    scaf_conformer = create_conformer(scaf_coords)

    return scaf_conformer

import pdb; pdb.set_trace()

ligand_fn = "./TestSamples/O14757/reference_active_molecules/Hit2Lead/O14757_Sries14139_reference_ligand_pose_with_h.sdf"
scaffold_smarts_df = pd.read_csv("./TestSamples/O14757/reference_active_molecules/Hit2Lead/top5_common_scaffold_info.csv")
series_id = "Sries14139"

smarts = scaffold_smarts_df.loc[scaffold_smarts_df['SeriseID'] == series_id, 'Scaffold'].values[0]

mol = Chem.MolFromMolFile(ligand_fn)
mol = Chem.RemoveAllHs(mol)

scaf_mol = Chem.MolFromSmarts(smarts) if smarts is not None else None
scaf_conformer = transfer_conformers(scaf_mol, mol)
scaf_mol.AddConformer(scaf_conformer) # This is the scaffold with 3D coordinates for h2l input.
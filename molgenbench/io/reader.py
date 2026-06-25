from pathlib import Path
from typing import List
from rdkit import Chem

from posecheck.utils.chem import remove_radicals

from molgenbench.io.types import MoleculeRecord


def read_sdf_to_records(
    uniprot: str,
    series: str,
    path: Path,
    protein_path: str,
    pocket_path: str,
    ref_active_path: str,
) -> List[MoleculeRecord]:
    """
    Read one SDF file and convert all molecules into MoleculeRecord objects.

    Args:
        path (Path): Path to the .sdf file.
        source_name (str): Optional name (e.g., UniProt ID or model name)
                           stored in metadata for provenance.

    Returns:
        List[MoleculeRecord]: List of MoleculeRecord objects.
    """
    suppl = Chem.SDMolSupplier(str(path), removeHs=False)
    records: List[MoleculeRecord] = []

    for i, mol in enumerate(suppl):
        mol = remove_radicals(mol) if mol is not None else None
        smiles = Chem.MolToSmiles(mol) if mol is not None else None
        num_rotatable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) if mol is not None else None
        record = MoleculeRecord(
            id=mol.GetProp("_Name") if mol is not None else None,
            smiles=smiles,
            uniprot=uniprot,
            series=series,
            rdkit_mol=mol,
            num_rotatable_bonds=num_rotatable_bonds,
            metadata={
                "source_file": str(path),
                "protein_path": protein_path, 
                "pocket_path": pocket_path,
                "ref_active_path": ref_active_path,
            },
        )
        records.append(record)

    return records


def attach_docked_molecules(
    records: List[MoleculeRecord],
    docked_sdf_path,
) -> List[MoleculeRecord]:
    """
    Try to attach docked molecules to existing MoleculeRecords via '_Name' match.
    If docked_sdf_path does not exist, silently skip.

    Args:
        records: List of MoleculeRecord (from read_sdf_file)
        docked_sdf_path: Path to vina-docked SDF file

    Returns:
        Updated list of MoleculeRecords with `metadata["docked_mol"]` if matched.
    """
    docked_sdf_path = Path(docked_sdf_path)
    if not docked_sdf_path.exists():
        return records  # nothing to attach

    docked_suppl = Chem.SDMolSupplier(str(docked_sdf_path))
    docked_dict = {
        mol.GetProp("_Name"): mol
        for mol in docked_suppl
        if mol is not None
    }

    for record in records:
        name = record.id
        if name in docked_dict:
            mol = docked_dict[name]
            record.metadata["docked_mol"] = mol
            # Extract vina affinity as a plain float so it survives pickle
            # roundtrips in parallel workers (RDKit mol properties are not
            # preserved through pickle).
            try:
                record.metadata["vina_dock_affinity"] = float(mol.GetProp("vina_dock"))
            except (KeyError, ValueError):
                pass

    return records

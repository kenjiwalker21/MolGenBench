import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Union
from rdkit import Chem
from rdkit.Chem import Lipinski

from molgenbench.metrics.base import MetricBase, Metric
from molgenbench.metrics.basic import is_valid
from molgenbench.io.types import MoleculeRecord
from molgenbench.io.reader import read_sdf_to_records

import prolif as plf
from posebusters import PoseBusters
from posecheck import PoseCheck
from spyrmsd import molecule
from spyrmsd import rmsd as spy_rmsd


class PoseBusterMetric(Metric):
    """
    Metric wrapper for PoseBusters evaluation.
    Runs PoseBusters on predicted ligand poses against the true reference
    and outputs a CSV summary with the geometric validation results.
    """
    name = "PoseBuster"

    def __init__(self, config: str = "dock"):
        self.config = config
        self.buster = PoseBusters(config=config)

    def compute(
        self,
        record: MoleculeRecord,
    ):
        """
        Run PoseBusters on an SDF/PDB input pair and store results to CSV.

        Args:
            records: MoleculeRecord object
            mol_cond: Path to receptor PDB file (used for context)
        """
        
        mol_pred = record.rdkit_mol
        if not is_valid(mol_pred):
            record.metadata[self.name] = None
            return None
        
        mol_cond = record.metadata.get("protein_path", None)
        
        try:
            df = self.buster.bust(
                mol_pred=mol_pred,
                mol_true=None,
                mol_cond=mol_cond,
            ).droplevel("file")
            
            posebuster_results = df.iloc[0].to_dict()
        except:
            posebuster_results = {}
        record.metadata[self.name] = posebuster_results
        return posebuster_results
    

class StrainEnergyMetrics(Metric):
    """
    Compute strain energy metrics for a given molecule using PoseCheck.
    Outputs total strain energy and per-atom strain energy.
    """
    name = "StrainEnergy"

    def compute(self, record: MoleculeRecord):
        """
        Compute strain energy metrics for the molecule.

        Args:
            record: MoleculeRecord object
        """
        strain = None
        mol = record.rdkit_mol
        
        if not is_valid(mol):
            record.metadata[self.name] = strain
            return record.metadata[self.name]
        
        try:
            pc = PoseCheck()
            pc.load_ligands_from_mols([mol])
            strain = pc.calculate_strain_energy()[0]
        except:
            strain = None

        record.metadata[self.name] = strain
        return strain
    

class ClashScoreMetric(Metric):
    """
    Compute clash score metric for a given molecule using PoseCheck.
    Outputs the clash score value.
    """
    name = "ClashScore"
    
    def _filterMol(self, mol: Chem.Mol) -> bool:
        """
        Filter molecule based on conformer.
        """
        if not is_valid(mol):
            return False
        
        if mol.GetNumAtoms() > 0 and not np.isnan(mol.GetConformer().GetPositions()).any():
            return True
        else:
            return False

    def compute(self, record: MoleculeRecord):
        """
        Compute clash score metric for the molecule.

        Args:
            record: MoleculeRecord object
        """
        clash_score = None
        mol = record.rdkit_mol
        
        if not self._filterMol(mol):
            record.metadata[self.name] = clash_score
            return record.metadata[self.name]
        
        mol = Chem.AddHs(mol, addCoords=True)
        # protein_path = record.metadata.get("protein_path", None)
        pocket_path = record.metadata.get("pocket_path", None)
        
        try:
            pc = PoseCheck()
            pc.load_ligands_from_mols([mol])
            pc.load_protein_from_pdb(pocket_path)
            clash_score = pc.calculate_clashes()[0]
        except:
            clash_score = None

        record.metadata[self.name] = clash_score
        return clash_score


class RMSDMetric(Metric):
    """
    Compute the redocking RMSD between a predicted and a reference molecule.

    This metric first tries to compute the symmetry-corrected RMSD (using spyrmsd),
    and falls back to a simple Cartesian RMSD if symmetry alignment fails.
    The result is stored in record.metadata["RMSD"].
    """

    name = "RMSD"

    def _symmetry_rmsd(self, mol_pred: Chem.Mol, mol_ref: Chem.Mol) -> float:
        """Compute symmetry-corrected RMSD using spyrmsd."""
        try:
            mol_ref = molecule.Molecule.from_rdkit(mol_ref)
            mol_pred = molecule.Molecule.from_rdkit(mol_pred)
            return spy_rmsd.symmrmsd(
                mol_ref.coordinates,
                mol_pred.coordinates,
                mol_ref.atomicnums,
                mol_pred.atomicnums,
                mol_ref.adjacency_matrix,
                mol_pred.adjacency_matrix,
            )
        except Exception as e:
            print(f"[RMSDMetric] Symmetry RMSD failed: {e}")
            return np.nan

    def compute(self, record: MoleculeRecord) -> Dict[str, Any]:
        """
        Compute RMSD between this record's molecule and the reference.

        Args:
            record: MoleculeRecord object with rdkit_mol

        Returns:
            dict: { "RMSD": value }
        """
        mol_ref = record.rdkit_mol
        mol_pred = record.metadata.get("docked_mol", None)
        if mol_pred is None or mol_ref is None:
            record.metadata[self.name] = np.nan
            return np.nan

        # Try symmetry RMSD first, fall back if needed
        rmsd_val = self._symmetry_rmsd(mol_pred, mol_ref)

        record.metadata[self.name] = rmsd_val
        return rmsd_val
    

class InteractionScoreMetric(Metric):
    """
    Compute per-molecule interaction fingerprints, build reference map from active ligands,
    and store both fingerprint and weighted interaction score in metadata.
    """

    name = "InteractionScore"

    def _filterMol(self, mol: Chem.Mol) -> bool:
        """Filter atoms in a molecule based on their atomic symbol and coordinates."""
        
        if Lipinski.NumRotatableBonds(mol) >= 15:
            return False
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "As":
                return False

        return True

    def _format_active_df(self, active_df: pd.DataFrame) -> pd.DataFrame:
        """Format the active interaction dataframe by flattening columns and removing unwanted data."""
        # "residue_interaction" columns to flat strings
        new_columns = (active_df.iloc[0] + "_" + active_df.iloc[1]).tolist()
        active_df.columns = new_columns
        # drop first 3 rows and unwanted columns and drop index column
        active_df = active_df.drop(index=[0,1,2], errors="ignore")
        active_df = active_df.drop(columns=["protein_interaction"], errors="ignore")
        active_df = active_df.replace({"True": True, "False": False})
        return active_df.reset_index(drop=True)

    def _nonBondInteractions(self, ligands: List[Chem.Mol], mol_pro: Chem.Mol) -> pd.DataFrame:
        """Compute interaction fingerprints using ProLIF for a list of ligands."""
        all_interactions = plf.Fingerprint.list_available()
        fp = plf.Fingerprint(interactions=all_interactions)
        fp.run_from_iterable(ligands, mol_pro, progress=False, n_jobs=1)
        df = fp.to_dataframe()
        return df

    def _filter_ligands(self, ligands: List[Chem.Mol]) -> List[Chem.Mol]:
        """Return records that pass geometry and atom filters."""
        return [
            lig for lig in ligands
            if lig is not None and self._filterMol(lig)
        ]

    def _prepare_ligands(self, records: List[MoleculeRecord]) -> List[plf.Molecule]:
        """Convert RDKit mols to ProLIF molecules with explicit hydrogens."""
        ligands = []
        for r in records:
            mol = r.rdkit_mol
            if mol is None:
                continue
            try:
                mol_h = Chem.AddHs(mol, addCoords=True)
                if mol_h is None:
                    continue
                if len(mol_h.GetAtoms())> 0 and not np.isnan(mol_h.GetConformer().GetPositions()).any():
                    ligands.append(plf.Molecule.from_rdkit(mol_h))
            except Exception:
                continue
        return [lig for lig in ligands if lig is not None and len(lig.GetAtoms())> 0 and not np.isnan(lig.GetConformer().GetPositions()).any()]
    
    def _fp_dict_to_flat(self, fp_dict: Dict) -> Dict[str, bool]:
        """Flatten nested fingerprint dict to single-level dict."""
        # example: ('UNL1', 'LEU15.A', 'Hydrophobic'): {0: True} → 'LEU15.A_Hydrophobic': True
        flat = {}
        for (_, residue, interaction), val in fp_dict.items():
            key = f"{residue}_{interaction}"
            flat[key] = list(val.values())[0]  # {0: True} → True
        return flat

    def _compute_fp_df(self, records: List[MoleculeRecord], protein_path: str) -> pd.DataFrame:
        """Compute a flattened interaction fingerprint dataframe for given records."""
        protein = Chem.MolFromPDBFile(protein_path, removeHs=False)
        ligands = self._prepare_ligands(records)
        ligands = self._filter_ligands(ligands)

        try:
            df = self._nonBondInteractions(ligands, plf.Molecule(protein))
            return df
        except Exception:
            return pd.DataFrame()

    def _active_interaction_path(self, ref_active_path: Path) -> Path:
        """Return csv path for cached active interactions."""
        return ref_active_path.parent / f"{ref_active_path.stem}_interactions.csv"
    
    def _active_interaction_score_path(self, ref_active_path: Path) -> Path:
        """Return csv path for cached active interaction scores."""
        return ref_active_path.parent / f"{ref_active_path.stem}_interaction_scores.csv"

    def _load_or_compute_active_df(
        self,
        protein_path: str,
        ref_active_path: str,
    ) -> pd.DataFrame:
        """Load cached active interaction fp or compute and cache."""
        ref_path = Path(ref_active_path)
        if "vina_docked" not in ref_path.stem:
            ref_path = ref_path.parent / f"{ref_path.stem}_vina_docked.sdf"
        csv_path = self._active_interaction_path(ref_path)
        score_csv_path = self._active_interaction_score_path(ref_path)

        if csv_path.exists():
            formatted_df = self._format_active_df(pd.read_csv(csv_path))
            # Compute and save interaction scores if not already done
            if not score_csv_path.exists():
                self._compute_and_save_active_scores(formatted_df, score_csv_path)
            return formatted_df
        
        active_records = read_sdf_to_records(
            uniprot=None,
            series=None,
            path=ref_path,
            protein_path=protein_path,
            pocket_path=None,
            ref_active_path=None,
        )

        active_df = self._compute_fp_df(active_records, protein_path)
        active_df.to_csv(csv_path)
        # must be read again to ensure formatting
        formatted_df = self._format_active_df(pd.read_csv(csv_path))
        # Compute and save interaction scores for each reference molecule
        self._compute_and_save_active_scores(formatted_df, score_csv_path)
        return formatted_df

    def _compute_and_save_active_scores(
        self,
        formatted_df: pd.DataFrame,
        score_csv_path: Path,
    ) -> None:
        """Compute interaction scores for each reference molecule and save to CSV."""
        interaction_ref_map = self.getInteractionMap(formatted_df)
        scores = []
        for idx, row in formatted_df.iterrows():
            row_dict = row.to_dict()
            score = self.getInteractionScores(row_dict, interaction_ref_map)
            scores.append(score)
        # Create a new dataframe with scores added, without modifying original
        df_with_scores = formatted_df.copy()
        df_with_scores["InteractionScore"] = scores
        df_with_scores.to_csv(score_csv_path, index=False)

    def getInteractionMap(self, df: pd.DataFrame, keep_threshold: float = 0.1) -> Dict[str, float]:
        """Compute the weighted interaction frequency map from active ligands."""
        df = df.loc[:, ~df.columns.str.contains("Hydrophobic|VdWContact")]
        counts = df.sum(axis=0).sort_values(ascending=False)
        total_rows = len(df)
        filtered = counts[counts > keep_threshold * total_rows]
        ratios = filtered / filtered.sum()
        return ratios.to_dict()

    def getInteractionScores(self, gen_fp_dcit: Dict[str, bool], interaction_ref_map: Dict[str, float]) -> float:
        """Apply reference interaction weights to generated ligands."""
        interaction_score = 0.0
        for col_name, present in gen_fp_dcit.items():
            if present and col_name in interaction_ref_map:
                interaction_score += interaction_ref_map[col_name]
        return interaction_score

    def _get_ref_map(
        self,
        protein_path: str,
        ref_active_path: str,
    ) -> Dict[str, float]:
        active_df = self._load_or_compute_active_df(protein_path, ref_active_path)
        ref_map = self.getInteractionMap(active_df)
        return ref_map

    def compute(self, record: MoleculeRecord):
        """
        Compute interaction fp and score for generated molecules and store as dict
        under `record.metadata[self.name]`.
        """
        
        mol = record.rdkit_mol
        if not self._filterMol(mol):
            record.metadata[self.name] = None
            return record.metadata[self.name]

        protein_path = record.metadata.get("protein_path", None)
        ref_active_path = record.metadata.get("ref_active_path", None)
        
        interaction_ref_map = self._get_ref_map(protein_path, ref_active_path)

        gen_df = self._compute_fp_df([record], protein_path)
        
        if gen_df.empty:
            record.metadata[self.name] = None
            return record.metadata[self.name]
        
        gen_fp_dict = self._fp_dict_to_flat(gen_df.to_dict())
        interaction_score = self.getInteractionScores(gen_fp_dict, interaction_ref_map)
        gen_fp_dict.update({"InteractionScore": interaction_score})
        record.metadata[self.name] = gen_fp_dict

        return record.metadata[self.name]

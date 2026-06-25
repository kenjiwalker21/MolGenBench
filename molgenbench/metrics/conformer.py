import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from typing import Dict, Any, List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski
from rdkit.DataStructs import TanimotoSimilarity

from molgenbench.metrics.base import Metric
from molgenbench.metrics.basic import is_valid
from molgenbench.io.types import MoleculeRecord
from molgenbench.io.reader import read_sdf_to_records

import prolif as plf
# import MDAnalysis as mda
from posebusters import PoseBusters
from posecheck import PoseCheck
from spyrmsd import molecule
from spyrmsd import rmsd as spy_rmsd

import logging
import filelock
import threading

# Suppress pandas downcasting warning (option added in pandas 2.1)
if pd.__version__ >= "2.1":
    pd.set_option('future.no_silent_downcasting', True)

# Module-level logger for conformer metrics
_logger = logging.getLogger(__name__)


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
        except Exception:
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
        except Exception:
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
        except Exception:
            clash_score = None

        record.metadata[self.name] = clash_score
        return clash_score


class VinaAffinityMetric(Metric):
    """
    Extract the raw Vina docking affinity (kcal/mol) from the pre-docked molecule
    attached to the record.  The value is stored as a float in
    record.metadata["VinaAffinity"]; np.nan if the molecule was not docked or
    the property is missing.

    Vina affinities are negative — more negative means better predicted binding.
    """

    name = "VinaAffinity"

    def compute(self, record: MoleculeRecord):
        # Prefer the pre-extracted float (set by attach_docked_molecules) because
        # RDKit mol properties do not survive pickle roundtrips in parallel workers.
        affinity = record.metadata.get("vina_dock_affinity")
        if affinity is None:
            docked_mol = record.metadata.get("docked_mol")
            if docked_mol is None:
                record.metadata[self.name] = np.nan
                return np.nan
            try:
                affinity = float(docked_mol.GetProp("vina_dock"))
            except (KeyError, ValueError):
                affinity = np.nan
        record.metadata[self.name] = affinity
        return affinity


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
            _logger.debug(f"[RMSD] Symmetry RMSD failed: {e}")
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
            _logger.warning(f"[RMSD] No docked molecule found for {record.id}")
            return np.nan

        # Strip H from both — docking RMSD is over heavy atoms, and the two SDFs
        # may have been read with different removeHs settings
        mol_ref = Chem.RemoveHs(mol_ref)
        mol_pred = Chem.RemoveHs(mol_pred)

        # Try symmetry RMSD first, fall back if needed
        rmsd_val = self._symmetry_rmsd(mol_pred, mol_ref)

        record.metadata[self.name] = rmsd_val
        return rmsd_val
    

class InteractionScoreMetric(Metric):
    """
    Compute per-molecule interaction fingerprints, build reference map from active ligands,
    and store both fingerprint and weighted interaction score in metadata.
    
    This metric uses caching at multiple levels to optimize performance:
    1. Instance-level cache for reference maps (keyed by ref_active_path)
    2. File-level cache for active interaction dataframes (CSV files)
    3. Thread-safe access using file locks for concurrent writes
    """

    name = "InteractionScore"
    
    # Class-level cache shared across instances within the same process
    # This helps when the same metric is used for multiple uniprots
    _ref_map_cache: Dict[str, Dict[str, float]] = {}
    _cache_lock = threading.Lock()

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
        try:
            # Check if dataframe has at least 2 rows before accessing
            if len(active_df) < 2:
                _logger.warning(f"Error formatting active interaction df: dataframe has only {len(active_df)} row(s), need at least 2")
                return pd.DataFrame()
            new_columns = (active_df.iloc[0] + "_" + active_df.iloc[1]).tolist()
        except Exception as e:
            _logger.warning(f"Error formatting active interaction df: {e}")
            return pd.DataFrame()
        active_df.columns = new_columns
        # drop first 3 rows and unwanted columns and drop index column
        active_df = active_df.drop(index=[0,1,2], errors="ignore")
        active_df = active_df.drop(columns=["protein_interaction"], errors="ignore")
        active_df = active_df.replace({"True": True, "False": False}).infer_objects(copy=False)
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
        # u = mda.Universe(protein_path, to_guess=['bonds'])
        ligands = self._prepare_ligands(records)
        ligands = self._filter_ligands(ligands)

        try:
            plf_protein = plf.Molecule(protein)
            # plf_protein = plf.Molecule.from_mda(u, "protein")
            df = self._nonBondInteractions(ligands, plf_protein)
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
        """Load cached active interaction fp or compute and cache.
        
        Uses file locking to prevent race conditions when multiple processes
        try to compute and write the same CSV file simultaneously.
        """
        ref_path = Path(ref_active_path)
        if "vina_docked" not in ref_path.stem and "reference_ligand_pose" not in ref_path.stem:
            ref_path = ref_path.parent / f"{ref_path.stem}_vina_docked.sdf"
        csv_path = self._active_interaction_path(ref_path)
        score_csv_path = self._active_interaction_score_path(ref_path)
        lock_path = csv_path.parent / f"{csv_path.stem}.lock"

        # Use file lock to prevent race conditions in parallel execution
        lock = filelock.FileLock(str(lock_path), timeout=300)  # 5 min timeout
        
        try:
            with lock:
                # Re-check after acquiring lock (another process may have computed it)
                if csv_path.exists():
                    try:
                        formatted_df = self._format_active_df(pd.read_csv(csv_path))
                        if formatted_df.empty:
                            _logger.warning(f"Cached active interaction df is empty for {csv_path}, recomputing...")
                        else:
                            # Compute and save interaction scores if not already done
                            if not score_csv_path.exists():
                                self._compute_and_save_active_scores(formatted_df, score_csv_path)
                            return formatted_df
                    except Exception as e:
                        _logger.warning(f"Error loading active interaction fp: {e}, file: {csv_path}, recomputing...")
                
                _logger.debug(f"Computing active interactions from scratch for {ref_path}")
                active_records = read_sdf_to_records(
                    uniprot=None,
                    series=None,
                    path=ref_path,
                    protein_path=protein_path,
                    pocket_path=None,
                    ref_active_path=None,
                )

                active_df = self._compute_fp_df(active_records, protein_path)
                if active_df.empty:
                    _logger.warning(f"No active interactions computed for {ref_path}")
                    return pd.DataFrame()
                active_df.to_csv(csv_path)
                # must be read again to ensure formatting
                try:
                    formatted_df = self._format_active_df(pd.read_csv(csv_path))
                    if formatted_df.empty:
                        _logger.warning(f"Formatted active interaction df is empty for {csv_path}")
                        return pd.DataFrame()
                except Exception as e:
                    _logger.error(f"Error loading active interaction fp: {e}, file: {csv_path}")
                    return pd.DataFrame()
                # Compute and save interaction scores for each reference molecule
                self._compute_and_save_active_scores(formatted_df, score_csv_path)
                return formatted_df
        except filelock.Timeout:
            _logger.error(f"Timeout waiting for lock on {lock_path}")
            return pd.DataFrame()

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
        """Get the reference interaction map, using cache if available.
        
        This method uses a class-level cache to avoid recomputing the reference
        map for every molecule in the same uniprot/series. The cache key is the
        ref_active_path since all molecules in the same group share the same reference.
        """
        # Check cache first (thread-safe)
        cache_key = ref_active_path
        with self._cache_lock:
            if cache_key in self._ref_map_cache:
                return self._ref_map_cache[cache_key]
        
        try:
            _logger.debug(f"Loading or computing active df for {ref_active_path}")
            active_df = self._load_or_compute_active_df(protein_path, ref_active_path)
            if active_df.empty:
                _logger.debug("Active interaction df is empty, returning empty ref_map")
                # Cache empty result to avoid repeated computation
                with self._cache_lock:
                    self._ref_map_cache[cache_key] = {}
                return {}
            
            ref_map = self.getInteractionMap(active_df)
            
            # Cache the result (thread-safe)
            with self._cache_lock:
                self._ref_map_cache[cache_key] = ref_map
            
            return ref_map
        except Exception as e:
            _logger.error(f"Error getting ref map: {e}")
            return {}

    @classmethod
    def clear_cache(cls):
        """Clear the class-level reference map cache.
        
        This should be called when processing a new batch of uniprots
        to free memory from previously cached reference maps.
        """
        with cls._cache_lock:
            cls._ref_map_cache.clear()

    def _extract_ifp_ligand_atoms(
        self, fp: plf.Fingerprint, frame: int = 0
    ) -> Dict[str, List[int]]:
        """Extract ligand atom indices per contact key from a ProLIF IFP.

        Returns a mapping of "RESIDUE_InteractionType" → sorted list of ligand atom
        indices (into the plf.Molecule that was passed to run_from_iterable).
        Mirrors the pattern used in flowr: interaction["parent_indices"]["ligand"].
        """
        result: Dict[str, List[int]] = {}
        if not hasattr(fp, "ifp") or frame not in fp.ifp:
            return result
        for (_, prot_res), res_interactions in fp.ifp[frame].items():
            for int_type, interactions in res_interactions.items():
                contact_key = f"{prot_res}_{int_type}"
                atom_idxs: set = set()
                for interaction in interactions:
                    atom_idxs.update(interaction["parent_indices"]["ligand"])
                if atom_idxs:
                    result[contact_key] = sorted(atom_idxs)
        return result

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

        try:
            interaction_ref_map = self._get_ref_map(protein_path, ref_active_path)

            if not interaction_ref_map:
                record.metadata[self.name] = None
                return record.metadata[self.name]

            gen_df = self._compute_fp_df([record], protein_path)

            if gen_df.empty:
                record.metadata[self.name] = None
                return record.metadata[self.name]

            gen_fp_dict = self._fp_dict_to_flat(gen_df.to_dict())
            interaction_score = self.getInteractionScores(gen_fp_dict, interaction_ref_map)
            gen_fp_dict.update({"InteractionScore": interaction_score})
            record.metadata[self.name] = gen_fp_dict

            return record.metadata[self.name]
        except Exception as e:
            _logger.error(f"Error computing interaction score: {e}")
            record.metadata[self.name] = None
            return record.metadata[self.name]


class InteractionDiversityMetric(InteractionScoreMetric):
    """
    Compute the average diversity of molecular substructures at native ligand contact sites.

    For each residue+interaction type in the native contact map (as defined by
    InteractionScoreMetric.getInteractionMap), ProLIF is used to identify which ligand
    atoms participate in that interaction (via fp.ifp["parent_indices"]["ligand"]).
    The substructure within `radius` bonds of those atoms is extracted using
    FindAtomEnvironmentOfRadiusN, fingerprinted with Morgan, and diversity
    (1 - mean pairwise Tanimoto) is computed across all generated molecules.
    Per-contact atom mappings are stored in record.metadata["InteractionAtoms"] for reuse.
    The per-contact diversities are averaged to produce a single scalar.
    """

    name = "InteractionDiversity"

    def __init__(self, radius: int = 3, fp_radius: int = 2, fp_bits: int = 2048):
        self.radius = radius
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits

    def _submol_fingerprint(self, mol: Chem.Mol, root_atoms: List[int]):
        """Morgan fingerprint of the union of N-bond environments around root_atoms."""
        if not root_atoms:
            return None
        try:
            bond_env: set = set()
            for root in root_atoms:
                if root < mol.GetNumAtoms():
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, self.radius, root)
                    if env:
                        bond_env |= set(env)
            if not bond_env:
                return None
            amap: Dict[int, int] = {}
            # PathToSubmol requires a list/tuple — passing a set raises a Boost type error
            sub = Chem.PathToSubmol(mol, list(bond_env), atomMap=amap)
            if sub is None or sub.GetNumAtoms() == 0:
                return None
            # PathToSubmol does not initialize ring info; Morgan FP requires it
            Chem.FastFindRings(sub)
            return AllChem.GetMorganFingerprintAsBitVect(sub, self.fp_radius, nBits=self.fp_bits)
        except Exception as e:
            print(f"[InteractionDiversity] _submol_fingerprint failed: {e}")
            _logger.debug(f"[InteractionDiversity] _submol_fingerprint failed: {e}")
            return None

    def _contact_diversity(self, fps: List) -> Optional[float]:
        """1 - mean pairwise Tanimoto over valid fingerprints; None if fewer than 2."""
        valid = [fp for fp in fps if fp is not None]

        if len(valid) < 2:
            return None

        sims = [TanimotoSimilarity(f1, f2) for f1, f2 in combinations(valid, 2)]
        return float(1.0 - np.mean(sims))

    def compute(self, records: List[MoleculeRecord]) -> Dict[str, Any]:
        """
        Compute average interaction-site diversity over native contacts.

        Runs ProLIF on each generated molecule to find which atoms participate in each
        native contact interaction. Stores the mapping in record.metadata["InteractionAtoms"]
        as {"RESIDUE_IntType": [atom_idx, ...]} indexed into the H-added molecule.

        Args:
            records: generated molecule records sharing protein_path and ref_active_path.

        Returns:
            dict with "InteractionDiversity" (mean over contacts) and
            "InteractionDiversity/<residue>_<interaction>" for each contact.
        """
        print(f"[InteractionDiversity] Computing diversity for {len(records)} records")
        first_valid = next((r for r in records if r.rdkit_mol is not None), None)
        if first_valid is None:
            return {self.name: None}

        protein_path = first_valid.metadata.get("protein_path")
        ref_active_path = first_valid.metadata.get("ref_active_path")
        print(ref_active_path)

        if not protein_path or not ref_active_path:
            return {self.name: None}

        ref_map = self._get_ref_map(protein_path, ref_active_path)
        if not ref_map:
            return {self.name: None}

        # Load protein as ProLIF molecule once for all records
        try:
            protein = Chem.MolFromPDBFile(protein_path, removeHs=False)
            if protein is None:
                return {self.name: None}
            plf_protein = plf.Molecule(protein)
        except Exception as e:
            _logger.error(f"[InteractionDiversity] Failed to load protein: {e}")
            print(f"[InteractionDiversity] Failed to load protein: {e}")
            return {self.name: None}

        all_interactions = plf.Fingerprint.list_available()
        # Run ProLIF per record; store interaction atom indices in metadata
        mol_and_atoms: List[Tuple[Chem.Mol, Dict[str, List[int]]]] = []
        for r in records:
            mol = r.rdkit_mol
            if mol is None or not self._filterMol(mol):
                continue
            try:
                mol_h = Chem.AddHs(mol, addCoords=True)
                if mol_h.GetNumConformers() == 0:
                    continue
                if np.isnan(mol_h.GetConformer().GetPositions()).any():
                    continue
                plf_lig = plf.Molecule.from_rdkit(mol_h)
                fp = plf.Fingerprint(interactions=all_interactions)
                fp.run_from_iterable([plf_lig], plf_protein, progress=False, n_jobs=1)
                contact_atoms = self._extract_ifp_ligand_atoms(fp, frame=0)
                r.metadata["InteractionAtoms"] = contact_atoms
                mol_and_atoms.append((mol_h, contact_atoms))
            except Exception as e:
                _logger.debug(f"[InteractionDiversity] ProLIF failed for {r.id}: {e}")
                print(f"[InteractionDiversity] ProLIF failed for {r.id}: {e}")
                continue

        if not mol_and_atoms:
            return {self.name: None}
        
        # For each native contact, collect per-mol substructure fingerprints.
        # Only include contacts where at least 2 molecules contributed a valid fingerprint.
        contact_diversities: Dict[str, float] = {}
        for contact_key in ref_map:
            fps = []
            for mol_h, contact_atoms in mol_and_atoms:
                root_atoms = contact_atoms.get(contact_key, [])
                if len(root_atoms) >= 1:
                    fps.append(self._submol_fingerprint(mol_h, root_atoms))
            diversity = self._contact_diversity(fps)
            print(f"[InteractionDiversity] Diversity for {contact_key}: {diversity}")
            if diversity is not None:
                contact_diversities[contact_key] = diversity

        if not contact_diversities:
            return {self.name: None}

        avg_diversity = float(np.mean(list(contact_diversities.values())))

        print(f"[InteractionDiversity] Average diversity: {avg_diversity}")
        # Return as a nested dict so _save_metrics_to_csv can flatten it into one CSV row,
        # matching the MotifDist convention: {"InteractionDiversity": {"mean": ..., contact: ...}}
        result: Dict[str, Any] = {"mean": avg_diversity}
        result.update(contact_diversities)
        return {self.name: result}

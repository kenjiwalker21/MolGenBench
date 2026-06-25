import numpy as np
from copy import deepcopy
from typing import Optional
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Lipinski, Crippen, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity
from medchem.structural import CommonAlertsFilters, NIBRFilters
from medchem.structural.lilly_demerits import LillyDemeritsFilters

from molgenbench.io.types import MoleculeRecord
from molgenbench.metrics.base import MetricBase, Metric
from molgenbench.utils.sascore import compute_sa_score


def is_valid(mol):
    """
    Check if a molecule is valid, i.e., it has no disconnected fragments.
    """
    if mol is None:
        return False
    return '.' not in Chem.MolToSmiles(mol)


class ValidMetric(Metric):
    name = "Validity"
    def compute(self, record: MoleculeRecord):
        # Check both RDKit parsing and multi-fragment structure
        record.valid = is_valid(record.rdkit_mol)
        record.metadata[self.name] = record.valid
        return record.valid

class QEDMetric(Metric):
    name = "QED"
    def compute(self, record: MoleculeRecord):
        if not is_valid(record.rdkit_mol):
            record.metadata[self.name] = None
            return None
        val = QED.qed(Chem.RemoveHs(record.rdkit_mol))
        record.metadata[self.name] = val
        return val

class SAMetric(Metric):
    name = "SA"
    def compute(self, record: MoleculeRecord):
        if not is_valid(record.rdkit_mol):
            record.metadata[self.name] = None
            return None
        sa_score = compute_sa_score(Chem.RemoveHs(record.rdkit_mol))
        record.metadata[self.name] = sa_score
        return sa_score
    
class ChemFilterMetric(Metric):
    """
    Compute chemical filter metric for a given molecule.
    Outputs True if the molecule passes the filters, False otherwise.
    """
    name = "ChemFilter"
    
    def _getScaffold(self, smiles: str) -> str:
        try:
            mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            return scaffold_smiles
        except:
            return ''
    
    def obey_lipinski(self, mol):
        mol = deepcopy(mol)
        Chem.SanitizeMol(mol)
        rule_1 = Descriptors.ExactMolWt(mol) < 500
        rule_2 = Lipinski.NumHDonors(mol) <= 5
        rule_3 = Lipinski.NumHAcceptors(mol) <= 10
        logp = Crippen.MolLogP(mol)
        rule_4 = (logp >= -2) & (logp <= 5)
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
        return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

    def passes_chem_filters(self, mol):
        common_filter = CommonAlertsFilters()
        nibr_filter = NIBRFilters()
        lilly_filter = LillyDemeritsFilters()

        results_common = common_filter([Chem.MolToSmiles(Chem.RemoveHs(mol))])
        results_nibr = nibr_filter([Chem.MolToSmiles(Chem.RemoveHs(mol))])
        results_lilly = lilly_filter([Chem.MolToSmiles(Chem.RemoveHs(mol))])
        results_RO5 = self.obey_lipinski(Chem.RemoveHs(mol)) == 5
        
        pass_all = all([results_common[0], results_nibr[0], results_lilly[0], results_RO5])
        
        return pass_all

    def compute(self, record: MoleculeRecord):
        """
        Compute chemical filter metric for the molecule.

        Args:
            record: MoleculeRecord object
        """

        passes_filters = False
        mol = record.rdkit_mol
        
        if not is_valid(mol):
            record.metadata[self.name] = passes_filters
            return record.metadata[self.name]
        
        try:
            passes_filters = self.passes_chem_filters(mol)
        except:
            passes_filters = False
            
        scaffold = self._getScaffold(record.smiles)
        scaffold = scaffold if scaffold != '' else None

        record.metadata[self.name] = {
            self.name: passes_filters,
            "scaffold": scaffold
        }
        return record.metadata[self.name]


class ReferenceSimilarityMetric(Metric):
    """
    Tanimoto similarity between a generated molecule and its parent molecule,
    loaded from record.metadata["ref_active_path"] (SDF file).
    Uses Morgan (ECFP4) fingerprints.
    """
    name = "ReferenceSimilarity"

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.radius = radius
        self.n_bits = n_bits

    def _fingerprint(self, mol: Chem.Mol):
        return AllChem.GetMorganFingerprintAsBitVect(
            Chem.RemoveHs(mol), self.radius, nBits=self.n_bits
        )

    def compute(self, record: MoleculeRecord):
        if not is_valid(record.rdkit_mol):
            record.metadata[self.name] = None
            return None

        ref_active_path = record.metadata.get("ref_active_path", None)
        if ref_active_path is None:
            record.metadata[self.name] = None
            return None

        supplier = Chem.SDMolSupplier(str(ref_active_path), removeHs=True)
        ref_mol = next((m for m in supplier if m is not None), None)
        if ref_mol is None or not is_valid(ref_mol):
            record.metadata[self.name] = None
            return None

        similarity = TanimotoSimilarity(
            self._fingerprint(record.rdkit_mol),
            self._fingerprint(ref_mol),
        )
        record.metadata[self.name] = similarity
        return similarity
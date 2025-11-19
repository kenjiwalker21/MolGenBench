import os
import pickle

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from molgenbench.io.types import MoleculeRecord
from molgenbench.metrics.base import Metric
       
class HitRediscoverMetric(Metric):

    name = "HitRediscover"
    
    def __init__(self, fixStereoFrom3D: bool = True):
        self.fixStereoFrom3D = fixStereoFrom3D

    def _enumerate_tautomer_and_partial_chirality(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            enumerator = rdMolStandardize.TautomerEnumerator()
            enumerator.SetMaxTautomers(50)
            canon_mols = enumerator.Enumerate(mol)
            opts = StereoEnumerationOptions(onlyUnassigned=True) 
            canon_isomers = [
                isomer 
                for canon_mol in canon_mols 
                for isomer in EnumerateStereoisomers(canon_mol, options=opts)
            ]
            isomers = canon_isomers + [mol]
            enumerated_inchi = set([Chem.MolToInchi(iso) for iso in isomers if iso is not None])
            return enumerated_inchi
        except:
            return set()
        
    def _getScaffold(self, smiles: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            return scaffold_smiles
        except:
            return ''

    def _fix_stereo_from_3D(self, mol: Chem.Mol) -> Chem.Mol:
        Chem.RemoveStereochemistry(mol)
        Chem.AssignStereochemistryFrom3D(mol)
        return mol
    
    def _load_reference_maps(self, ref_active_path: str): 
        cached_smiles_path = ref_active_path.replace('_reference_active_molecules.sdf', '_smiles.pkl')
        cached_scaffold_path = ref_active_path.replace('_reference_active_molecules.sdf', '_scaffold.pkl')
        
        # load or create cached reference smiles map
        if not os.path.exists(cached_smiles_path):
            ref_smiles_set = set()
            ref_inchi_map_ref_smiles = {}
            suppl = Chem.SDMolSupplier(ref_active_path)
            for ref_mol in suppl:
                if ref_mol is None:
                    continue
                frags = Chem.GetMolFrags(ref_mol, asMols=True)
                ref_mol = max(frags, key=lambda x: x.GetNumAtoms())
                ref_smiles_set.add(Chem.MolToSmiles(ref_mol))
                
            for ref_smiles in ref_smiles_set:
                ref_inchi_set = self._enumerate_tautomer_and_partial_chirality(ref_smiles)
                # multi inchi map to one smiles
                ref_inchi_map_ref_smiles.update({key: ref_smiles for key in ref_inchi_set})
                
            with open(cached_smiles_path, 'wb') as f:
                pickle.dump(ref_inchi_map_ref_smiles, f)
        
        with open(cached_smiles_path, 'rb') as f:
            ref_inchi_map_ref_smiles = pickle.load(f)
            
        # load or create cached reference scaffold map
        if not os.path.exists(cached_scaffold_path):
            ref_smiles_set = set()
            ref_scaffold_set = set()
            ref_inchi_map_ref_scaffold = {}
            suppl = Chem.SDMolSupplier(ref_active_path)
            for ref_mol in suppl:
                if ref_mol is None:
                    continue
                frags = Chem.GetMolFrags(ref_mol, asMols=True)
                ref_mol = max(frags, key=lambda x: x.GetNumAtoms())
                ref_smiles_set.add(Chem.MolToSmiles(ref_mol))
            
            for ref_smiles in ref_smiles_set:
                scaffold_smiles = self._getScaffold(ref_smiles)
                if scaffold_smiles != '':
                    ref_scaffold_set.add(scaffold_smiles)
            
            for ref_scaffold in ref_scaffold_set:
                ref_inchi_set = self._enumerate_tautomer_and_partial_chirality(ref_scaffold)
                # multi inchi map to one smiles
                ref_inchi_map_ref_scaffold.update({key: ref_scaffold for key in ref_inchi_set})
                
            with open(cached_scaffold_path, 'wb') as f:
                pickle.dump(ref_inchi_map_ref_scaffold, f)
        
        with open(cached_scaffold_path, 'rb') as f:
            ref_inchi_map_ref_scaffold = pickle.load(f)
    
        return ref_inchi_map_ref_smiles, ref_inchi_map_ref_scaffold

    def compute(self, record: MoleculeRecord):
        """
        Check whether the molecule matches any reference actives by SMILES or
        Bemis-Murcko scaffold (with tautomer/partial chirality enumeration) and
        store the hit info under `record.metadata[self.name]`.
        """
        result = {
            "gen_smiles": None,
            "gen_scaffold": None,
            "smiles_hit": None,
            "scaffold_hit": None,
            "found_smiles": None,
            "found_smiles_inchi": None,
            "found_scaffold": None,
            "found_scaffold_inchi": None,
        }

        mol = record.rdkit_mol
        if mol is None:
            record.metadata[self.name] = result
            return record.metadata[self.name]
        
        frags = Chem.GetMolFrags(mol, asMols=True)
        gen_mol = max(frags, key=lambda x: x.GetNumAtoms())
        
        if self.fixStereoFrom3D:
            gen_mol = self._fix_stereo_from_3D(gen_mol)
        
        gen_smiles = Chem.MolToSmiles(gen_mol)
        gen_smiles_inchi = Chem.MolToInchi(gen_mol)
        
        gen_scaffold = self._getScaffold(gen_smiles)
        gen_scaffold_inchi = Chem.MolToInchi(Chem.MolFromSmiles(gen_scaffold)) if gen_scaffold != '' else None

        result["gen_smiles"] = gen_smiles
        result["gen_scaffold"] = gen_scaffold if gen_scaffold != '' else None

        ref_active_path = record.metadata.get("ref_active_path", None)
        ref_inchi_map_ref_smiles, ref_inchi_map_ref_scaffold = self._load_reference_maps(ref_active_path)
        
        if gen_smiles_inchi is not None and gen_smiles_inchi in ref_inchi_map_ref_smiles:
            result["smiles_hit"] = True
            result["found_smiles_inchi"] = gen_smiles_inchi
            result["found_smiles"] = ref_inchi_map_ref_smiles[gen_smiles_inchi]
        else:
            result["smiles_hit"] = False
        
        if gen_scaffold_inchi is not None and gen_scaffold_inchi in ref_inchi_map_ref_scaffold:
            result["scaffold_hit"] = True
            result["found_scaffold_inchi"] = gen_scaffold_inchi
            result["found_scaffold"] = ref_inchi_map_ref_scaffold[gen_scaffold_inchi]
        else:
            result["scaffold_hit"] = False
            
        record.metadata[self.name] = result

        return record.metadata[self.name]

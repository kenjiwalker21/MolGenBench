import os
import pickle
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from sup_info.inchi_map_smiles import crossdock_trainset_duplicated_inchi_map_smiles
from molgenbench.io.types import MoleculeRecord
from molgenbench.metrics.base import Metric
       
class HitRediscoverMetric(Metric):

    name = "HitRediscover"
    
    def __init__(self, fixStereoFrom3D: bool = True):
        self.fixStereoFrom3D = fixStereoFrom3D
        # cache for reference maps
        self._cached_ref_active_path = None
        self._cached_series = None
        self._cached_ref_inchi_map_ref_smiles = None
        self._cached_ref_inchi_map_ref_scaffold = None

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
    
    def _load_reference_maps(self, ref_active_path: str, series: str = None, round: str = None):
        """
        Unified method to load reference maps with caching.
        If series is not None, use h2l logic; otherwise use denovo logic.
        """
        # check if cache is valid
        if (self._cached_ref_active_path == ref_active_path 
            and self._cached_series == series
            and self._cached_ref_inchi_map_ref_smiles is not None
            and self._cached_ref_inchi_map_ref_scaffold is not None):
            return self._cached_ref_inchi_map_ref_smiles, self._cached_ref_inchi_map_ref_scaffold
        
        # load reference maps
        if round == "Scaling":
            ref_inchi_map_ref_smiles, ref_inchi_map_ref_scaffold = self._load_reference_maps_scaling(ref_active_path)
        elif series is not None:
            ref_inchi_map_ref_smiles, ref_inchi_map_ref_scaffold = self._load_reference_maps_h2l(ref_active_path, series)
        else:
            ref_inchi_map_ref_smiles, ref_inchi_map_ref_scaffold = self._load_reference_maps_denovo(ref_active_path)
        
        # update cache
        self._cached_ref_active_path = ref_active_path
        self._cached_series = series
        self._cached_ref_inchi_map_ref_smiles = ref_inchi_map_ref_smiles
        self._cached_ref_inchi_map_ref_scaffold = ref_inchi_map_ref_scaffold
        
        return ref_inchi_map_ref_smiles, ref_inchi_map_ref_scaffold

    def _load_reference_maps_h2l(self, ref_active_path: str, series: str):
        cached_inchi_map_smiles_path = ref_active_path.replace('_reference_active_molecules.sdf', f'_{series}_inchi_map_smiles.pkl')
        cached_inchi_map_scaffold_path = ref_active_path.replace('_reference_active_molecules.sdf', f'_{series}_inchi_map_scaffold.pkl')
        
        # load or create cached reference smiles map
        if not os.path.exists(cached_inchi_map_smiles_path):
            ref_smiles_set = set()
            ref_inchi_map_ref_smiles = {}
            suppl = Chem.SDMolSupplier(ref_active_path)
            for ref_mol in suppl:
                if ref_mol is None:
                    continue
                if series in ref_mol.GetProp('_Name'):
                    frags = Chem.GetMolFrags(ref_mol, asMols=True)
                    ref_mol = max(frags, key=lambda x: x.GetNumAtoms())
                    ref_smiles_set.add(Chem.MolToSmiles(ref_mol))
            
            for ref_smiles in ref_smiles_set:
                ref_inchi_set = self._enumerate_tautomer_and_partial_chirality(ref_smiles)
                # multi inchi map to one smiles
                ref_inchi_map_ref_smiles.update({key: ref_smiles for key in ref_inchi_set})
                
            with open(cached_inchi_map_smiles_path, 'wb') as f:
                pickle.dump(ref_inchi_map_ref_smiles, f)
        
        with open(cached_inchi_map_smiles_path, 'rb') as f:
            ref_inchi_map_ref_smiles = pickle.load(f)
            
        # load or create cached reference scaffold map
        if not os.path.exists(cached_inchi_map_scaffold_path):
            ref_smiles_set = set()
            ref_scaffold_set = set()
            ref_inchi_map_ref_scaffold = {}
            suppl = Chem.SDMolSupplier(ref_active_path)
            for ref_mol in suppl:
                if ref_mol is None:
                    continue
                if series in ref_mol.GetProp('_Name'):
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
                
            with open(cached_inchi_map_scaffold_path, 'wb') as f:
                pickle.dump(ref_inchi_map_ref_scaffold, f)
        
        with open(cached_inchi_map_scaffold_path, 'rb') as f:
            ref_inchi_map_ref_scaffold = pickle.load(f)
        
        return ref_inchi_map_ref_smiles, ref_inchi_map_ref_scaffold
    
    def _load_reference_maps_denovo(self, ref_active_path: str): 
        cache_smiles_path = ref_active_path.replace('_reference_active_molecules.sdf', '_smiles.pkl')
        cache_scaffold_path = ref_active_path.replace('_reference_active_molecules.sdf', '_scaffold.pkl')
        cached_inchi_map_smiles_path = ref_active_path.replace('_reference_active_molecules.sdf', '_inchi_map_smiles.pkl')
        cached_inchi_map_scaffold_path = ref_active_path.replace('_reference_active_molecules.sdf', '_inchi_map_scaffold.pkl')

        # load or create cached reference smiles map
        if not os.path.exists(cached_inchi_map_smiles_path):
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
                
            with open(cached_inchi_map_smiles_path, 'wb') as f:
                pickle.dump(ref_inchi_map_ref_smiles, f)
                
            if not os.path.exists(cache_smiles_path):
                with open(cache_smiles_path, 'wb') as f:
                    pickle.dump(ref_smiles_set, f)
                    
        with open(cached_inchi_map_smiles_path, 'rb') as f:
            ref_inchi_map_ref_smiles = pickle.load(f)
            
        # load or create cached reference scaffold map
        if not os.path.exists(cached_inchi_map_scaffold_path):
            ref_smiles_set = set()
            ref_scaffold_set = set()
            ref_inchi_map_ref_scaffold = {}
            ref_scaffold_map_smiles = defaultdict(set)
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
                    ref_scaffold_map_smiles[scaffold_smiles].add(ref_smiles)
            
            for ref_scaffold in ref_scaffold_set:
                ref_inchi_set = self._enumerate_tautomer_and_partial_chirality(ref_scaffold)
                # multi inchi map to one smiles
                ref_inchi_map_ref_scaffold.update({key: ref_scaffold for key in ref_inchi_set})
                
            with open(cached_inchi_map_scaffold_path, 'wb') as f:
                pickle.dump(ref_inchi_map_ref_scaffold, f)
            
            if not os.path.exists(cache_scaffold_path):
                with open(cache_scaffold_path, 'wb') as f:
                    pickle.dump(ref_scaffold_map_smiles, f)
        
        with open(cached_inchi_map_scaffold_path, 'rb') as f:
            ref_inchi_map_ref_scaffold = pickle.load(f)
    
        return ref_inchi_map_ref_smiles, ref_inchi_map_ref_scaffold
    
    def _load_reference_maps_scaling(self, ref_active_path: str): 
        # scaling round use the same active pool to denovo, for both h2l and denovo mode.
        cache_smiles_path = ref_active_path.replace('_reference_active_molecules.sdf', '_smiles.pkl')
        cache_scaffold_path = ref_active_path.replace('_reference_active_molecules.sdf', '_scaffold.pkl')
        cached_inchi_map_smiles_path = ref_active_path.replace('_reference_active_molecules.sdf', '_inchi_map_smiles.pkl')
        cached_inchi_map_scaffold_path = ref_active_path.replace('_reference_active_molecules.sdf', '_inchi_map_scaffold.pkl')

        # load or create cached reference smiles map
        if not os.path.exists(cached_inchi_map_smiles_path):
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
                
            with open(cached_inchi_map_smiles_path, 'wb') as f:
                pickle.dump(ref_inchi_map_ref_smiles, f)
                
            if not os.path.exists(cache_smiles_path):
                with open(cache_smiles_path, 'wb') as f:
                    pickle.dump(ref_smiles_set, f)
                    
        with open(cached_inchi_map_smiles_path, 'rb') as f:
            ref_inchi_map_ref_smiles = pickle.load(f)
            
        # load or create cached reference scaffold map
        if not os.path.exists(cached_inchi_map_scaffold_path):
            ref_smiles_set = set()
            ref_scaffold_set = set()
            ref_inchi_map_ref_scaffold = {}
            ref_scaffold_map_smiles = defaultdict(set)
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
                    ref_scaffold_map_smiles[scaffold_smiles].add(ref_smiles)
            
            for ref_scaffold in ref_scaffold_set:
                ref_inchi_set = self._enumerate_tautomer_and_partial_chirality(ref_scaffold)
                # multi inchi map to one smiles
                ref_inchi_map_ref_scaffold.update({key: ref_scaffold for key in ref_inchi_set})
                
            with open(cached_inchi_map_scaffold_path, 'wb') as f:
                pickle.dump(ref_inchi_map_ref_scaffold, f)
            
            if not os.path.exists(cache_scaffold_path):
                with open(cache_scaffold_path, 'wb') as f:
                    pickle.dump(ref_scaffold_map_smiles, f)
        
        with open(cached_inchi_map_scaffold_path, 'rb') as f:
            ref_inchi_map_ref_scaffold = pickle.load(f)
    
        return ref_inchi_map_ref_smiles, ref_inchi_map_ref_scaffold

    def _check_smiles_inchi_in_trainset(self, uniprot: str, smiles_inchi: str):
        if uniprot not in crossdock_trainset_duplicated_inchi_map_smiles:
            return False
        return smiles_inchi in crossdock_trainset_duplicated_inchi_map_smiles[uniprot]
    
    def _check_scaffold_inchi_in_trainset(self, uniprot: str, scaffold_inchi: str):
        if f"{uniprot}_scaffold" not in crossdock_trainset_duplicated_inchi_map_smiles:
            return False
        return scaffold_inchi in crossdock_trainset_duplicated_inchi_map_smiles[f"{uniprot}_scaffold"]
    
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
            "found_smiles_inchi_in_trainset": None,
            "found_scaffold": None,
            "found_scaffold_inchi": None,
            "found_scaffold_inchi_in_trainset": None,
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
        gen_scaffold_mol = Chem.MolFromSmiles(gen_scaffold) if gen_scaffold != '' else None
        gen_scaffold_inchi = Chem.MolToInchi(gen_scaffold_mol) if gen_scaffold_mol != None else None

        result["gen_smiles"] = gen_smiles
        result["gen_scaffold"] = gen_scaffold if gen_scaffold != '' else None

        ref_active_path = record.metadata.get("ref_active_path", None)
        # check if denovo mode or h2l mode
        series = record.series
        round = record.round
        ref_inchi_map_ref_smiles, ref_inchi_map_ref_scaffold = self._load_reference_maps(ref_active_path, series, round)
        
        if gen_smiles_inchi is not None and gen_smiles_inchi in ref_inchi_map_ref_smiles:
            result["smiles_hit"] = True
            result["found_smiles_inchi"] = gen_smiles_inchi
            result["found_smiles"] = ref_inchi_map_ref_smiles[gen_smiles_inchi]
            result["found_smiles_inchi_in_trainset"] = self._check_smiles_inchi_in_trainset(record.uniprot, gen_smiles_inchi)
        else:
            result["smiles_hit"] = False
        
        if gen_scaffold_inchi is not None and gen_scaffold_inchi in ref_inchi_map_ref_scaffold:
            result["scaffold_hit"] = True
            result["found_scaffold_inchi"] = gen_scaffold_inchi
            result["found_scaffold"] = ref_inchi_map_ref_scaffold[gen_scaffold_inchi]
            result["found_scaffold_inchi_in_trainset"] = self._check_scaffold_inchi_in_trainset(record.uniprot, gen_scaffold_inchi)
        else:
            result["scaffold_hit"] = False
            
        record.metadata[self.name] = result

        return record.metadata[self.name]

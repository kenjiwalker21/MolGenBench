import os
import logging
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from joblib import Parallel, delayed
from molgenbench.io.types import MoleculeRecord
from molgenbench.io.reader import read_sdf_to_records, attach_docked_molecules
from molgenbench.metrics.basic import ValidMetric, QEDMetric, SAMetric, ChemFilterMetric, ReferenceSimilarityMetric
from molgenbench.metrics.conformer import PoseBusterMetric, StrainEnergyMetrics, RMSDMetric, InteractionScoreMetric, InteractionDiversityMetric, ClashScoreMetric, VinaAffinityMetric
from molgenbench.metrics.distribution import DiversityMetric, UniquenessMetric, MotifDistMetric
from molgenbench.metrics.hitrate import HitRediscoverMetric

_logger = logging.getLogger(__name__)


# https://gist.github.com/tsvikas/5f859a484e53d4ef93400751d0a116de
class ParallelTqdm(Parallel):
    """joblib.Parallel, but with a tqdm progressbar

    Additional parameters:
    ----------------------
    total_tasks: int, default: None
        the number of expected jobs. Used in the tqdm progressbar.
        If None, try to infer from the length of the called iterator, and
        fallback to use the number of remaining items as soon as we finish
        dispatching.
        Note: use a list instead of an iterator if you want the total_tasks
        to be inferred from its length.

    desc: str, default: None
        the description used in the tqdm progressbar.

    disable_progressbar: bool, default: False
        If True, a tqdm progressbar is not used.

    show_joblib_header: bool, default: False
        If True, show joblib header before the progressbar.

    Removed parameters:
    -------------------
    verbose: will be ignored


    Usage:
    ------
    >>> from joblib import delayed
    >>> from time import sleep
    >>> ParallelTqdm(n_jobs=-1)([delayed(sleep)(.1) for _ in range(10)])
    80%|████████  | 8/10 [00:02<00:00,  3.12tasks/s]

    """

    def __init__(
        self,
        *,
        total_tasks: Optional[int] = None,
        desc: Optional[str] = None,
        disable_progressbar: bool = False,
        show_joblib_header: bool = False,
        **kwargs,
    ):
        if "verbose" in kwargs:
            raise ValueError(
                "verbose is not supported. "
                "Use show_progressbar and show_joblib_header instead."
            )
        super().__init__(verbose=(1 if show_joblib_header else 0), **kwargs)
        self.total_tasks = total_tasks
        self.desc = desc
        self.disable_progressbar = disable_progressbar
        self.progress_bar: tqdm | None = None

    def __call__(self, iterable):
        try:
            if self.total_tasks is None:
                # try to infer total_tasks from the length of the called iterator
                try:
                    self.total_tasks = len(iterable)
                except (TypeError, AttributeError):
                    pass
            # call parent function
            return super().__call__(iterable)
        finally:
            # close tqdm progress bar
            if self.progress_bar is not None:
                self.progress_bar.close()

    __call__.__doc__ = Parallel.__call__.__doc__

    def dispatch_one_batch(self, iterator):
        # start progress_bar, if not started yet.
        if self.progress_bar is None:
            self.progress_bar = tqdm(
                desc=self.desc,
                total=self.total_tasks,
                disable=self.disable_progressbar,
                unit="tasks",
            )
        # call parent function
        return super().dispatch_one_batch(iterator)

    dispatch_one_batch.__doc__ = Parallel.dispatch_one_batch.__doc__

    def print_progress(self):
        """Display the process of the parallel execution using tqdm"""
        # if we finish dispatching, find total_tasks from the number of remaining items
        if self.progress_bar is not None:
            if self.total_tasks is None and self._original_iterator is None:
                self.total_tasks = self.n_dispatched_tasks
                self.progress_bar.total = self.total_tasks
                self.progress_bar.refresh()
            # update progressbar
            self.progress_bar.update(self.n_completed_tasks - self.progress_bar.n)


class Evaluator:
    """
    Unified evaluator that handles molecule-level and dataset-level evaluation,
    grouped by Uniprot or Series depending on task type.
    """

    def __init__(self, metric_names: List[str] = None, fixStereoFrom3D: bool = True):
        self.metric_names = metric_names or ["Validity", "QED", "SA", "Uniqueness", "Diversity", "ReferenceSimilarity", "PoseBuster", "StrainEnergy", "RMSD", "HitRediscover"]
        self.metric_map = {
            "Validity": ValidMetric(),
            "QED": QEDMetric(),
            "SA": SAMetric(),
            "Uniqueness": UniquenessMetric(),
            "Diversity": DiversityMetric(),
            "ChemFilter": ChemFilterMetric(),
            "ReferenceSimilarity": ReferenceSimilarityMetric(),

            "PoseBuster": PoseBusterMetric(),
            "StrainEnergy": StrainEnergyMetrics(),
            "RMSD": RMSDMetric(),
            "InteractionScore": InteractionScoreMetric(),
            "InteractionDiversity": InteractionDiversityMetric(),
            "ClashScore": ClashScoreMetric(),
            "VinaAffinity": VinaAffinityMetric(),
            
            "MotifDist": MotifDistMetric(),
            
            "HitRediscover": HitRediscoverMetric(fixStereoFrom3D=fixStereoFrom3D),
        }

        # 按类型分类 metric（单分子级 vs 数据集级）
        self.molecule_metrics = [
            self.metric_map[n] for n in self.metric_names if n in [
                "Validity",
                "QED",
                "SA",
                
                "PoseBuster",
                "StrainEnergy",
                "RMSD",
                
                "ClashScore",
                "InteractionScore",
                "VinaAffinity",
                "ChemFilter",
                "ReferenceSimilarity",
                "HitRediscover"
                
            ]
        ]
        self.dataset_metrics = [
            self.metric_map[n] for n in self.metric_names if n in ["Diversity", "Uniqueness", "MotifDist", "InteractionDiversity"]
        ]

    
    def _save_metrics_to_csv(
        self,
        molecule_records: List[MoleculeRecord],
        dataset_metrics: List[Dict[str, Any]],
        uniprot: str,
        series: str,
        save_dir: str,
        model_name: str,
        mode: str,
    ):
        os.makedirs(save_dir, exist_ok=True)
        molecule_metric_data = {}
        dataset_metric_data = {}

        # 1️⃣ 分子级指标
        for r in molecule_records:
            smiles = getattr(r, "smiles", None)
            mol_id = getattr(r, "id", None)
            num_rotatable_bonds = getattr(r, "num_rotatable_bonds", None)

            for metric_name, value in r.metadata.items():
                if value is None:
                    continue
                if metric_name not in molecule_metric_data:
                    molecule_metric_data[metric_name] = []

                base_info = {
                    "id": mol_id, 
                    "smiles": smiles, 
                    "num_rotatable_bonds": num_rotatable_bonds, 
                    "uniprot": uniprot, 
                    "series": series,
                }

                # 如果是 dict，展开
                if isinstance(value, dict):
                    base_info.update(value)
                else:
                    base_info[metric_name] = value

                molecule_metric_data[metric_name].append(base_info)

        # 2️⃣ 数据集级指标
        for k, v in dataset_metrics.items():
            if k not in dataset_metric_data:
                dataset_metric_data[k] = []
            if k in ("MotifDist", "InteractionDiversity") and isinstance(v, dict):
                row = {"uniprot": uniprot, "series": series}
                row.update(v)
                dataset_metric_data[k].append(row)
            else:
                dataset_metric_data[k].append({"uniprot": uniprot, "series": series, k: v})

        # 3️⃣ 保存 molecule-level
        for metric_name, rows in molecule_metric_data.items():
            if metric_name not in self.metric_names:
                continue
            df = pd.DataFrame(rows)
            if metric_name == "InteractionScore":
                df = df.fillna(False)
            out_path = os.path.join(save_dir, f"{metric_name}.csv")
            df.to_csv(out_path, index=False)

        # 4️⃣ 保存 dataset-level
        for metric_name, rows in dataset_metric_data.items():
            if metric_name not in self.metric_names:
                continue
            df = pd.DataFrame(rows)
            out_path = os.path.join(save_dir, f"{metric_name}.csv")
            df.to_csv(out_path, index=False)
            
    def _check_metrics_exist(self, results_dir: str) -> bool:
        """检查所有需要的 metric CSV 文件是否已存在"""
        if not os.path.exists(results_dir):
            return False
        for metric_name in self.metric_names:
            csv_path = os.path.join(results_dir, f"{metric_name}.csv")
            if not os.path.exists(csv_path):
                return False
        return True

    # ---------------------------- #
    #   Molecule-level evaluation
    # ---------------------------- #
    def evaluate_molecule_metrics(self, records: List[MoleculeRecord]) -> List[MoleculeRecord]:
        """Compute metrics like Valid, QED, SA for each molecule."""
        for record in records:
            for metric in self.molecule_metrics:
                metric.compute(record)
        return records

    # ---------------------------- #
    #   Dataset-level evaluation
    # ---------------------------- #
    def evaluate_dataset_metrics(self, records: List[MoleculeRecord]) -> Dict[str, Any]:
        """Compute dataset-level metrics like Diversity, Uniqueness."""
        results = {}
        for metric in self.dataset_metrics:
            results.update(metric.compute(records))
        return results

    def _process_uniprot(
        self,
        uniprot: str,
        root_dir: str,
        model_name: str,
        round: str,
        mode: str,
        skip_existing: bool,
    ):
        """Process a single uniprot entry.
        
        This method is designed to be called in parallel by joblib workers.
        Each uniprot is processed independently, and any metric caches
        are cleared after processing to free memory.
        """
        uniprot_path = os.path.join(root_dir, uniprot, round, mode)
        prot_path = os.path.join(root_dir, uniprot, f"{uniprot}_prep.pdb")
        pocket_path = os.path.join(root_dir, uniprot, f"{uniprot}_pocket10.pdb")
        ref_active_path = os.path.join(root_dir, uniprot, "reference_active_molecules", f"{uniprot}_reference_active_molecules.sdf")
        
        try:
            if mode == "De_novo_Results":
                self._process_denovo(
                    uniprot, uniprot_path, prot_path, pocket_path, 
                    ref_active_path, model_name, skip_existing, mode
                )

            elif mode == "Hit_to_Lead_Results":
                self._process_hit2lead(
                    uniprot, uniprot_path, prot_path, pocket_path,
                    ref_active_path, model_name, skip_existing, mode
                )
        finally:
            # Clear InteractionScoreMetric cache after processing each uniprot
            # to prevent memory buildup in long-running parallel jobs
            InteractionScoreMetric.clear_cache()
    
    def _process_denovo(
        self,
        uniprot: str,
        uniprot_path: str,
        prot_path: str,
        pocket_path: str,
        ref_active_path: str,
        model_name: str,
        skip_existing: bool,
        mode: str,
    ):
        """Process a single uniprot entry for de novo mode."""
        sdf_path = os.path.join(uniprot_path, model_name, f"{uniprot}_{model_name}.sdf")
        if not os.path.exists(sdf_path):
            return
        
        results_dir = os.path.join(uniprot_path, model_name, "results")
        if skip_existing and self._check_metrics_exist(results_dir):
            return
        
        docked_path = sdf_path.replace(".sdf", "_vina_docked.sdf")
        records = read_sdf_to_records(
            uniprot, None, sdf_path, prot_path, pocket_path, ref_active_path
        )
        records = attach_docked_molecules(records, docked_path)
        
        records = self.evaluate_molecule_metrics(records)
        dataset_results = self.evaluate_dataset_metrics(records)
        self._save_metrics_to_csv(
            records, dataset_results, save_dir=results_dir,
            uniprot=uniprot, series=None, model_name=model_name, mode=mode,
        )
    
    def _process_hit2lead(
        self,
        uniprot: str,
        uniprot_path: str,
        prot_path: str,
        pocket_path: str,
        ref_active_path: str,
        model_name: str,
        skip_existing: bool,
        mode: str,
    ):
        """Process a single uniprot entry for hit-to-lead mode."""
        if not os.path.exists(uniprot_path):
            return
        for series_id in os.listdir(uniprot_path):
            series_path = os.path.join(uniprot_path, series_id)
            sdf_path = os.path.join(series_path, model_name, f"{uniprot}_{series_id}_{model_name}.sdf")
            if not os.path.exists(sdf_path):
                continue
            
            results_dir = os.path.join(series_path, model_name, "results")
            if skip_existing and self._check_metrics_exist(results_dir):
                continue

            # try to read sdf files
            try:
                _  = Chem.SDMolSupplier(sdf_path)
            except Exception:
                _logger.warning(f"Error reading SDF file: {sdf_path}")
                continue
            
            docked_path = sdf_path.replace(".sdf", "_vina_docked.sdf")
            # series_ref_active_path = os.path.join(
            #     os.path.dirname(ref_active_path), "Hit2Lead",
            #     f"{uniprot}_{series_id}_reference_ligand_pose_with_h.sdf"
            # )
            # if not os.path.exists(series_ref_active_path):
            
            series_ref_active_path = ref_active_path
                
            records = read_sdf_to_records(
                uniprot, series_id, sdf_path, prot_path, pocket_path, series_ref_active_path
            )
            records = attach_docked_molecules(records, docked_path)
            
            records = self.evaluate_molecule_metrics(records)
            dataset_results = self.evaluate_dataset_metrics(records)
            self._save_metrics_to_csv(
                records, dataset_results, save_dir=results_dir,
                uniprot=uniprot, series=series_id, model_name=model_name, mode=mode,
            )

    # ---------------------------- #
    #   顶层 pipeline
    # ---------------------------- #
    def run(
        self, 
        root_dir: str, 
        model_name: str, 
        round: str,
        mode: str = "De_novo_Results",
        skip_existing: bool = False,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """
        Run the full evaluation pipeline.

        Args:
            root_dir: directory containing Uniprot folders
            mode: 'denovo' or 'hit2lead'
            skip_existing: if True, skip evaluation when all metric CSVs already exist
        Returns:
            pd.DataFrame of aggregated results
        """
        
        uniprot_list = [u for u in os.listdir(root_dir) if u.startswith(('O', 'P', 'Q'))]
        
        if n_jobs != 1:
            ParallelTqdm(n_jobs=n_jobs, total_tasks=len(uniprot_list))(
                delayed(self._process_uniprot)(
                    uniprot, root_dir, model_name, round, mode, skip_existing
                )
                for uniprot in uniprot_list
            )
        else:
            for uniprot in tqdm(uniprot_list):
                self._process_uniprot(
                    uniprot, root_dir, model_name, round, mode, skip_existing
                )

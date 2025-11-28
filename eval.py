import os
import argparse

from molgenbench.pipeline.evaluator import Evaluator
from molgenbench.pipeline.aggregator import Aggregator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='Path/to/data/', help='Path to the data directory contain uniprot folders')
    parser.add_argument('--round_name', type=str, default='Round1', help='Name of the current round')
    parser.add_argument('--mode', choices=['De_novo_Results', 'Hit_to_Lead_Results'], default='De_novo_Results', help='Task mode')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the generative model')
    args = parser.parse_args()
    
    evaluator = Evaluator(
        [
            "Validity", 
            "QED",
            "SA",
            "Uniqueness",
            "Diversity",
            "MotifDist",
            "ChemFilter",
            "HitRediscover",

            # ----below are 3D metrics----
            "PoseBuster",
            "StrainEnergy",
            "RMSD",
            "InteractionScore"
            "ClashScore"
            
         ]
    )
    evaluator.run(
        root_dir=args.data_path,
        model_name=args.model_name,
        round=args.round_name,
        mode=args.mode
    )
    
    output_path = f"./logs/{args.round_name}/{args.mode}/{args.model_name}/results.yaml"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    aggregator = Aggregator()
    aggregator.run(
        root_dir=args.data_path,
        model_name=args.model_name,
        round=args.round_name,
        mode=args.mode,
        output_path=output_path,
    )


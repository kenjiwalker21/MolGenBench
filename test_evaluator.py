from molgenbench.pipeline.evaluator import Evaluator
from molgenbench.pipeline.aggregator import Aggregator


if __name__ == "__main__":
    evaluator = Evaluator(
        [
            "Validity", 
            "QED",
            "SA",
            "Uniqueness",
            "Diversity",
            "PoseBuster",
            "StrainEnergy",
            "RMSD",
            
            "ChemFilter",
            "InteractionScore"
            "ClashScore"
            "MotifDist",
            
            "HitRediscover",
         ]
    )
    evaluator.run(
        root_dir="./TestSamples",
        model_name="PocketFlow_generated_molecules",
        round="Round1",
        mode="De_novo_Results"
    )
    # evaluator.run(
    #     root_dir="./TestSamples",
    #     model_name="DeleteHit2Lead(CrossDock)_Hit_to_Lead",
    #     round="Round1",
    #     mode="Hit_to_Lead_Results"
    # )

    aggregator = Aggregator()
    aggregator.run(
        root_dir="./TestSamples",
        round="Round1",
        mode="De_novo_Results",
        model_name="PocketFlow_generated_molecules",
        output_path="./pocketflow_report.yaml"
    )


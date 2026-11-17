import pandas as pd

from riskset_tune.experiments.extended import write_aggregates


def test_lambda_selection_uses_validation_and_pairs_only_completed_stages(tmp_path):
    stage = tmp_path / "stage_200" / "GBSG2_seed_42_K_32"
    stage.mkdir(parents=True)
    rows = [
        {"dataset": "GBSG2", "K": 32, "seed": 42, "method": "random", "lambda_risk": 0,
         "steps_requested": 200, "completed_steps": 0, "C-index IPCW": .55, "IBS": .20,
         "best_validation_loss": 1.0},
        {"dataset": "GBSG2", "K": 32, "seed": 42, "method": "tunetables", "lambda_risk": 0,
         "steps_requested": 200, "completed_steps": 200, "C-index IPCW": .56, "IBS": .19,
         "best_validation_loss": .9},
        {"dataset": "GBSG2", "K": 32, "seed": 42, "method": "riskset-tune", "lambda_risk": .01,
         "steps_requested": 200, "completed_steps": 200, "C-index IPCW": .57, "IBS": .188,
         "best_validation_loss": .8},
        {"dataset": "GBSG2", "K": 32, "seed": 42, "method": "riskset-tune", "lambda_risk": .1,
         "steps_requested": 200, "completed_steps": 200, "C-index IPCW": .58, "IBS": .185,
         "best_validation_loss": .7},
        # Best apparent test score is deliberately incomplete and cannot be selected.
        {"dataset": "GBSG2", "K": 32, "seed": 42, "method": "riskset-tune", "lambda_risk": 1.0,
         "steps_requested": 200, "completed_steps": 100, "C-index IPCW": .99, "IBS": .01,
         "best_validation_loss": .1},
    ]
    pd.DataFrame(rows).to_csv(stage / "results.csv", index=False)

    write_aggregates(tmp_path)

    selected = pd.read_csv(tmp_path / "validation_selected_lambda.csv")
    assert selected.lambda_risk.tolist() == [.1]
    paired = pd.read_csv(tmp_path / "paired_comparisons.csv")
    assert set(paired.lambda_risk) == {.01, .1}
    selected_paired = pd.read_csv(tmp_path / "paired_selected_lambda.csv")
    assert set(selected_paired.baseline) == {"random", "tunetables"}
    assert set(selected_paired.lambda_risk) == {.1}

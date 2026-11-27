"""Main pipeline runner for the Hotel Reservations Cancellation project.

Orchestrates the five CRISP-DM stages sequentially:
1. Data Understanding   (step1_data_understanding.py)
2. Data Preparation     (step2_data_preparation.py)
3. Data Modeling        (step3_data_modeling.py)
4. Data Testing         (step4_data_testing.py)
5. Evaluation           (step5_evaluation.py)
"""

from step1_data_understanding import run_data_understanding
from step2_data_preparation import run_data_preparation
from step3_data_modeling import run_data_modeling
from step4_data_testing import run_data_testing
from step5_evaluation import run_evaluation


def main(show_plots: bool = True, save_artifacts: bool = True) -> None:
    """Execute all five stages of the hotel cancellation pipeline."""
    raw_df = run_data_understanding(show_plots=show_plots)
    prepared = run_data_preparation(
        raw_df=raw_df, show_plots=show_plots, save_artifacts=save_artifacts
    )
    models = run_data_modeling(prepared=prepared, save_artifacts=save_artifacts)
    test_results = run_data_testing(
        prepared=prepared,
        models=models,
        show_plots=show_plots,
        save_artifacts=save_artifacts,
    )
    run_evaluation(
        prepared=prepared,
        models=models,
        test_results=test_results,
        show_plots=show_plots,
    )


if __name__ == "__main__":
    main()

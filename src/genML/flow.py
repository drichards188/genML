"""
CrewAI Flow for Generic Machine Learning Pipeline

This module defines a generalized ML pipeline that can work with different datasets
while maintaining the core workflow of data loading, feature engineering, model training,
and prediction generation. The flow uses CrewAI's orchestration system to ensure
proper sequencing and error handling between pipeline stages.
"""
import json
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from crewai.flow import Flow, listen, start

# Import ML pipeline functions - these handle the core data science operations
from src.genML.pipeline import (
    engineer_features,  # Feature engineering and preprocessing
    generate_predictions,  # Final predictions and submission file creation
    load_dataset,  # Data loading and initial exploration
    run_model_selection_advisor,  # AI model advisor
    train_model_pipeline,  # Model training and selection
)

# Import progress tracking
from src.genML.progress_tracker import ProgressTracker
from src.genML.resource_monitor import ResourceMonitor


class PipelineAbort(Exception):
    """Signal that the pipeline should stop immediately."""


def _abort_if_failed(stage_result, stage_name):
    """Raise PipelineAbort when an upstream stage marked itself as failed."""
    if isinstance(stage_result, dict) and stage_result.get("status") == "failed":
        reason = stage_result.get("error") or "unknown reason"
        message = f"{stage_name} reported failure: {reason}"
        print(f"❌ Aborting pipeline - {message}")
        raise PipelineAbort(message)

class MLPipelineFlow(Flow):
    """
    Orchestrates a complete machine learning pipeline using CrewAI Flow system.

    This class defines the sequence of ML operations and handles data flow between stages.
    Each stage is a method decorated with CrewAI flow decorators that control execution order
    and data passing. The pipeline is designed to be resilient with proper error handling
    and status tracking between stages.
    """

    def __init__(self, dataset_name: str = "unknown"):
        """Initialize the pipeline flow with progress tracking."""
        super().__init__()

        # Initialize progress tracker
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.progress_tracker = ProgressTracker(run_id=run_id, dataset_name=dataset_name)
        self.resource_monitor = ResourceMonitor(self.progress_tracker)

        # Store in pipeline modules' config so they can access it
        import src.genML.pipeline.config as config
        config.PROGRESS_TRACKER = self.progress_tracker

    @start()
    def load_data_task(self):
        """
        Stage 1: Data Loading and Initial Exploration

        This is the entry point of the ML pipeline. It loads training and test datasets,
        performs initial data exploration, and validates data quality. This stage is critical
        because all subsequent stages depend on properly loaded and understood data.

        Returns:
            dict: Data summary with shapes, columns, missing values, and basic statistics
        """
        print("🔄 Step 1: Loading and exploring dataset...")

        # Start progress tracking
        self.progress_tracker.track_stage_start(1, "Data Loading", "Loading and exploring dataset")
        self.resource_monitor.start()

        # Execute data loading function - returns structured JSON with data insights
        # This approach allows for detailed error handling and progress tracking
        try:
            result_json = load_dataset()
            data_summary = json.loads(result_json)

            print("=== DATA LOADING RESULTS ===")
            print(json.dumps(data_summary, indent=2))

            # Mark stage as complete
            self.progress_tracker.track_stage_complete(1, {
                "rows": data_summary.get("train_shape", [0])[0],
                "columns": data_summary.get("train_shape", [0, 0])[1]
            })

            return data_summary

        except PipelineAbort:
            raise
        except Exception as e:
            print("❌ DATA LOADING FAILED:", str(e))
            self.progress_tracker.set_pipeline_status("failed")
            raise PipelineAbort(f"Data loading failed: {e}") from e

    @listen(load_data_task)
    def feature_engineering_task(self, data_summary):
        """
        Stage 2: Feature Engineering and Preprocessing

        This stage transforms raw data into ML-ready features. It handles missing values,
        creates derived features, encodes categorical variables, and scales numerical features.
        Feature engineering is often the most impactful stage for model performance, as it
        determines what information the model can learn from.

        Args:
            data_summary (dict): Results from data loading stage containing data insights

        Returns:
            dict: Feature engineering results with processed feature information
        """
        print("\n🔄 Step 2: Engineering features from the dataset...")

        # Track stage start
        self.progress_tracker.track_stage_start(2, "Feature Engineering", "Engineering features from dataset")

        # Check if previous stage failed - critical for pipeline integrity
        _abort_if_failed(data_summary, "Data loading")

        # Execute feature engineering pipeline
        try:
            result_json = engineer_features()
            feature_results = json.loads(result_json)

            print("\n=== FEATURE ENGINEERING RESULTS ===")
            print(json.dumps(feature_results, indent=2))

            # Mark stage complete
            self.progress_tracker.track_stage_complete(2, {
                "num_features": feature_results.get("num_features", 0)
            })

            return feature_results

        except PipelineAbort:
            raise
        except Exception as e:
            print("❌ FEATURE ENGINEERING FAILED:", str(e))
            self.progress_tracker.set_pipeline_status("failed")
            raise PipelineAbort(f"Feature engineering failed: {e}") from e

    @listen(feature_engineering_task)
    def model_selection_task(self, feature_results):
        """
        Stage 3: AI-assisted model selection guidance.

        Executes the model selection advisor to rank supported models before training.
        """
        print("\n🔄 Step 3: Consulting AI model advisor for recommended algorithms...")

        # Track stage start
        self.progress_tracker.track_stage_start(3, "Model Selection", "Consulting AI model advisor")

        _abort_if_failed(feature_results, "Feature engineering")

        try:
            result_json = run_model_selection_advisor()
            advisor_results = json.loads(result_json)

            status = advisor_results.get("status", "unknown")
            print("=== MODEL SELECTION ADVISOR RESULTS ===")
            print(json.dumps(advisor_results, indent=2))

            # Track AI insight
            if status == "success":
                self.progress_tracker.track_ai_insight("model_selection", advisor_results)

            # Mark stage complete
            self.progress_tracker.track_stage_complete(3)

            feature_results['model_advisor'] = advisor_results
            feature_results['model_advisor_status'] = status
            return feature_results
        except PipelineAbort:
            raise
        except Exception as e:
            print(f"❌ MODEL ADVISOR FAILED: {e}")
            feature_results['model_advisor_error'] = str(e)
            return feature_results

    @listen(model_selection_task)
    def model_training_task(self, feature_results):
        """
        Stage 4: Model Training and Selection

        This stage trains multiple ML models using cross-validation and selects the best
        performing one. Model selection is crucial for achieving optimal performance on
        unseen data. The stage evaluates different algorithms and uses statistical validation
        to ensure the selected model generalizes well.

        Args:
            feature_results (dict): Results from feature engineering containing processed features

        Returns:
            dict: Model training results with performance metrics and best model info
        """
        print("\n🔄 Step 4: Training and selecting the best ML model...")

        # Track stage start
        self.progress_tracker.track_stage_start(4, "Model Training", "Training and selecting best model")

        # Ensure previous stage completed successfully
        _abort_if_failed(feature_results, "Model advisor / feature engineering")

        # Execute model training and selection pipeline
        try:
            result_json = train_model_pipeline()
            model_results = json.loads(result_json)

            print("\n=== MODEL TRAINING RESULTS ===")
            print(json.dumps(model_results, indent=2))

            # Mark stage complete
            self.progress_tracker.track_stage_complete(4, {
                "best_model": model_results.get("best_model"),
                "best_score": model_results.get("best_score")
            })

            return model_results

        except PipelineAbort:
            raise
        except Exception as e:
            print("❌ MODEL TRAINING FAILED:", str(e))
            self.progress_tracker.set_pipeline_status("failed")
            raise PipelineAbort(f"Model training failed: {e}") from e

    @listen(model_training_task)
    def prediction_task(self, model_results):
        """
        Stage 5: Prediction Generation and Submission Creation

        The final stage of the ML pipeline. It uses the trained model to generate predictions
        on the test dataset and creates a properly formatted submission file. This stage is
        critical for competition submission and real-world model deployment.

        Args:
            model_results (dict): Results from model training containing best model info

        Returns:
            dict: Prediction results with submission file info and prediction statistics
        """
        print("\n🔄 Step 5: Generating predictions and creating submission file...")

        # Track stage start
        self.progress_tracker.track_stage_start(5, "Prediction", "Generating predictions and creating submission")

        # Verify model training completed successfully
        _abort_if_failed(model_results, "Model training")

        # Execute prediction generation and submission file creation
        try:
            result_json = generate_predictions()
            prediction_results = json.loads(result_json)

            print("\n=== PREDICTION RESULTS ===")
            print(json.dumps(prediction_results, indent=2))
            print("\n=== PIPELINE COMPLETED ===")
            print("✅ Submission file 'submission.csv' has been created and is ready for upload!")

            # Mark stage and pipeline complete
            self.progress_tracker.track_stage_complete(5, {
                "submission_file": prediction_results.get("submission_file")
            })
            self.progress_tracker.set_pipeline_status("completed")

            # Archive this run
            self.progress_tracker.archive_run()

            # Stop resource monitoring
            self.resource_monitor.stop()

            return prediction_results

        except PipelineAbort:
            raise
        except Exception as e:
            print("❌ PREDICTION GENERATION FAILED:", str(e))
            self.progress_tracker.set_pipeline_status("failed")
            self.resource_monitor.stop()
            raise PipelineAbort(f"Prediction generation failed: {e}") from e


def create_ml_pipeline_flow(dataset_name: str = "unknown"):
    """
    Factory function to create and return the ML pipeline flow.

    This factory pattern provides a clean interface for creating flow instances
    and allows for future extensibility (e.g., passing configuration parameters).

    Args:
        dataset_name: Name of the dataset being processed

    Returns:
        MLPipelineFlow: A configured ML pipeline flow ready for execution
    """
    return MLPipelineFlow(dataset_name=dataset_name)

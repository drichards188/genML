"""
CrewAI Flow for Generic Machine Learning Pipeline

This module defines a generalized ML pipeline that can work with different datasets
while maintaining the core workflow of data loading, feature engineering, model training,
and prediction generation. The flow uses CrewAI's orchestration system to ensure
proper sequencing and error handling between pipeline stages.
"""
import json
from crewai import Agent, Task, Crew, Process
from crewai.flow import Flow, listen, start

# Import ML pipeline functions - these handle the core data science operations
from src.genML.tools import (
    load_dataset,            # Data loading and initial exploration
    engineer_features,       # Feature engineering and preprocessing
    run_model_selection_advisor,  # AI model advisor
    train_model_pipeline,    # Model training and selection
    generate_predictions     # Final predictions and submission file creation
)


class MLPipelineFlow(Flow):
    """
    Orchestrates a complete machine learning pipeline using CrewAI Flow system.

    This class defines the sequence of ML operations and handles data flow between stages.
    Each stage is a method decorated with CrewAI flow decorators that control execution order
    and data passing. The pipeline is designed to be resilient with proper error handling
    and status tracking between stages.
    """

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

        # Execute data loading function - returns structured JSON with data insights
        # This approach allows for detailed error handling and progress tracking
        try:
            result_json = load_dataset()
            data_summary = json.loads(result_json)

            print("=== DATA LOADING RESULTS ===")
            print(json.dumps(data_summary, indent=2))

            return data_summary

        except Exception as e:
            # Error handling is crucial - we need to fail gracefully and propagate
            # the failure status to prevent downstream stages from executing with bad data
            error_result = {
                "error": f"Data loading failed: {str(e)}",
                "status": "failed"
            }
            print("❌ DATA LOADING FAILED:", str(e))
            return error_result

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

        # Check if previous stage failed - critical for pipeline integrity
        if data_summary.get("status") == "failed":
            print("❌ Skipping feature engineering - data loading failed")
            return data_summary

        # Execute feature engineering pipeline
        try:
            result_json = engineer_features()
            feature_results = json.loads(result_json)

            print("\n=== FEATURE ENGINEERING RESULTS ===")
            print(json.dumps(feature_results, indent=2))

            return feature_results

        except Exception as e:
            # Capture and propagate feature engineering failures
            error_result = {
                "error": f"Feature engineering failed: {str(e)}",
                "status": "failed"
            }
            print("❌ FEATURE ENGINEERING FAILED:", str(e))
            return error_result

    @listen(feature_engineering_task)
    def model_selection_task(self, feature_results):
        """
        Stage 3: AI-assisted model selection guidance.

        Executes the model selection advisor to rank supported models before training.
        """
        print("\n🔄 Step 3: Consulting AI model advisor for recommended algorithms...")

        if feature_results.get("status") == "failed":
            print("❌ Skipping model advisor - feature engineering failed")
            return feature_results

        try:
            result_json = run_model_selection_advisor()
            advisor_results = json.loads(result_json)

            status = advisor_results.get("status", "unknown")
            print("=== MODEL SELECTION ADVISOR RESULTS ===")
            print(json.dumps(advisor_results, indent=2))

            feature_results['model_advisor'] = advisor_results
            feature_results['model_advisor_status'] = status
            return feature_results
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

        # Ensure previous stage completed successfully
        if feature_results.get("status") == "failed":
            print("❌ Skipping model training - feature engineering failed")
            return feature_results

        # Execute model training and selection pipeline
        try:
            result_json = train_model_pipeline()
            model_results = json.loads(result_json)

            print("\n=== MODEL TRAINING RESULTS ===")
            print(json.dumps(model_results, indent=2))

            return model_results

        except Exception as e:
            # Handle model training failures - these can occur due to data issues,
            # hyperparameter problems, or computational constraints
            error_result = {
                "error": f"Model training failed: {str(e)}",
                "status": "failed"
            }
            print("❌ MODEL TRAINING FAILED:", str(e))
            return error_result

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
        print("\n🔄 Step 4: Generating predictions and creating submission file...")

        # Verify model training completed successfully
        if model_results.get("status") == "failed":
            print("❌ Skipping predictions - model training failed")
            return model_results

        # Execute prediction generation and submission file creation
        try:
            result_json = generate_predictions()
            prediction_results = json.loads(result_json)

            print("\n=== PREDICTION RESULTS ===")
            print(json.dumps(prediction_results, indent=2))
            print("\n=== PIPELINE COMPLETED ===")
            print("✅ Submission file 'submission.csv' has been created and is ready for upload!")

            return prediction_results

        except Exception as e:
            # Handle prediction failures - these might occur due to model loading issues,
            # test data problems, or file writing permissions
            error_result = {
                "error": f"Prediction generation failed: {str(e)}",
                "status": "failed"
            }
            print("❌ PREDICTION GENERATION FAILED:", str(e))
            return error_result


def create_ml_pipeline_flow():
    """
    Factory function to create and return the ML pipeline flow.

    This factory pattern provides a clean interface for creating flow instances
    and allows for future extensibility (e.g., passing configuration parameters).

    Returns:
        MLPipelineFlow: A configured ML pipeline flow ready for execution
    """
    return MLPipelineFlow()

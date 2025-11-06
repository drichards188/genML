#!/usr/bin/env python3
"""
Main entry point for the CrewAI ML Pipeline project.

This script orchestrates a complete machine learning pipeline for various datasets
using CrewAI Flows. The pipeline is designed to work with any dataset that follows
the standard train.csv/test.csv format.

Usage:
    python main.py  (from project root)

Prerequisites:
    - Place train.csv and test.csv dataset files in the project root directory
    - Install required dependencies: pip install -r requirements.txt
"""
import os
import sys
import time
import logging
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.genML.flow import PipelineAbort, create_ml_pipeline_flow
from src.genML.logging_config import setup_logging
from src.genML.pipeline import config as pipeline_config


def check_data_files():
    """
    Check if data source is configured.

    Supports two modes:
    1. Ingestion mode: INGESTION_CONFIG is set (no CSV files needed)
    2. Legacy mode: train.csv and test.csv files must exist
    """
    # Check if using ingestion pipeline mode
    if pipeline_config.INGESTION_CONFIG is not None:
        print("✅ Using ingestion pipeline mode")
        data_source = pipeline_config.INGESTION_CONFIG.get('data_source', {})
        source_type = data_source.get('type', 'unknown')
        print(f"   Data source type: {source_type}")

        # Validate data source configuration
        if source_type in ['sql', 'postgresql', 'mysql', 'sqlite']:
            print(f"   Connection: {data_source.get('connection_string', 'N/A')[:50]}...")
        elif source_type in ['nosql', 'mongodb']:
            print(f"   Database: {data_source.get('database', 'N/A')}")
            print(f"   Collection: {data_source.get('collection', 'N/A')}")
        elif source_type == 'csv':
            print(f"   File: {data_source.get('file_path', 'N/A')}")

        return True

    # Legacy CSV mode: check for train.csv and test.csv
    print("Using legacy CSV discovery mode")
    required_files = ['train.csv', 'test.csv']

    # Check datasets/current first, then project root
    dataset_paths = [
        project_root / 'datasets' / 'current',
        project_root
    ]

    for base_path in dataset_paths:
        missing_files = []
        for file in required_files:
            file_path = base_path / file
            if not file_path.exists():
                missing_files.append(file)

        if not missing_files:
            print(f"✅ Data files found: train.csv and test.csv")
            if base_path != project_root:
                print(f"   Using dataset from: {base_path.relative_to(project_root)}")
            return True

    print("❌ Missing required data files:")
    print("   - train.csv and/or test.csv not found")
    print("\nPlease either:")
    print("1. Place dataset files in one of these locations:")
    print("   - datasets/current/ (recommended for organized datasets)")
    print("   - project root directory")
    print("2. OR configure INGESTION_CONFIG in src/genML/pipeline/config.py")
    print("   to load data from databases or other sources")
    return False


def detect_dataset_name():
    """Detect the dataset name from the current dataset location"""
    dataset_paths = [
        project_root / 'datasets' / 'current',
        project_root
    ]

    for base_path in dataset_paths:
        train_file = base_path / 'train.csv'
        if train_file.exists():
            # If using datasets/current, try to find original dataset name
            if base_path.name == 'current':
                # Check which dataset folder has matching files
                datasets_dir = project_root / 'datasets'
                for subdir in datasets_dir.iterdir():
                    if subdir.is_dir() and subdir.name not in ['current', 'active']:
                        subdir_train = subdir / 'train.csv'
                        if subdir_train.exists():
                            # Simple heuristic: if file sizes match, likely same dataset
                            if abs(train_file.stat().st_size - subdir_train.stat().st_size) < 1000:
                                return subdir.name
                return 'current'
            return base_path.name

    return 'unknown'


def main():
    """Main function to run the ML Pipeline Flow"""
    print("=" * 60)
    print("🤖 CREWAI ML PIPELINE FLOW")
    print("=" * 60)
    print("This pipeline will:")
    print("1. 📊 Load and explore the dataset")
    print("2. 🔧 Engineer features for machine learning")
    print("3. 🤖 Train and select the best ML model")
    print("4. 📈 Generate predictions and create submission file")
    print("=" * 60)

    # Change to project root for file operations
    os.chdir(project_root)

    # Check if data files exist
    if not check_data_files():
        return 1

    # Detect dataset name and initialize logging
    dataset_name = detect_dataset_name()
    log_filepath = setup_logging(dataset_name=dataset_name)
    logger = logging.getLogger(__name__)

    print(f"📝 Logging to: {log_filepath}")
    logger.info("=" * 60)
    logger.info("ML Pipeline Started")
    logger.info(f"Dataset: {dataset_name}")
    logger.info("=" * 60)

    try:
        # Create and run the flow
        print("\n🚀 Starting CrewAI Flow execution...")
        print("-" * 40)

        start_time = time.time()

        # Initialize and run the flow
        logger.info("Initializing ML Pipeline Flow")
        flow = create_ml_pipeline_flow(dataset_name=dataset_name)
        logger.info("Starting flow execution")
        flow.kickoff()

        end_time = time.time()
        execution_time = end_time - start_time

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Total execution time: {execution_time:.2f} seconds")
        logger.info("=" * 60)

        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
        print("\n📁 Generated files:")

        # Check outputs folder structure
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            print(f"   📁 {outputs_dir}/")

            # Check each subdirectory
            subdirs = ['data', 'features', 'models', 'predictions', 'reports', 'logs', 'submissions']
            for subdir in subdirs:
                subdir_path = outputs_dir / subdir
                if subdir_path.exists():
                    files = list(subdir_path.glob('*'))
                    print(f"   ├── 📂 {subdir}/ ({len(files)} files)")
                    logger.info(f"Output directory {subdir}/: {len(files)} files")
                    for file in sorted(files):
                        size = file.stat().st_size
                        print(f"   │   ✅ {file.name} ({size} bytes)")
                        logger.debug(f"  - {file.name} ({size} bytes)")
                else:
                    print(f"   ├── 📂 {subdir}/ (missing)")
        else:
            print("   ❌ outputs/ directory not found")

        # Check submission files
        submission_dir = Path("outputs/submissions")
        if submission_dir.exists():
            submission_files = list(submission_dir.glob('*.csv'))
            if submission_files:
                latest_submission = max(submission_files, key=lambda p: p.stat().st_mtime)
                size = latest_submission.stat().st_size
                print(f"\n   ✅ Latest submission: {latest_submission.name} ({size} bytes)")
                logger.info(f"Latest submission file: {latest_submission.name} ({size} bytes)")

        print("\n🎯 Your submission file is ready in outputs/submissions/!")
        print("   Use this file for competition submissions or model evaluation.")
        print(f"📝 Full logs saved to: {log_filepath}")

        return 0

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        print("\n\n⚠️  Pipeline interrupted by user.")
        return 1
    except PipelineAbort as e:
        logger.error(f"Pipeline aborted: {e}")
        print(f"\n\n⚠️ Pipeline aborted: {e}")
        print("Please address the issue above before re-running the pipeline.")
        print(f"📝 Full log saved to: {log_filepath}")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        print(f"\n\n❌ Pipeline failed with error: {str(e)}")
        print("Please check the error messages above for more details.")
        print(f"📝 Full error log saved to: {log_filepath}")
        return 1


def kickoff():
    """Entry point for CrewAI CLI"""
    main()


def plot():
    """Plot flow diagram"""
    flow = create_ml_pipeline_flow()
    flow.plot()


if __name__ == "__main__":
    sys.exit(main())

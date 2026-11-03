#!/usr/bin/env python3
"""
Main entry point for the CrewAI ML Pipeline project.

This script orchestrates a complete machine learning pipeline for various datasets
using CrewAI Flows. The pipeline is designed to work with any dataset that follows
the standard train.csv/test.csv format.

Usage:
    python src/genML/main.py                         # legacy: datasets/current/, everything inferred
    python src/genML/main.py --problem fraud-s6e11   # run a declared problem (problems/<name>/problem.yaml)
    python src/genML/main.py --problem NAME --download   # fetch Kaggle data first if missing
    python src/genML/main.py --list-problems
    python src/genML/main.py --init-kaggle playground-series-s6e11 --name fraud-s6e11
"""
import argparse
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
from src.genML.pipeline import context as pipeline_context
from src.genML.pipeline.problem import list_problems, load_problem, locate_data_dir


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


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the genML pipeline")
    parser.add_argument("--problem", help="Problem name (problems/<name>/problem.yaml) or path to a spec")
    parser.add_argument("--download", action="store_true", help="Download Kaggle data if it is missing")
    parser.add_argument("--list-problems", action="store_true", help="List declared problems and exit")
    parser.add_argument("--init-kaggle", metavar="COMPETITION",
                        help="Create a problem spec for a Kaggle competition and exit")
    parser.add_argument("--name", help="Problem name for --init-kaggle (default: the competition slug)")
    return parser.parse_args(argv)


def _prepare_problem(args) -> str | None:
    """Register the requested problem spec; returns its name, or None for legacy mode."""
    pipeline_context.reset()
    if not args.problem:
        return None
    spec = load_problem(args.problem)
    data_dir = locate_data_dir(spec, download=args.download)
    spec.source = {**spec.source, "path": str(data_dir)}
    pipeline_context.set_requested_spec(spec)
    print(f"✅ Problem '{spec.name}' using data from {data_dir}")
    return spec.name


def main(argv=None):
    """Main function to run the ML Pipeline Flow"""
    args = _parse_args(argv)
    os.chdir(project_root)

    if args.list_problems:
        names = list_problems()
        print("Declared problems:" if names else "No problems declared under problems/")
        for name in names:
            spec = load_problem(name)
            print(f"  {name}: {spec.description or ''} [{spec.task or '?'} / {spec.metric or '?'}]")
        return 0

    if args.init_kaggle:
        from src.genML.pipeline.kaggle import init_problem_from_kaggle

        path, resolved = init_problem_from_kaggle(args.init_kaggle, name=args.name)
        print(f"✅ Wrote {path}")
        print(f"   target={resolved.target} id={resolved.id_column} task={resolved.task} "
              f"metric={resolved.metric} output={resolved.output}")
        if resolved.inferred:
            print("   Inferred (review these):")
            for key, reason in resolved.inferred.items():
                print(f"     {key}: {reason}")
        return 0
    print("=" * 60)
    print("🤖 CREWAI ML PIPELINE FLOW")
    print("=" * 60)
    print("This pipeline will:")
    print("1. 📊 Load and explore the dataset")
    print("2. 🔧 Engineer features for machine learning")
    print("3. 🤖 Train and select the best ML model")
    print("4. 📈 Generate predictions and create submission file")
    print("=" * 60)

    try:
        problem_name = _prepare_problem(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"❌ {exc}")
        return 1

    # Legacy mode: check data files exist
    if problem_name is None and not check_data_files():
        return 1

    # Detect dataset name and initialize logging
    dataset_name = problem_name or detect_dataset_name()
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

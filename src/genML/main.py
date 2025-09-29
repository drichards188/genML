#!/usr/bin/env python3
"""
Main entry point for the CrewAI ML Pipeline project.

This script orchestrates a complete machine learning pipeline for various datasets
using CrewAI Flows. The pipeline is designed to work with any dataset that follows
the standard train.csv/test.csv format.

Usage:
    python main.py  (from project root)
    OR
    crewai run      (using CrewAI CLI)

Prerequisites:
    - Place train.csv and test.csv dataset files in the project root directory
    - Install required dependencies: pip install -r requirements.txt
"""
import os
import sys
import time
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.genML.flow import create_ml_pipeline_flow


def check_data_files():
    """Check if required dataset files are present in datasets/current or project root"""
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
    print("\nPlease place dataset files in one of these locations:")
    print("- datasets/current/ (recommended for organized datasets)")
    print("- project root directory")
    return False


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

    try:
        # Create and run the flow
        print("\n🚀 Starting CrewAI Flow execution...")
        print("-" * 40)

        start_time = time.time()

        # Initialize and run the flow
        flow = create_ml_pipeline_flow()
        flow.kickoff()

        end_time = time.time()
        execution_time = end_time - start_time

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
            subdirs = ['data', 'features', 'models', 'predictions', 'reports']
            for subdir in subdirs:
                subdir_path = outputs_dir / subdir
                if subdir_path.exists():
                    files = list(subdir_path.glob('*'))
                    print(f"   ├── 📂 {subdir}/ ({len(files)} files)")
                    for file in sorted(files):
                        size = file.stat().st_size
                        print(f"   │   ✅ {file.name} ({size} bytes)")
                else:
                    print(f"   ├── 📂 {subdir}/ (missing)")
        else:
            print("   ❌ outputs/ directory not found")

        # Check main submission file
        submission_file = Path("submission.csv")
        if submission_file.exists():
            size = submission_file.stat().st_size
            print(f"   ✅ submission.csv ({size} bytes) - Main submission file")
        else:
            print("   ❌ submission.csv (missing)")

        print("\n🎯 Your submission.csv file is ready for upload!")
        print("   Use this file for competition submissions or model evaluation.")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {str(e)}")
        print("Please check the error messages above for more details.")
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

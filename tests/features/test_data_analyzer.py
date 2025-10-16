import numpy as np
import pandas as pd

from src.genML.features.data_analyzer import DataTypeAnalyzer


def test_small_integer_series_not_misclassified():
    df = pd.DataFrame({
        'num_reported_accidents': [0, 1, 2, 3, 4, 5, 1, 2, 3, 4],
        'small_sequence': list(range(1, 11)),
        'timestamp': [1_609_459_200 + 86_400 * i for i in range(10)]  # Unix seconds for 10 days
    })

    analyzer = DataTypeAnalyzer()
    analysis = analyzer.analyze_dataset(df)

    # Count-like column should remain numerical, not datetime or ID
    assert analysis['column_types']['num_reported_accidents']['detected_type'] == 'numerical'
    assert analysis['column_types']['small_sequence']['detected_type'] != 'datetime'
    assert analysis['column_types']['small_sequence']['detected_type'] != 'id'

    # Large magnitude numeric column should still be recognized as datetime
    assert analysis['column_types']['timestamp']['detected_type'] == 'datetime'


def test_manual_overrides_force_expected_types():
    df = pd.DataFrame({
        'holiday': [f'Holiday {i}' for i in range(10)],
        'speed_limit': [25, 35, 45, 55, 65, 25, 35, 45, 55, 65],
        'num_lanes': [1, 2, 3, 4, 2, 3, 4, 1, 2, 3]
    })

    overrides = {
        'holiday': 'categorical',
        'speed_limit': 'numerical',
        'num_lanes': 'numerical'
    }

    analyzer = DataTypeAnalyzer()
    analysis = analyzer.analyze_dataset(df, overrides=overrides)

    assert analysis['column_types']['holiday']['detected_type'] == 'categorical'
    assert 'holiday' in analysis['categorical_columns']

    assert analysis['column_types']['speed_limit']['detected_type'] == 'numerical'
    assert 'speed_limit' in analysis['numerical_columns']

    assert analysis['column_types']['num_lanes']['detected_type'] == 'numerical'
    assert 'num_lanes' in analysis['numerical_columns']

    assert 'manual_override' in analysis['column_types']['holiday']['patterns']


def test_small_range_ids_not_flagged_but_real_ids_preserved():
    small_range_df = pd.DataFrame({
        'record_sequence': list(range(1, 11))
    })

    analyzer = DataTypeAnalyzer()
    small_range_analysis = analyzer.analyze_dataset(small_range_df)

    assert small_range_analysis['column_types']['record_sequence']['detected_type'] != 'id'

    real_id_df = pd.DataFrame({
        'record_id': np.arange(100, 130)
    })

    real_id_analysis = analyzer.analyze_dataset(real_id_df)
    assert real_id_analysis['column_types']['record_id']['detected_type'] == 'id'

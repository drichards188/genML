import pandas as pd

from src.genML.features.feature_engine import AutoFeatureEngine


def test_interaction_features_and_polynomials_generated():
    df = pd.DataFrame({
        'speed_limit': [25, 35, 45, 55],
        'curvature': [0.10, 0.15, 0.25, 0.40],
        'num_lanes': [1, 2, 3, 4],
        'holiday': ['None', 'Christmas', 'None', 'New Year'],
        'target': [0.2, 0.4, 0.1, 0.6]
    })

    config = {
        'manual_type_hints': {
            'num_lanes': 'numerical',
            'curvature': 'numerical',
            'speed_limit': 'numerical',
            'holiday': 'categorical'
        },
        'interaction_pairs': [
            ('speed_limit', 'curvature')
        ],
        'enable_feature_selection': False,
        'numerical_config': {
            'enable_scaling': False,
            'enable_binning': False,
            'enable_polynomial': True,
            'polynomial_degree': 2,
            'enable_log_transform': False
        },
        'categorical_config': {
            'encoding_method': 'label',
            'enable_frequency': False,
            'max_categories': 10
        }
    }

    engine = AutoFeatureEngine(config)
    engine.analyze_data(df)
    engine.fit(df, target_col='target')

    transformed = engine.transform(df)
    interaction_column = 'speed_limit_x_curvature'

    assert interaction_column in transformed.columns
    # Polynomial feature for speed_limit should exist
    assert any(col.endswith('_squared') for col in transformed.columns if col.startswith('speed_limit'))

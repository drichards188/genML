import lightgbm as lgb
import numpy as np
import sys

print('lightgbm.__version__ =', getattr(lgb, '__version__', 'unknown'))
print('lightgbm.__file__ =', getattr(lgb, '__file__', 'unknown'))

X = np.array([[0.0], [1.0]], dtype=np.float32)
y = np.array([0, 1], dtype=np.int32)

try:
    model = lgb.LGBMClassifier(n_estimators=1, device='gpu')
    model.fit(X, y)
    print('RESULT: LIGHTGBM_GPU_AVAILABLE')
    # Print booster params if possible
    try:
        print('model.get_params() =', model.get_params())
    except Exception:
        pass
except Exception as e:
    print('RESULT: LIGHTGBM_GPU_NOT_AVAILABLE')
    print('ERROR:', repr(e))
    sys.exit(2)


"""Value-net pipeline (training + deployed inference helpers).

``pickup``  — the deployed bidding featurizer + calibration (36-dim hand+trump)
``betli``   — betli value-net training stack (features, net, datagen)
``net``/``net_nnue``/``augment`` — shared model + augmentation building blocks
"""

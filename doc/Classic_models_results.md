---------|-------|----
RandomForestRegressor | 0.08 | Works poorly if n_estimators > 1, better if max_depth > 500
SVR                   |  --  | Kernel hangs during .fit() (No scaling)
knn Regressor         | 0.02 | --

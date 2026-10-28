"""Extensible supervised model and acquisition interfaces (contact-free v2)."""
import importlib
import numpy as np

FEATURE_SCHEMA = 'structure14-v4-geometry27'
FEATURE_NAMES = ('length_mean', 'length_std', 'length_cv', 'orientation_entropy',
                 'degree2', 'degree3', 'degree4', 'degree_other', 'gyration_ratio',
                 'nodes_scaled', 'edges_scaled', 'edges_per_node', 'aspect', 'length_ratio')

# Parameter descriptor: default, minimum, maximum, Chinese, English.
MODEL_SPECS = {
    'mlp': ('浅层神经网络', 'Shallow neural network', {
        'hidden': (32, 4, 256, '隐藏层宽度', 'Hidden width')}),
    'deep_mlp': ('多层神经网络', 'Deep neural network', {
        'hidden': (64, 4, 256, '隐藏层宽度', 'Hidden width'),
        'depth': (3, 2, 5, '隐藏层数', 'Hidden layers')}),
    'ridge': ('岭回归', 'Ridge regression', {
        'alpha': (1.0, .000001, 100., '正则强度', 'Regularization')}),
    'knn': ('近邻回归', 'Nearest neighbors', {
        'neighbors': (5, 1, 30, '邻居数', 'Neighbors')}),
    'random_forest': ('随机森林', 'Random forest', {
        'trees': (80, 10, 300, '树数量', 'Trees'),
        'depth': (8, 2, 30, '最大深度', 'Max depth')}),
    'extra_trees': ('极端随机树', 'Extra trees', {
        'trees': (80, 10, 300, '树数量', 'Trees'),
        'depth': (8, 2, 30, '最大深度', 'Max depth')}),
}
ACQUISITIONS = {
    'random': ('随机采样', 'Random sampling'),
    'diversity': ('最大距离探索', 'Farthest-point exploration'),
    'committee': ('委员会分歧', 'Query by committee'),
    'hybrid': ('分歧与多样性', 'Uncertainty and diversity'),
}
MODEL_PLUGINS = {}
ACQUISITION_PLUGINS = {}


def register_model(key, names, parameters, factory):
    if key in MODEL_SPECS or not callable(factory):
        raise ValueError('model key must be new and factory callable')
    MODEL_SPECS[key] = (*names, parameters)
    MODEL_PLUGINS[key] = factory


def register_acquisition(key, names, select):
    if key in ACQUISITIONS or not callable(select):
        raise ValueError('acquisition key must be new and selector callable')
    ACQUISITIONS[key] = names
    ACQUISITION_PLUGINS[key] = select


def select_candidate(pool, X, Y, method, rng):
    """Select one pool index; only finite physical labels train the committee."""
    pool = np.asarray(pool, float)
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if (pool.ndim != 2 or not len(pool) or not np.isfinite(pool).all() or
            X.ndim != 2 or X.shape[1] != pool.shape[1] or
            Y.shape != (len(X), 3)):
        raise ValueError('candidate pool and labelled arrays have invalid shape')
    if method in ACQUISITION_PLUGINS:
        return int(ACQUISITION_PLUGINS[method](pool, X, Y, rng))
    if method not in ACQUISITIONS:
        raise ValueError('unknown acquisition')
    if method == 'random' or len(X) == 0:
        return int(rng.integers(len(pool)))
    scale = np.std(np.vstack([X, pool]), axis=0)
    scale[scale < 1e-9] = 1.
    distance = np.full(len(pool), np.inf)
    for row in X:
        distance = np.minimum(distance, np.sum(((pool - row) / scale) ** 2, axis=1))
    if method == 'diversity' or len(X) < 8 or not np.isfinite(Y).all():
        return int(np.argmax(distance))
    center = X.mean(axis=0)
    a = np.column_stack([(X - center) / scale, np.ones(len(X))])
    b = np.column_stack([(pool - center) / scale, np.ones(len(pool))])
    ys = np.maximum(Y.std(axis=0), 1e-9)
    predictions = []
    for _ in range(5):
        ids = rng.integers(len(X), size=len(X))
        z = a[ids]
        w = np.linalg.solve(z.T @ z + np.eye(a.shape[1]), z.T @ (Y[ids] / ys))
        predictions.append(b @ w)
    uncertainty = np.var(predictions, axis=0).mean(axis=1)
    if method == 'hybrid':
        uncertainty = uncertainty / max(uncertainty.max(), 1e-9) + distance / max(distance.max(), 1e-9)
    return int(np.argmax(uncertainty))


class Regressor:
    """sklearn adapter with training-only normalization and fixed held-out rows."""
    def __init__(self, key='mlp', params=None, seed=0):
        if key not in MODEL_SPECS:
            raise ValueError('unknown model: ' + key)
        self.key, self.seed = key, int(seed)
        self.params = {k: v[0] for k, v in MODEL_SPECS[key][2].items()}
        self.params.update(params or {})
        for k, value in self.params.items():
            if k in MODEL_SPECS[key][2] and isinstance(MODEL_SPECS[key][2][k][0], int) and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError('parameter must be an integer: ' + k)
            if k not in MODEL_SPECS[key][2] or not MODEL_SPECS[key][2][k][1] <= value <= MODEL_SPECS[key][2][k][2]:
                raise ValueError('invalid model parameter: ' + k)

    def train(self, X, Y, lr=.003, epochs=200, batch=16, val_frac=.2,
              patience=15, min_delta=.0001, epoch_cb=None, stop_cb=None, *,
              feature_schema=None):
        """Publish a completed fit together; callbacks see the previous fit.

        ``feature_schema`` is an optional caller-declared identity, not inferred
        from width or verified from values. Use ``FEATURE_SCHEMA`` only for the
        canonical 14 ``light_features`` columns. Other nonempty printable strings
        (up to 128 characters, no outer whitespace) are opaque declarations.
        History returns ``schema`` (None if undeclared) and ``feature_count``.
        """
        import copy
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        if X.ndim != 2 or Y.shape != (len(X), 3) or len(X) < 5 or not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError('training requires at least 5 finite, physically labelled samples')
        if (not 0 < val_frac < 0.5 or int(epochs) < 1 or int(batch) < 1 or
                int(patience) < 1 or not np.isfinite(lr) or lr <= 0):
            raise ValueError('invalid training controls')
        if feature_schema is not None and (
                not isinstance(feature_schema, str) or
                not 1 <= len(feature_schema) <= 128 or
                not feature_schema.isprintable() or
                feature_schema != feature_schema.strip()):
            raise ValueError('feature_schema must be None or a nonempty printable string '
                             'of at most 128 characters without outer whitespace')
        if feature_schema == FEATURE_SCHEMA and X.shape[1] != len(FEATURE_NAMES):
            raise ValueError('canonical feature_schema requires 14 feature columns')
        if self.key == 'mlp':
            from fibernet.ml.physical_surrogate import MLP
            estimator = MLP(hidden=self.params['hidden'], seed=self.seed)
            if 'estimator' in self.__dict__ and estimator is self.estimator:
                raise ValueError('model factory must return a new estimator for refitting')
            history = estimator.train(X, Y, lr=lr, epochs=epochs, batch=batch,
                val_frac=val_frac, patience=patience, min_delta=min_delta,
                epoch_cb=epoch_cb, stop_cb=stop_cb)
            if not history['train']:
                raise InterruptedError('training stopped before a fitted model was available')
            fitted_state = {name: getattr(estimator, name) for name in
                            ('val_indices', 'train_indices', 'x_mean', 'x_std', 'y_mean', 'y_std')}
            fitted_state['estimator'] = estimator
            history.update(validation_count=len(fitted_state['val_indices']),
                           schema=feature_schema, feature_count=X.shape[1], progress_kind='epoch')
            self.__dict__.update(fitted_state)
            return history
        ids = np.random.default_rng(self.seed).permutation(len(X))
        nv = max(2, min(len(X) - 2, int(len(X) * val_frac)))
        val_indices, train_indices = ids[:nv], ids[nv:]
        ti, vi = train_indices, val_indices
        x_mean, x_std = X[ti].mean(0), X[ti].std(0)
        y_mean, y_std = Y[ti].mean(0), Y[ti].std(0)
        x_std[x_std < 1e-12] = 1.
        y_std[y_std < 1e-12] = 1.
        x, y = (X - x_mean) / x_std, (Y - y_mean) / y_std
        p = self.params
        if self.key in MODEL_PLUGINS:
            estimator = MODEL_PLUGINS[self.key](p, self.seed)
        elif self.key in ('mlp', 'deep_mlp'):
            cls = importlib.import_module('sklearn.neural_network').MLPRegressor
            estimator = cls(hidden_layer_sizes=(p['hidden'],) * p.get('depth', 1),
                                 learning_rate_init=lr, batch_size=min(batch, len(ti)),
                                 max_iter=1, random_state=self.seed)
        elif self.key == 'ridge':
            estimator = importlib.import_module('sklearn.linear_model').Ridge(alpha=p['alpha'])
        elif self.key == 'knn':
            estimator = importlib.import_module('sklearn.neighbors').KNeighborsRegressor(
                n_neighbors=min(p['neighbors'], len(ti)), weights='distance', n_jobs=1)
        else:
            module = importlib.import_module('sklearn.ensemble')
            cls = module.RandomForestRegressor if self.key == 'random_forest' else module.ExtraTreesRegressor
            estimator = cls(n_estimators=p['trees'], max_depth=p['depth'], random_state=self.seed, n_jobs=1, warm_start=True)
        if 'estimator' in self.__dict__ and estimator is self.estimator:
            raise ValueError('model factory must return a new estimator for refitting')
        neural = self.key in ('mlp', 'deep_mlp')
        train, val, best, bad, best_epoch, saved = [], [], np.inf, 0, 0, None
        forest = self.key in ('random_forest', 'extra_trees')
        steps = (list(range(10, p['trees'], 10)) + [p['trees']] if forest
                 else range(1, int(epochs if neural else 1) + 1))
        for ep in steps:
            if stop_cb and stop_cb():
                break
            if neural:
                estimator.partial_fit(x[ti], y[ti])
            else:
                if forest:
                    estimator.set_params(n_estimators=ep)
                estimator.fit(x[ti], y[ti])
            train.append(float(np.mean((estimator.predict(x[ti]) - y[ti]) ** 2)))
            val.append(float(np.mean((estimator.predict(x[vi]) - y[vi]) ** 2)))
            if not neural or val[-1] < best - min_delta:
                best, bad, best_epoch = val[-1], 0, ep
                saved = copy.deepcopy(estimator)
            else:
                bad += 1
            if epoch_cb:
                epoch_cb(ep, train[-1], val[-1])
            if neural and bad >= patience:
                break
        if saved is None:
            raise InterruptedError('training stopped before a fitted model was available')
        fitted_state = dict(estimator=saved, val_indices=val_indices, train_indices=train_indices,
                            x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
        history = dict(train=train, val=val, epochs_run=len(train), best_epoch=best_epoch,
                    early_stopped=neural and bad >= patience, validation_count=len(vi),
                    schema=feature_schema, feature_count=X.shape[1],
                    progress_kind='trees' if forest else ('epoch' if neural else 'fit'),
                    trees_fitted=len(saved.estimators_) if forest else None)
        self.__dict__.update(fitted_state)
        return history

    def predict(self, X):
        """Predict from finite (D,) or (N, D) features at the fitted width D."""
        if (not hasattr(self, 'estimator') or
                getattr(self, 'x_mean', None) is None):
            raise RuntimeError('train the physical regressor before prediction')
        X = np.asarray(X, float)
        if X.ndim not in (1, 2):
            raise ValueError('prediction features must have shape (D,) or (N, D)')
        width = len(self.x_mean)
        if X.shape[-1] != width:
            raise ValueError(
                f'prediction feature width must match fitted width {width}; '
                f'got {X.shape[-1]}')
        if not np.isfinite(X).all():
            raise ValueError('prediction features must be finite')
        if self.key == 'mlp':
            return self.estimator.predict(X)
        out = self.estimator.predict((np.atleast_2d(X) - self.x_mean) / self.x_std)
        out = out * self.y_std + self.y_mean
        return out[0] if X.ndim == 1 else out


PhysicalRegressor = Regressor

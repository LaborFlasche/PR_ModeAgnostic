"""Model wrapper that counts how many rows are scored.

The number of model evaluations (forward passes) is the fair, library-agnostic
x-axis for comparing approximators: each library exposes its budget in different
units (shapiq ``budget``, shap ``nsamples``/``max_evals``, lightshap ``max_iter``,
dalex ``B``), but they all ultimately call the model on masked coalitions. Wrap
the model in ``CountingModel`` before handing it to a backend and read
``.n_rows`` afterwards to get the real evaluation count.
"""

from __future__ import annotations

_COUNTED_METHODS = ("predict", "predict_proba", "decision_function")


class CountingModel:
    """Transparent proxy around a fitted model that counts scored rows.

    ``predict``/``predict_proba`` are wrapped lazily through ``__getattr__`` (not
    defined as methods) so that ``hasattr(counter, "predict_proba")`` reflects the
    *wrapped* model — a regressor stays without ``predict_proba``. Every other
    attribute is forwarded untouched, so the proxy can stand in for the model.
    """

    def __init__(self, model):
        self._model = model
        self.n_calls = 0  # number of predict/predict_proba invocations
        self.n_rows = 0   # total rows scored across all invocations

    def _wrap(self, fn):
        def counted(X, *args, **kwargs):
            self.n_calls += 1
            try:
                self.n_rows += len(X)
            except TypeError:
                self.n_rows += int(getattr(X, "shape", [1])[0])
            return fn(X, *args, **kwargs)
        return counted

    def reset(self) -> None:
        self.n_calls = 0
        self.n_rows = 0

    def __getattr__(self, item):
        # Reached only for attributes not found on CountingModel itself; raises
        # AttributeError if the wrapped model lacks the attribute (so hasattr works).
        attr = getattr(self._model, item)
        if item in _COUNTED_METHODS and callable(attr):
            return self._wrap(attr)
        return attr

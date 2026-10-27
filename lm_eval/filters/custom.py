from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


@register_filter("identity")
class IdentityFilter(Filter):
    """Pass the responses through unchanged.

    Useful when a task scores every sample of a document itself (e.g. the AIME
    `maj@k` / `avg@k` metrics, which need all `--repeats` generations at once):
    the default `take_first` filter would keep only the first sample.
    """

    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        return resps


@register_filter("custom")
class CustomFilter(Filter):
    """
    Custom filter that applies a custom, user-defined function to the model responses.
    """

    def __init__(self, **kwargs) -> None:
        self.filter_fn = kwargs.pop("filter_fn")

        super().__init__(**kwargs)

    def apply(self, resps, docs):
        return self.filter_fn(resps, docs)

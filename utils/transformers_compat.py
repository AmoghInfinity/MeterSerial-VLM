# importing libraries

import transformers


def ensure_transformers_v5_compatibility() -> None:
    """
    Provide compatibility for legacy remote-code models that do not
    initialize all_tied_weights_keys under Transformers 5.x.
    """

    if not hasattr(
        transformers.modeling_utils.PreTrainedModel,
        "all_tied_weights_keys",
    ):
        transformers.modeling_utils.PreTrainedModel.all_tied_weights_keys = {}
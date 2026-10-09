import torch.nn as nn


class Jet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise ModuleNotFoundError(
            "The legacy `cont.modelling.jet` flow encoder is not bundled in this "
            "workspace. SOFTVQ/MAETOK smoke tests should not hit this code path, "
            "but ContinuousTokenizerVAE/FlowModel still require the original JET implementation."
        )

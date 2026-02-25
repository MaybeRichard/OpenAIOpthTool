import torch


def get_device(preference=None):
    """
    自动选择计算设备 / Auto-select compute device.

    Priority: CUDA > MPS > CPU (unless overridden).

    Args:
        preference: "cuda", "cuda:0", "mps", "cpu", or None for auto.

    Returns:
        torch.device
    """
    if preference is not None:
        return torch.device(preference)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

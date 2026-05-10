from torch import accelerator


def fetch_device():
    current_accelerator = accelerator.current_accelerator()
    if current_accelerator and accelerator.is_available():
        return current_accelerator.type
    return "cpu"

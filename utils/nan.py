import torch

def replace_nan(input_data, replace_value):
    """
    Replace NaN values in a tensor or a list of tensors with 1.

    Parameters:
        input_data (torch.Tensor or list[torch.Tensor]): Input tensor or list of tensors to process.

    Returns:
        torch.Tensor or list[torch.Tensor]: Processed tensor or list of tensors with NaN values replaced by 1.
    """
    if isinstance(input_data, torch.Tensor):
        return torch.where(torch.isnan(input_data), torch.tensor(replace_value, dtype=input_data.dtype, device=input_data.device), input_data)
    elif isinstance(input_data, list):
        return [
            torch.where(torch.isnan(t), torch.tensor(replace_value, dtype=t.dtype, device=t.device), t) if isinstance(t, torch.Tensor) else t
            for t in input_data
        ]
    else:
        raise TypeError("Input must be a torch.Tensor or a list of torch.Tensor")

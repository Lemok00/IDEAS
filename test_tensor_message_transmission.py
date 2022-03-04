import torch
import warnings

warnings.simplefilter('ignore')


def message_to_tensor(message, sigma, delta):
    secret_tensor = torch.zeros(size=(message.shape[0], message.shape[1] // sigma))
    step = 2 / 2 ** sigma
    random_interval_size = step * delta
    message_nums = torch.zeros_like(secret_tensor)
    for i in range(sigma):
        message_nums += message[:, i::sigma] * 2 ** (sigma - i - 1)
    secret_tensor = step * (message_nums + 0.5) - 1
    secret_tensor = secret_tensor + (torch.rand_like(secret_tensor) * random_interval_size * 2 - random_interval_size)
    return secret_tensor


def tensor_to_message(secret_tensor, sigma):
    message = torch.zeros(size=(secret_tensor.shape[0], secret_tensor.shape[1] * sigma))
    step = 2 / 2 ** sigma
    secret_tensor = torch.clamp(secret_tensor, min=-1, max=1) + 1
    message_nums = secret_tensor / step
    zeros = torch.zeros_like(message_nums)
    ones = torch.ones_like(message_nums)
    for i in range(sigma):
        zero_one_map = torch.where(message_nums >= 2 ** (sigma - i - 1), ones, zeros)
        message[:, i::sigma] = zero_one_map
        message_nums -= zero_one_map * 2 ** (sigma - i - 1)
    return message


if __name__ == "__main__":
    SIGMA = 3
    DELTA = 0.1
    messages = torch.randint(low=0, high=2, size=(100, 1 * 16 * 16 * SIGMA))
    noises = message_to_tensor(messages, sigma=SIGMA, delta=DELTA)
    print('noises.shape: ',noises.shape)
    recovered_messages = tensor_to_message(noises, sigma=SIGMA)

    ACC_avg = 1 - torch.mean(torch.abs(messages - recovered_messages))

    print(f"ACC AVG: {ACC_avg:.6f}")

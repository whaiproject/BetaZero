import numpy as np
import torch
import torch.nn as nn

def convert_state_to_tensor(s_raw, player_perspective=1):
    s = torch.zeros((2, 3, 3), dtype=torch.int32)
    s[0, s_raw == player_perspective] = 1
    s[1, s_raw == -player_perspective] = 1
    return s

def dist(x, tau=1):
    if tau == 0:
        probs = np.zeros_like(x)
        probs[np.argmax(x)] = 1
        return probs
    elif tau == np.inf:
        # return equal probabilities for all entries
        probs = np.ones_like(x) / np.count_nonzero(x)
        probs[x == 0] = 0
        return probs
    else:
        return np.power(x, 1 / tau) / np.sum(np.power(x, 1 / tau))
    
# AlphaGo Zero Loss
def alphago_zero_loss(model, policy_output, value_output, true_policy, true_value, c=1e-4):
    value_loss = nn.MSELoss()(value_output, true_value)
    policy_loss = -torch.mean(torch.sum(true_policy * torch.log(policy_output + 1e-7), 1))
    l2_reg = c * sum(torch.sum(param ** 2) for param in model.parameters())
    
    # print("#"*45)
    # print(value_loss)
    # a = torch.log(policy_output + 1e-7)
    # isnan = torch.any(torch.isnan(a))
    # isinf = torch.any(torch.isinf(a))
    # #combined_check = isnan | isinf

    # # Check if any element in the tensor is NaN, Inf, or -Inf
    # #contains_unwanted = torch.any(combined_check)
    # print(isnan)
    # print(isinf)   
    # #print(contains_unwanted)
    # print(policy_loss)
    # print(l2_reg)

    total_loss = value_loss + policy_loss + l2_reg
    return total_loss
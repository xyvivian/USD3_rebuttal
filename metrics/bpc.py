import torch

def bpc(samples,pred,num_classes):
    samples = samples.detach().cpu().long()
    pred = pred.detach().cpu().long()
    true_prob = calculate_prob(samples,num_classes)
    pred_prob = calculate_prob(pred,num_classes)
    return torch.nn.functional.cross_entropy(input=pred_prob, 
                                             target=true_prob,
                                             reduction='mean')
    


def calculate_prob(data_tensor, vocab_size):
    one_hot_encoded = torch.nn.functional.one_hot(data_tensor, num_classes=vocab_size)
    counts_per_pos = one_hot_encoded.sum(dim=0).float()  
    probabilities_per_bit = counts_per_pos / counts_per_pos.sum(dim=1, keepdim=True)
    return probabilities_per_bit
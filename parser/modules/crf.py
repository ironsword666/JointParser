import torch.nn as nn
import torch


class TreeCRFLoss(nn.Module):
    def __init__(self, n_labels):
        super(TreeCRFLoss, self).__init__()
        self.n_labels = n_labels
        self.transitions = nn.Parameter(
            torch.Tensor(n_labels, n_labels, n_labels))
        self.start_transitions = nn.Parameter(torch.Tensor(n_labels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)
        torch.nn.init.normal_(self.start_transitions)

    def forward(self, scores):
        
        return scores
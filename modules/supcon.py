import torch
import torch.nn as nn
import torch.distributed as dist

import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N . 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class SupConLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.1, world_size=1):
        super(SupConLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        return mask

    def forward(self, z_i, labels):
        if self.world_size > 1:
            z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
        
        # sample similarity NxN (N=batch_size)
        sim = self.similarity_f(z_i.unsqueeze(1), z_i.unsqueeze(0)) / self.temperature

        # Calculate label similarity, NxN
        label_similarity = torch.matmul(labels, labels.t())
        label_similarity.fill_diagonal_(0)
        # print("label_similarity", label_similarity)

        # Find the indices of the maximum values in the label similarity matrix
        max_label_sim_indices = label_similarity.max(dim=1)[1]

        max_mask = torch.zeros_like(label_similarity, dtype=torch.bool)
        max_mask.scatter_(1, max_label_sim_indices.unsqueeze(1), 1)

        # Get the indices of the non-zero maximum elements
        non_zero_max_indices = torch.nonzero(max_mask & (label_similarity != 0))
        # Check if all non-diagonal elements of label similarity are zero
        non_diag_label_similarity = label_similarity.fill_diagonal_(0)
        if torch.all(non_diag_label_similarity == 0):
            return 0 ## supcon did not work, need to use ntxent
        
        # Create a mask for positive samples
        positive_mask = torch.zeros_like(label_similarity)
        # positive_mask.scatter_(1, non_zero_max_indices, 1) ## puts 1 in the index of the max value in each row, all other entries are 0
        positive_mask[non_zero_max_indices[:,0], non_zero_max_indices[:,1]] = 1
        positive_mask = positive_mask.bool()

        # Mask out the positive samples from the similarity matrix
        sim_masked = sim.masked_fill(positive_mask, float('-inf')) #one positive sample is put as -inf, rest are same and considred negative samples
        # sim_masked = sim.fill_diagonal_(float('-inf'))
        
        # Calculate the loss
        labels = torch.arange(sim.size(0)).to(labels.device)
        # labels = max_label_sim_indices
        # print("labels", labels)
        loss = self.criterion(sim_masked, labels)
        loss /= self.batch_size
        # print("loss", loss)
        if torch.isinf(loss):
            print('########################loss is infinity printing')
            # print('sim_masked', sim_masked)
            # print('positive_mask', positive_mask)
            # loss = 0
        return loss
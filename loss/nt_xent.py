import torch
import numpy as np
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity, alpha_weight):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.alpha_weight = alpha_weight
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def softXEnt(self, target, logits):
        logprobs = torch.nn.functional.log_softmax(logits, dim = 1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(self.batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs,
                    norm=True,
                    weights=1.0):

        temperature = self.temperature
        alpha = self.alpha_weight

        LARGE_NUM = 1e9
        """Compute loss for model.
        Args:
        hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
        hidden_norm: whether or not to use normalization on the hidden vector.
        temperature: a `floating` number for temperature scaling.
        tpu_context: context information for tpu.
        weights: a weighting number or vector.
        Returns:
        A loss scalar.
        The logits for contrastive prediction task.
        The labels for contrastive prediction task.
        """
        # Get (normalized) hidden1 and hidden2.
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)
            
        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float()
        labels = labels.to(self.device)
        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        
    #     logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large,0, 1)) / temperature
    #     logits_aa = logits_aa - masks * LARGE_NUM
    #     logits_bb = torch.matmul(hidden2,  torch.transpose(hidden2_large,0, 1)) / temperature
    #     logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large,0, 1)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large,0, 1)) / temperature

        loss_a = self.softXEnt(
            labels, logits_ab)
        # print(loss_a)
        loss_b = self.softXEnt(
            labels, logits_ba)
        # print(loss_b)

        # loss = alpha*loss_a + (1-alpha)*loss_b
        loss = alpha*loss_a + (1-alpha)*loss_b
        # print(loss)
    
        return loss

        #Loss antiga
        # # representations = torch.cat([zis, zjs], dim=0)
        # similarity_matrix = self.similarity_function(zjs, zis)

        # # filter out the scores from the positive samples
        # l_pos = torch.diag(similarity_matrix, 0)
        # # r_pos = torch.diag(similarity_matrix, -self.batch_size)
        # # print(r_pos)
        # positives = torch.cat([l_pos]).view(self.batch_size, 1)
        # # print(self.mask_samples_from_same_repr)
        # negatives = similarity_matrix[self.mask_samples_from_same_repr].view(self.batch_size, -1)

        # logits = torch.cat((positives, negatives), dim=1)
        # logits /= self.temperature

        # labels = torch.zeros(self.batch_size).to(self.device).long()
        # labels = torch.nn.functional.one_hot(torch.zeros(self.batch_size, dtype = torch.int64), 
        #                                      num_classes=self.batch_size).to(self.device)
        
        # loss = self.softXEnt(logits, labels)

        # return loss*2
from typing import List

import torch
import torch.nn as nn


class FragmentEmbeddings(nn.Module):
    def __init__(self, fragments_num_attachements: List[int], hidden_size: int):
        super(FragmentEmbeddings, self).__init__()
        self.hidden_size = hidden_size

        self._fragment_embeddings = nn.Embedding(len(fragments_num_attachements), hidden_size)

        self.num_fragments = len(fragments_num_attachements)

        self.max_attachement = fragments_num_attachements.max()
        self._attachemment_embeddings = nn.Embedding(self.max_attachement * self.num_fragments, self.hidden_size)

        attach_mask = torch.ones((self.num_fragments, self.max_attachement))

        for i in range(self.num_fragments):
            attach_mask[i, : fragments_num_attachements[i]] = 1

        self.register_buffer("attachement_mask", attach_mask)

    def get_fragment(self, fragment_idx):
        return self._fragment_embeddings(fragment_idx)

    def get_fragment_attachment(self, fragment_idx, attachement_idx):
        return self._attachemment_embeddings(fragment_idx * self.max_attachement + attachement_idx)

    def get_all_fragments(self):
        return self._fragment_embeddings(torch.arange(self.num_fragments))

    def get_all_attachements(self, fragment_idx):
        if len(fragment_idx.shape) == 0:
            fragment_idx = fragment_idx.unsqueeze(0)
        shape = fragment_idx.shape
        indexing = fragment_idx.unsqueeze(-1).expand(*shape, self.max_attachement) + torch.arange(self.max_attachement).unsqueeze(0).expand(
            *shape, self.max_attachement
        )
        return self._attachemment_embeddings(indexing), self.attachement_mask[fragment_idx]

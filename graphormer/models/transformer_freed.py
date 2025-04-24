import sys
from pathlib import Path
from typing import List

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent.parent))

from graphormer.data.dataloader import Dataloader
from graphormer.embeddings.fragments_ebeddings import FragmentEmbeddings
from graphormer.nn.encoder import GraphEncoderWithNodes
from graphormer.utils.nn import masked_cross_entropy


class FreedTransformer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, fragments_data: List[int]):
        super().__init__()
        self.graph_encoder = GraphEncoderWithNodes(node_dim, edge_dim, hidden_dim, 1)
        self.transformers = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(hidden_dim, 8, 4 * hidden_dim), num_layers=4)
        self.reward_encoder = nn.Linear(1, hidden_dim)

        self.fragments = FragmentEmbeddings(fragments_data, hidden_dim)

    def prepare_context(self, batch):
        graphs_encoded, attachements = self.graph_encoder(batch["graphs"], batch["attachements"])

        graphs_encoded = graphs_encoded.reshape(attachements.shape[0], attachements.shape[1], -1)
        rewards_encoded = self.reward_encoder(batch["rewards"].unsqueeze(-1))

        actions = batch["actions"]
        first_action = attachements[:, :, actions[:, :, 0]]
        second_action = self.fragments.get_fragment(actions[:, :, 1])
        thirds_action = self.fragments.get_fragment_attachment(actions[:, :, 1], actions[:, :, 2])
        bs, _, hidden_dim = graphs_encoded.shape

        context = torch.stack([graphs_encoded, first_action, second_action, thirds_action, rewards_encoded], dim=1).reshape((bs, -1, hidden_dim))
        return context, attachements

    def forward(self, batch):
        seq, attachements = self.prepare_context(batch)

        # first_true = seq[:, -4]
        # second_true = seq[:, -3]
        # third_true = seq[:, -2]

        # last_action = batch["actions"][:, -1, :]

        # all_attachements, all_attachements_mask = attachements[:, -1, :, :], batch["masks"][:, -1, :]
        # all_fragments = self.fragments.get_all_fragments()
        # all_fragments_attachements, all_fragments_attachements_mask = self.fragments.get_all_attachements(last_action[:, 1])

        # first_context = torch.cat([all_attachements, seq], dim=1)
        # second_context = torch.cat([all_fragments, seq], dim=1)
        # third_context = torch.cat([all_fragments_attachements, seq], dim=1)

        # first_mask = t
        # second_mask = ...
        # third_mask = ...

        # first_predicted = self.transformers(first_context[:, :-4], src_mask=first_context)[:, -1, :]
        # second_predicted = self.transformers(second_context[:, :-3])[:, -1, :]
        # third_predicted = self.transformers(third_context[:, :-2], src_mask=third_mask)[:, -1, :]

        # loss_first = masked_cross_entropy(first_predicted)

        return

    def act(self, graph: dgl.DGLGraph, attachement_idx: torch.Tensor, timestep):
        graph, attachement_embeddings = self.state_encoder(graph, attachement_idx)
        graph = graph.to(self.device)
        attachement_embeddings = attachement_embeddings.to(self.device).unsqueeze(0)

        graph_encoded = graph + self.positional_encoder(torch.tensor([0], device=self.device))
        context = torch.concat([context, graph_encoded], dim=1)

        first_context = torch.concat([attachement_embeddings, context], dim=1)
        first_concat_output = self.encoders(first_context)

        first_action_embedding = first_concat_output[:, -1, :]

        first_action = torch.argmax(torch.einsum("bjk, bk->bj", attachement_embeddings, first_action_embedding), dim=-1)
        context = torch.concat([context, attachement_embeddings[:, first_action : first_action + 1, :]], dim=1)

        fragments = self.fragments_embedding.get_all_fragments().unsqueeze(0)
        second_context = torch.concat([fragments, context], dim=1)
        second_context_output = self.encoders(second_context)

        second_action_embedding = second_context_output[:, -1, :]
        second_action = torch.argmax(torch.einsum("bjk, bk->bj", fragments, second_action_embedding), dim=-1)
        context = torch.concat([context, attachement_embeddings[:, second_action : second_action + 1, :]], dim=1)

        fragment_attachement_embeddings, fragment_attachement_mask = self.fragments_embedding.get_all_attachements(second_action)
        fragment_attachement = fragment_attachement_embeddings[fragment_attachement_mask == 0].unsqueeze(0)

        third_context = torch.concat([fragment_attachement, context], dim=1)
        third_context_output = self.encoders(third_context)
        third_action_embedding = third_context_output[:, -1, :]

        third_action = torch.argmax(torch.einsum("bjk, bk->bj", fragment_attachement, third_action_embedding), dim=-1)
        context = torch.concat([context, fragment_attachement[:, third_action : third_action + 1, :]], dim=1)

        return (first_action.item(), second_action.item(), third_action.item()), context


if __name__ == "__main__":
    import json

    from rdkit import Chem

    from graphormer.env.mol.state import State

    hidden_size = 512
    fragments_path = "/root/projets/FFREED/zinc_crem.json"
    loader = Dataloader(
        data_root="/root/data/graphs",
        buffer_size=1,  # Number of tasks per buffer
        steps_per_buffer=100,  # How many batches to use before switching buffers
        num_steps=128,  # Sequence length
        batch_size=2,
    )

    with open(fragments_path, "r") as f:
        data = json.loads(f.read())
        bond_vocab = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        fragment_attach = [
            len(
                State(
                    s,
                    0,
                    atom_dim=29,
                    bond_dim=4,
                    bond_vocab=bond_vocab,
                    atom_vocab=list(("H", "C", "N", "O", "S", "P", "F", "I", "Cl", "Br")),
                    attach_vocab=["*"],
                ).get_attachments()
            )
            for s in data
        ]

    model = FreedTransformer(29, 4, 512, torch.tensor(fragment_attach)).to("cuda")
    batch = loader.get_batch().to("cuda")

    model(batch)

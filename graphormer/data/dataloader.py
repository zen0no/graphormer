import queue
import random
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import dgl
import numpy as np
import torch
from tqdm import tqdm


def load_epoch(epoch_path: Path):
    actions = torch.load(epoch_path / "actions.pt")
    attachements = torch.load(epoch_path / "attachements.pt")
    rewards = torch.load(epoch_path / "rewards.pt")
    graphs = dgl.load_graphs(str(epoch_path / "graphs.bin"))[0]
    max_, _ = attachements.max(dim=-1)
    masks = torch.load(epoch_path / "mask.pt")

    return actions, attachements, rewards, graphs, masks


def load_train(train_path: Path):
    paths = sorted(list(train_path.glob("epoch_*")), key=lambda p: int(p.stem.split("_")[-1]))
    result = tqdm(map(load_epoch, paths), total=len(paths))

    zip_data = list(zip(*result))
    actions = torch.cat(zip_data[0])
    attachements = torch.cat(zip_data[1])
    rewards = torch.cat(zip_data[2])
    masks = torch.cat(zip_data[4])
    graphs = sum(zip_data[3], start=[])
    return {"actions": actions, "attachements": attachements, "rewards": rewards, "graphs": graphs, "masks": masks}


class Dataloader:
    def __init__(
        self,
        data_root,
        buffer_size=1,
        steps_per_buffer=500,
        num_steps=1024,
        batch_size=32,
    ):
        self.data_root = Path(data_root)

        self.buffer_size = buffer_size  # Number of tasks per buffer
        self.steps_per_buffer = steps_per_buffer
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.buffer = self._load_buffer(self.buffer_size, num_steps=self.num_steps)

    def _load_buffer(self, num_tasks, num_steps):
        """Load and preprocess a buffer of tasks"""
        task_paths = list(self.data_root.glob("task_*"))
        selected = np.random.choice(task_paths, num_tasks, replace=False)
        buffer = Buffer(
            task_paths=selected,
            num_steps=num_steps,
            batch_size=self.batch_size,
        )

        return buffer

    def get_batch(self):
        """Get next batch with buffer management"""
        if self.buffer.step_counter >= self.steps_per_buffer:
            self.buffer = self._load_buffer()

        return self.buffer.sample()


class Buffer:
    def __init__(self, task_paths, num_steps, batch_size):
        self.task_paths = task_paths
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.step_counter = 0

        self.data = [load_train(train_path=path) for path in self.task_paths]

    def sample(self):
        train_id = random.randint(0, len(self.data) - 1)
        data = self.data[train_id]
        total_len = data["actions"].shape[0]

        start_index = torch.randint(0, total_len - self.num_steps, size=(self.batch_size,), dtype=torch.long)
        index = torch.arange(self.num_steps).expand((self.batch_size, -1)) + start_index.unsqueeze(-1).expand((-1, self.num_steps))

        batch_attachements = data["attachements"][index]
        batch_actions = data["actions"][index]
        batch_rewards = data["rewards"][index]
        batch_masks = data["masks"][index]

        batch_graphs = []
        for i, idx in enumerate(start_index):
            k = data["graphs"][idx : idx + self.num_steps]
            batch_graphs.extend(k)
            # print(batch_attachements.shape)
            # m, _ = batch_attachements[i].max(dim=-1)
            # for j, (g, m) in enumerate(zip(k, m)):
            #     print(m.item(), g.ndata["x"].shape[0])
            #     assert m.item() < g.ndata["x"].shape[0]

        batch_graphs = dgl.batch(batch_graphs)

        nodes = batch_graphs.batch_num_nodes()

        cum_sum = nodes.cumsum(0) - nodes[0]
        shape = batch_attachements.shape
        batch_attachements = batch_attachements.reshape(cum_sum.shape + shape[-1:]) + cum_sum.unsqueeze(-1).expand((-1, shape[-1]))
        batch_attachements = (batch_attachements.reshape(shape) * batch_masks).to(dtype=torch.long)

        return Batch({"actions": batch_actions, "attachements": batch_attachements, "rewards": batch_rewards, "graphs": batch_graphs, "masks": batch_masks})


class Batch:
    def __init__(self, data=None):
        if not data:
            self.data = dict()
        else:
            self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def to(self, device):
        for k, v in self.data.items():
            self.data[k] = v.to(device)

        return self


# if __name__ == "__main__":
#     torch.random.manual_seed(0)
#     torch.cuda.random.manual_seed(0)
#     random.seed(0)
#     np.random.seed(0)
#     hidden_size = 512
#     fragments_path = "/root/projets/FFREED/zinc_crem.json"
#     loader = Dataloader(
#         data_root="/root/data/graphs",
#         buffer_size=1,  # Number of tasks per buffer
#         steps_per_buffer=512,  # How many batches to use before switching buffers
#         num_steps=512,  # Sequence length
#         batch_size=4,
#     )

#     batch = loader.get_batch().to("cuda")

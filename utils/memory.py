import numpy as np
import torch
import hnswlib

class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def mine_nearest_neighbors_hnsw(self, topk, calculate_accuracy=True):
        #"""use HNSW for neighbor searching
        features = self.features.cpu().numpy()
        print(f"HNSW Features shape: n={features.shape[0]}, dim={features.shape[1]}")
        n, dim = features.shape[0], features.shape[1]
        
        # initialize HNSW index
        index = hnswlib.Index(space='cosine', dim=dim)
        
        # Configure index parameters
        index.init_index(
            max_elements=n,
            ef_construction=200,  # Search depth at build time
            M=16,  # The maximum number of connections per node
            random_seed=100
        )
        
        # add data to index
        index.add_items(
            features, 
            num_threads=4  # use multi-threading to accelerate building
        )
        
        # set search parameters
        index.set_ef(50)  # candidate set size at search time
        
        try:
            indices, distances = index.knn_query(features, k=topk+1)
            
            if calculate_accuracy:
                targets = self.targets.cpu().numpy()
                # only consider real neighbors
                neighbor_targets = np.take(targets, indices[:,1:], axis=0)
                anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
                accuracy = np.mean(neighbor_targets == anchor_targets)
                return indices, accuracy
            else:
                return indices
            
        except Exception as e:
            print(f"HNSW searching error: {str(e)}")
            raise e

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()
    device = next(model.parameters()).device

    for i, batch in enumerate(loader):
        #batch = tuple(t.cuda(non_blocking=True) for t in batch)
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
        feature = model(X, output_hidden_states=True)["hidden_states"]

        memory_bank.update(feature, label_ids)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))
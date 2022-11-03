#%%
import torch
import torch.nn.functional as F
import numpy as np

def KNN(emb, cls, batch_size, Ks=[1, 10, 50, 100]):
    """Apply KNN for different K and return the maximum acc"""
    preds = []
    mask = torch.eye(batch_size).bool().to(emb.device)
    mask = F.pad(mask, (0, len(emb) - batch_size))
    for batch_x in torch.split(emb, batch_size):
        dist = torch.norm(
            batch_x.unsqueeze(1) - emb.unsqueeze(0), dim=2, p="fro")
        now_batch_size = len(batch_x)
        mask = mask[:now_batch_size]
        dist = torch.masked_fill(dist, mask, float('inf'))
        # update mask
        mask = F.pad(mask[:, :-now_batch_size], (now_batch_size, 0))
        pred = []
        for K in Ks:
            knn = dist.topk(K, dim=1, largest=False).indices
            knn = cls[knn].cpu()
            pred.append(torch.mode(knn).values)
        pred = torch.stack(pred, dim=0)
        preds.append(pred)
    preds = torch.cat(preds, dim=1)
    accs = [(pred == cls.cpu()).float().mean().item() for pred in preds]
    return max(accs)

#%%
embeddings_np = np.load('C:/CCBDA/HW2/test_original_rn34.npy')
embeddings = torch.from_numpy(embeddings_np)

N = 125
classes = torch.cat([
    torch.zeros(N),
    torch.ones(N),
    torch.full((N,),2),
    torch.full((N,),3)
], dim=0)
acc = KNN(embeddings, classes, batch_size=16)
print("Accuracy: %.5f" % acc)
# %%

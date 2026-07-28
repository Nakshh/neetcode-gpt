import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        L = 0.0
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            starts = torch.randint(len(data) - context_length, (batch_size, ))[:, None]
            offsets = torch.arange(context_length)

            X = data[starts + offsets]
            Y = data[starts + offsets + 1]

            optimizer.zero_grad()

            logits = model(X)
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.reshape(-1, C),
                Y.reshape(-1)
            )

            loss.backward()
            optimizer.step()
            
            L = loss.item()
        
        return round(L, 4)




import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from concept_audit.models.native_cbm import NativeCBM
from concept_audit.readouts import IdentityReadout


EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3

FEATURES = "features_cub/embeddings.npy"
LABELS = "features_cub/labels.npy"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.from_numpy(np.load(FEATURES)).float()
    y = torch.from_numpy(np.load(LABELS)).long()

    dataset = TensorDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    readout = IdentityReadout(latent_dim=112)

    model = NativeCBM(
        feature_dim=X.shape[1],
        num_classes=200,
        readout=readout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            _, logits = model(features)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"loss={total_loss / len(loader):.4f} "
            f"accuracy={correct / total:.4f}"
        )

    torch.save(model.state_dict(), "cub_cbm.pt")


if __name__ == "__main__":
    main()
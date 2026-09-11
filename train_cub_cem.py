import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from concept_audit.models.native_cem import NativeCEM


EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3

FEATURES = "features_cub/embeddings.npy"
LABELS = "features_cub/labels.npy"

NUM_CONCEPTS = 112
BLOCK_SIZE = 4
NUM_CLASSES = 200


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --------------------------------------------------
    # Load frozen DINO features
    # --------------------------------------------------
    X = torch.from_numpy(
        np.load(FEATURES)
    ).float()

    y = torch.from_numpy(
        np.load(LABELS)
    ).long()

    print("Features:", X.shape)
    print("Labels:", y.shape)

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # --------------------------------------------------
    # CEM
    # --------------------------------------------------
    model = NativeCEM(
        feature_dim=X.shape[1],
        num_classes=NUM_CLASSES,
        num_concepts=NUM_CONCEPTS,
        block_size=BLOCK_SIZE,
    ).to(device)

    print(
        "Latent dimension:",
        model.readout.latent_dim,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

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

            concepts, logits = model(features)
            
            """
            OR TO USE A WEIGHTED SUM OF THE LOSSES.
            concept_loss = nn.BCEWithLogitsLoss()(
                concepts,
                attributes
            )

            class_loss = nn.CrossEntropyLoss()(
                logits,
                labels
            )

            loss = class_loss + lambda_concept * concept_loss
            """

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        accuracy = correct / total
        avg_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"loss={avg_loss:.4f} "
            f"accuracy={accuracy:.4f}"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    torch.save(
        model.state_dict(),
        "cub_cem.pt",
    )

    print("Saved: cub_cem.pt")


if __name__ == "__main__":
    main()
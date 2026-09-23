import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

import time
from datasets import load_dataset, load_from_disk

# print("[1/3] Téléchargement + chargement du dataset (5.92 GB, ça va prendre un moment)...")
# t0 = time.time()
# dataset = load_dataset("randall-lab/shapes3d", split="train", trust_remote_code=True)
# print(f"✓ Terminé en {time.time() - t0:.1f}s")

# print("[2/3] Sauvegarde sur disque...")
# t0 = time.time()
# dataset.save_to_disk("ignore/data/shapes3d_local")
# print(f"✓ Terminé en {time.time() - t0:.1f}s")

print("[3/3] Rechargement depuis le disque...")
t0 = time.time()
dataset = load_from_disk("ignore/data/shapes3d_local")
print(f"✓ Terminé en {time.time() - t0:.1f}s")


example = dataset[5]
image = example["image"]
label = example["label"]          # Value labels: [floor_hue, wall_hue, object_hue, scale, shape, orientation]
print(f"Label (factor values): {label}")
label_index = example["label_index"]  # Index labels: [floor_idx, wall_idx, object_idx, scale_idx, shape_idx, orientation_idx]
print(f"Label (factor indices): {label_index}")

# Label Value
floor_value = example["floor"]      # 0-1
wall_value = example["wall"]        # 0-1
object_value = example["object"]    # 0-1
scale_value = example["scale"]      # 0.75-1.25
shape_value = example["shape"]      # 0,1,2,3
orientation_value = example["orientation"]  # -30 - 30

# Label index
floor_idx = example["floor_idx"]      # 0-9
wall_idx = example["wall_idx"]        # 0-9
object_idx = example["object_idx"]    # 0-9
scale_idx = example["scale_idx"]      # 0-7
shape_idx = example["shape_idx"]      # 0-3
orientation_idx = example["orientation_idx"]  # 0-14

image.show()  # Display the image

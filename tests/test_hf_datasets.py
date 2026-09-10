import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from datasets import Dataset, Features, Image as HFImage, ClassLabel, Sequence, Value

from dataloaders import load_dataset
from dataloaders.huggingface import HFImageDataset, DATASETS
from concept_audit.data.extraction import extract_cache, main
from concept_audit.data import load_cache
from models.backbones import BackboneInfo


def fixture(image_column="image", label_column="label", annotated=False):
    features = {image_column: HFImage(), label_column: ClassLabel(names=["a", "b"])}
    values = {image_column: [Image.new("RGB", (4, 4), (i, 0, 0)) for i in range(4)],
              label_column: [1, 0, 1, 0]}
    if annotated:
        features["attributes"] = Sequence(Value("float32"))
        values["attributes"] = [[1., 0.], [0., 1.], [1., 1.], [0., 0.]]
    return Dataset.from_dict(values, features=Features(features))


class FakeBackbone:
    info = BackboneInfo("test", "fake", 2)

    def encode_pil(self, images):
        return torch.tensor([[float(image.getpixel((0, 0))[0]), 1.] for image in images])


class HFTests(unittest.TestCase):
    def test_local_cifar_uses_existing_files_and_preserves_slice(self):
        class LocalCIFAR:
            classes = ["a", "b"]
            targets = [1, 0, 1, 0]
            def __len__(self):
                return 4
            def __getitem__(self, index):
                return Image.new("RGB", (4, 4), (index, 0, 0)), self.targets[index]
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/"cifar-10-batches-py").mkdir()
            with patch("torchvision.datasets.CIFAR10", return_value=LocalCIFAR()) as local, patch("datasets.load_dataset") as hub:
                dataset = load_dataset("cifar10", split="test[1:3]", data_dir=temp)
                local.assert_called_once_with(root=str(Path(temp).resolve()), train=False, download=False)
                hub.assert_not_called()
                self.assertEqual(len(dataset), 2)
                self.assertEqual(dataset[0][0].getpixel((0, 0)), (1, 0, 0))
                self.assertEqual([dataset[i][2] for i in range(2)], [0, 1])
                self.assertEqual(dataset.metadata["source_row_start"], 1)
                self.assertIsNone(dataset.metadata["repository"])

    def test_alias_dispatch_and_split_revision(self):
        for alias, (repo, image, label) in DATASETS.items():
            with self.subTest(dataset=alias), patch("datasets.load_dataset", return_value=fixture(image, label)) as call:
                dataset = load_dataset(alias, split="test[:4]", cache_dir="data/hf", revision="abc123")
                call.assert_called_once_with(repo, split="test[:4]", cache_dir="data/hf", revision="abc123")
                self.assertEqual(dataset[0][2], 1)
                self.assertIsNone(dataset[0][1])
                self.assertEqual(dataset[0][0].mode, "RGB")
                self.assertEqual(dataset.classes, ["a", "b"])

    def test_concept_column_and_cache_roundtrip(self):
        dataset = HFImageDataset(fixture(annotated=True), image_column="image", label_column="label", concept_column="attributes")
        with tempfile.TemporaryDirectory() as temp:
            extract_cache(dataset, FakeBackbone(), temp, batch_size=3)
            z, g, y = load_cache(temp)
            self.assertEqual(z[:, 0].tolist(), [0, 1, 2, 3])
            self.assertEqual(g.tolist(), [[1, 0], [0, 1], [1, 1], [0, 0]])
            self.assertEqual(y.tolist(), [1, 0, 1, 0])
            meta = json.loads((Path(temp)/"meta.json").read_text())
            self.assertEqual(meta["dataset"]["fingerprint"], dataset.dataset._fingerprint)

    def test_external_annotations_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"concepts.npy"
            np.save(path, np.eye(4, dtype=np.float32))
            dataset = HFImageDataset(fixture(), image_column="image", label_column="label", concepts_file=path)
            np.testing.assert_array_equal(dataset[2][1], [0, 0, 1, 0])
            self.assertIn("concepts_sha256", dataset.metadata)
            for bad in (np.ones((3, 2)), np.full((4, 2), np.nan), np.full((4, 2), 2.)):
                np.save(path, bad)
                with self.assertRaises(ValueError):
                    HFImageDataset(fixture(), image_column="image", label_column="label", concepts_file=path)

    def test_missing_concepts_and_unannotated_features(self):
        with patch("datasets.load_dataset") as call:
            with self.assertRaisesRegex(ValueError, "requires"):
                load_dataset("cifar10", require_concepts=True)
            call.assert_not_called()
        dataset = HFImageDataset(fixture(), image_column="image", label_column="label")
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "annotations"):
                extract_cache(dataset, FakeBackbone(), temp)
            extract_cache(dataset, FakeBackbone(), temp, require_concepts=False)
            self.assertFalse((Path(temp)/"concepts.npy").exists())
            self.assertEqual(np.load(Path(temp)/"labels.npy").tolist(), [1, 0, 1, 0])
            np.save(Path(temp)/"concepts.npy", np.ones((4, 2)))
            with self.assertRaisesRegex(ValueError, "old concepts"):
                extract_cache(dataset, FakeBackbone(), temp, require_concepts=False)

    def test_real_hf_parquet_loader_and_cli(self):
        # Real datasets.load_dataset call against a local Parquet fixture: no network.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/"fixture"
            source.mkdir()
            fixture(annotated=True).to_parquet(source/"train.parquet")
            out = root/"features"
            args = ["extract", "--dataset", str(source), "--split", "train",
                    "--cache-dir", str(root/"hf-cache"), "--concept-column", "attributes",
                    "--require-concepts", "--out-dir", str(out)]
            with patch("sys.argv", args), patch("concept_audit.backbones.build_backbone", return_value=FakeBackbone()):
                main()
            self.assertEqual(load_cache(out)[0].shape, (4, 2))

    def test_bad_column_and_ragged_concepts(self):
        with self.assertRaisesRegex(ValueError, "Missing HF columns"):
            HFImageDataset(fixture(), image_column="img", label_column="label")
        dataset = fixture(annotated=True).remove_columns("attributes").add_column("attributes", [[1.,0.], [1.], [0.,0.], [1.,1.]])
        wrapped = HFImageDataset(dataset, image_column="image", label_column="label", concept_column="attributes")
        with self.assertRaisesRegex(ValueError, "length"):
            wrapped[1]


if __name__ == "__main__":
    unittest.main()

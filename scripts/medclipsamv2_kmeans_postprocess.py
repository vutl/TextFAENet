from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


def keep_largest_components(mask: np.ndarray, count: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1 or count <= 0:
        return np.zeros_like(mask, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    selected = np.argsort(areas)[::-1][:count] + 1
    return (np.isin(labels, selected).astype(np.uint8) * 255)


def process_one(saliency_path: Path, output_path: Path, num_contours: int) -> None:
    saliency = cv2.imread(str(saliency_path), cv2.IMREAD_GRAYSCALE)
    if saliency is None:
        raise RuntimeError(f"Cannot read saliency map: {saliency_path}")
    height, width = saliency.shape
    resized = cv2.resize(
        saliency.astype(np.float32) / 255.0,
        (256, 256),
        interpolation=cv2.INTER_NEAREST,
    )
    flat = resized.reshape(-1, 1)
    if float(flat.max()) == float(flat.min()):
        binary = np.zeros((256, 256), dtype=np.uint8)
    else:
        kmeans = KMeans(n_clusters=2, random_state=10, n_init=10)
        labels = kmeans.fit_predict(flat).reshape(256, 256)
        background = int(np.argmin(kmeans.cluster_centers_.reshape(-1)))
        binary = np.where(labels == background, 0, 255).astype(np.uint8)
    binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
    filtered = keep_largest_components(binary, num_contours)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), filtered):
        raise RuntimeError(f"Cannot write coarse mask: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        "MedCLIP-SAMv2 k-means postprocessing without the unused pydensecrf dependency."
    )
    parser.add_argument("--sal-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--num-contours", type=int, default=1)
    args = parser.parse_args()

    paths = sorted(path for path in args.sal_path.iterdir() if path.is_file())
    if not paths:
        raise RuntimeError(f"No saliency maps found in {args.sal_path}")
    processed = 0
    for path in tqdm(paths, desc="k-means postprocess"):
        output_path = args.output_path / path.name
        if output_path.is_file():
            continue
        process_one(path, output_path, args.num_contours)
        processed += 1
    print(
        f"Processed {processed}; available {len(paths)} coarse masks in {args.output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()

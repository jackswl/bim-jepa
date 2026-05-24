import argparse
from typing import Tuple
import pandas as pd
from sklearn.manifold import TSNE
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from bimjepa.datasets.ifcnet_datamodule import IFCNetCoreDataModule

from bimjepa.models import BimJepa
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns


def parse_args() -> None:
    parser = argparse.ArgumentParser(
        description="BimJepa Embedding Cluster Visualization"
    )
    parser.add_argument("--finetuned_ckpt_path", "-ch", type=str, required=True)
    return parser.parse_args()


def xy(
    dataloader: DataLoader, model: pl.LightningModule
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_list = []
    label_list = []

    with torch.no_grad():
        for points, label in tqdm(dataloader):
            points: torch.Tensor = points.cuda()
            label: torch.Tensor = label.cuda()
            points = model.val_transformations(points)
            embeddings, centers = model.tokenizer(points.float())
            pos = model.positional_encoding(centers)
            x = model.student(embeddings, pos).last_hidden_state
            x = torch.cat([x.max(dim=1).values, x.mean(dim=1)], dim=-1)
            x_list.append(x.cpu())

            label_list.append(label.cpu())

    x = torch.cat(x_list, dim=0)  # (N, 768)
    y = torch.cat(label_list, dim=0)  # (N,)
    return x, y


def main():
    args = parse_args()
    li_model = BimJepa.load_from_checkpoint(args.finetuned_ckpt_path, strict=False)

    li_model = li_model.cuda()
    li_model.eval()

    data_module = IFCNetCoreDataModule(
        data_dir="data/processed_IFCNetCore_pointclouds_4096",
        batch_size=32
    )

    data_module.setup()

    x, y = xy(data_module.val_dataloader(), li_model)

    tsne = TSNE(n_components=2, verbose=1, perplexity=50, max_iter=1000)
    embeddings_tsne = tsne.fit_transform(x.detach().numpy())

    class_to_idx = data_module.class_to_idx
    sorted_items = sorted(class_to_idx.items(), key=lambda item: item[1])
    label_names = [item[0] for item in sorted_items]

    mapped_labels = [label_names[label] for label in y.numpy()]

    df = pd.DataFrame({
        'Principal Component 1': embeddings_tsne[:, 0],
        'Principal Component 2': embeddings_tsne[:, 1],
        'Label': mapped_labels
    })

    plt.figure(figsize=(14, 8))
    scatter_plot = sns.scatterplot(
        x='Principal Component 1',
        y='Principal Component 2',
        hue='Label',
        data=df,
        palette=sns.color_palette("hsv", 20)
    )

    scatter_plot.set(xlabel=None, ylabel=None)
    scatter_plot.set_xticks([])
    scatter_plot.set_yticks([])

    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='x-large')

    plt.tight_layout()
    plt.savefig("tsne_embedding_visualization.png", dpi=600, bbox_inches='tight')


if __name__ == "__main__":
    main()

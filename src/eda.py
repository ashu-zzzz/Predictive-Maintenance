from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from .config import NUMERICAL_FEATURES, TARGET

def generate_eda_plots(frame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    frame[TARGET].value_counts().sort_index().plot.bar(color=["#2a9d8f", "#e76f51"])
    plt.title("Machine failure distribution"); plt.xlabel("Failure label"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(output_dir / "failure_distribution.png", dpi=150); plt.close()
    frame[NUMERICAL_FEATURES + [TARGET]].corr().pipe(
        sns.heatmap, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Numeric feature correlation"); plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150); plt.close()
    frame[NUMERICAL_FEATURES].hist(figsize=(12, 8), bins=30, color="#457b9d")
    plt.suptitle("Feature distributions"); plt.tight_layout()
    plt.savefig(output_dir / "feature_distributions.png", dpi=150); plt.close()

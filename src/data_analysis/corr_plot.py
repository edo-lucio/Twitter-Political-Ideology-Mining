import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def correlation_heatmap_grid(dfs, titles):

    # Customize the heatmaps using seaborn
    for i in range(0, 2):
        corr_matrix = dfs[i].corr()
        masks = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix, mask=masks, cmap="coolwarm", annot=True, fmt=".2f")

        plt.suptitle(f"{titles[i]} Users Heatmap", fontsize=10)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    folder_path = "data\\scores\\frame-axis\\aggregated\\{}"

    paths = ["right-regular-aggregated.csv",
             "left-regular-aggregated.csv"]

    df_list = [pd.read_csv(folder_path.format(
        path), encoding="utf-8", on_bad_lines="skip") for path in paths]

    df_scores = [df.iloc[:, 1:11] for df in df_list]

    correlation_heatmap_grid(df_scores, ["Conservatives", "Labor"])

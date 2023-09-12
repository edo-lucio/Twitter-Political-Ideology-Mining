import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import plotly.express as px


def dim_reduction(folder_path, path_1, path_2, groups):
    # Load the data into a pandas DataFrame
    df_1 = pd.read_csv(folder_path.format(path_1),
                       encoding="utf-8", on_bad_lines="skip")
    df_2 = pd.read_csv(folder_path.format(path_2),
                       encoding="utf-8", on_bad_lines="skip")

    X = pd.concat([df_1, df_2], ignore_index=True)
    X = X.drop("tweet_count_range", axis=1)

    # Create a PCA object with the desired number of components
    pca = PCA(n_components=2)

    # Fit the PCA object to the data and transform the data
    X_pca = pca.fit_transform(X)

    # Create a new DataFrame with the transformed data
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    target = [0] * len(df_1.index) + [1] * len(df_2.index)
    df_pca['group'] = target

    print(df_pca['group'])

    # Remove data points below the PC1 threshold
    df_pca_filtered = df_pca[df_pca['PC1'] >= 0]

    # Print the explained variance ratio of the principal components
    print(pca.explained_variance_ratio_)

    colors = {0: "red", 1: "blue"}
    # Create a scatter plot of the transformed data using Plotly
    fig = px.scatter(df_pca_filtered, x='PC1', y='PC2',
                     color='group', color_discrete_map=colors)

    # Add axis labels and title
    fig.update_layout(
        xaxis_title='PC1',
        yaxis_title='PC2',
        title='PCA Biplot'
    )

    # Show the plot
    fig.show()

    return df_pca_filtered


if __name__ == "__main__":
    folder_path = "data\\scores\\frame-axis\\aggregated\\{}"
    df_1 = "right-regular-aggregated.csv"
    df_2 = "left-regular-aggregated.csv"

    dim_reduction(folder_path, df_1, df_2, ["pro-brexit", "anti-brexit"])

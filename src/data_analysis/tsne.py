import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.utils import resample

import plotly.express as px

from scipy.spatial import ConvexHull


def assign_group_labels(data_frames, labels):
    result = []

    for i, df in enumerate(data_frames):
        df["group"] = labels[i]
        result.append(df)

    return result


def perform_pca(data_frames):
    X = pd.concat([df for df in data_frames], ignore_index=True)
    X_ = X.iloc[:, :-1]

    # Create a PCA object with the desired number of components
    pca = PCA(n_components=2)

    # Fit the PCA object to the data and transform the data
    X_pca = pca.fit_transform(X_)

    # Create a new DataFrame with the transformed data
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca["group"] = X["group"]

    # Remove data points below the PC1 threshold
    df_pca_filtered = df_pca[df_pca['PC1'] != None]

    # Calculate the Silhouette Score
    silhouette_avg = silhouette_score(
        df_pca_filtered[['PC1', 'PC2']], df_pca_filtered["group"])
    print(f"Silhouette Score: {silhouette_avg}")

    # Print the explained variance ratio of the principal components
    print(pca.explained_variance_ratio_.cumsum())

    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    # Create a scatter plot of the transformed data using Plotly
    fig = px.scatter(df_pca_filtered, x='PC1', y='PC2', color=df_pca_filtered['group'],
                     labels=labels, opacity=0.8, size_max=1, color_discrete_sequence=["red", "lightblue", "green", "blue"])  # Add 'labels' parameter for the legend title

    # Add axis labels and title
    fig.update_layout(
        xaxis_title='PC1',
        yaxis_title='PC2',
        title='PCA Biplot'
    )

    fig.update_traces(marker=dict(size=6,
                                  line=dict(width=0.5,
                                            color='DarkSlateGrey'), ),
                      selector=dict(mode='markers'))

    # Show the plot
    fig.show()

    return df_pca_filtered


def convex_hull(data_frames):
    X = pd.concat([df for df in data_frames], ignore_index=True)
    X_ = X.iloc[:, :-1]

    # Create a PCA object with the desired number of components
    pca = PCA(n_components=2)

    # Fit the PCA object to the data and transform the data
    X_pca = pca.fit_transform(X_)

    # Create a new DataFrame with the transformed data
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca["group"] = X["group"]

    # Remove data points below the PC1 threshold (you may adjust this threshold if needed)
    df_pca_filtered = df_pca[df_pca['PC1'] != None]

    # Print the explained variance ratio of the principal components
    print(pca.explained_variance_ratio_)

    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    # Create a scatter plot of the transformed data using Matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the data points
    for group, color in zip(df_pca_filtered['group'].unique(), ['red', 'blue', 'green', 'orange']):
        group_data = df_pca_filtered[df_pca_filtered['group'] == group]
        ax.scatter(group_data['PC1'], group_data['PC2'],
                   label=f"Group {group}", color=color, alpha=0.7)

        # Calculate convex hull and plot it
        hull = ConvexHull(group_data[['PC1', 'PC2']])
        for simplex in hull.simplices:
            ax.plot(group_data['PC1'].values[simplex],
                    group_data['PC2'].values[simplex], color=color)

    # Add axis labels and title
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA Biplot')
    ax.legend()

    # Show the plot
    plt.show()

    return df_pca_filtered


def perform_tsne_with_undersampling(data_frames, title):
    # Combine all data frames and labels
    X = pd.concat([df for df in data_frames], ignore_index=True)
    X_ = X.iloc[:, :-1]
    y = X['group']

    # Identify the minority and majority classes
    minority_class = y.value_counts().idxmin()
    majority_class = y.value_counts().idxmax()

    # Undersample the majority class to match the minority class size
    minority_samples = X[y == minority_class]
    majority_samples = resample(X[y == majority_class], n_samples=len(
        minority_samples), random_state=42)
    X_undersampled = pd.concat([minority_samples, majority_samples])

    print(X_undersampled.columns)

    silhouette_avg = silhouette_score(
        X_undersampled.iloc[:, :-1], X_undersampled["group"])
    print(f"Silhouette Score: {silhouette_avg}")

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_undersampled.iloc[:, :-1])

    # Create a scatter plot of the t-SNE results
    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1],
                     color=X_undersampled['group'], opacity=0.8, size_max=1, color_discrete_sequence=["#DC3912", "#3283FE"])

    # Update the layout with title and axis labels
    fig.update_layout(
        height=600, width=1200,
        title_text=title,
        legend={"title": {"text": "Activism"}}
    )
    fig.show()


def main():
    folder_path = "data\\scores\\frame-axis\\aggregated\\{}"
    paths = ["internal-regular#climatechange-aggregated.csv",
             "external-regular#climatechange-aggregated.csv", ]

    labels = ["Insider", "Outsider"]
    data_frames = [pd.read_csv(folder_path.format(path)) for path in paths]
    data_frames = assign_group_labels(
        data_frames, labels)

    bias_frames = [df.iloc[:, list(range(1, 6)) + [-1]] for df in data_frames]
    int_frames = [df.iloc[:, list(range(6, 11)) + [-1]] for df in data_frames]

    print(list(data_frames[0].columns[1:10]))

    data_frames = [df.iloc[:, list(range(1, 10)) + [-1]] for df in data_frames]

    perform_tsne_with_undersampling(
        int_frames, f"t-SNE representation of {labels[0]} and {labels[1]} Intensity scores")


if __name__ == "__main__":
    main()

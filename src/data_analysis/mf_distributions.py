import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy import stats
from numpy.random import seed
from numpy.random import randn
from numpy.random import normal

from numpy import mean, sqrt, var

from scipy.stats import ttest_ind

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import pandas as pd

from plotnine import *

import pandas as pd

from remove_intersection import remove_intersections


def cohens_d(v1, v2):
    var1 = var(v1)
    var2 = var(v2)

    n1 = len(v1) - 1
    n2 = len(v2) - 1

    pooled_std = sqrt(((n1 * var1) + (n2 * var2)) / (n1 + n2))

    mean_v1 = mean(v1)
    mean_v2 = mean(v2)

    print("M", round(mean_v1, 4), round(mean_v2, 4))
    print("SD", round(sqrt(var1), 4), round(sqrt(var2), 4))

    d = round(abs((mean_v1 - mean_v2) / pooled_std), 2)
    return d


def boxplot(df, labels, description=''):
    df_A = df[df["group"] == labels[0]]
    df_B = df[df["group"] == labels[1]]

    columns = list(df.columns)
    columns.pop()

    colors = ["#fb0d0d", "#511cfb"]
    # colors = ["#ff7f0e", "#3366cc"]

    fig = make_subplots(rows=1, cols=len(columns),
                        subplot_titles=columns, horizontal_spacing=0.1, vertical_spacing=0.1)

    for j, column in enumerate(columns):
        # Create a boxplot for this column
        data = [df_A[column], df_B[column]]

        fig.add_trace(go.Box(
            y=df_A[column], name=labels[0], marker_color=colors[0]), row=1, col=j+1)
        fig.add_trace(go.Box(
            y=df_B[column], name=labels[1], marker_color=colors[1]), row=1, col=j+1)

        print("\n", column)
        c_d = cohens_d(df_A[column], df_B[column])

        t_stat, p_value = ttest_ind(df_A[column], df_B[column])
        print("stat", t_stat)

        fig.update_xaxes(
            title_text=f"cohen's d={c_d}, p-value={round(p_value, 4)}", row=1, col=j+1)

        # Only show the legend for the first subplot
        if j == 0:
            fig.update_traces(showlegend=True, row=1, col=j+1)
        else:
            fig.update_traces(showlegend=False, row=1, col=j+1)

    # Update the layout with title and axis labels
    fig.update_layout(
        height=600, width=1200,
        title_text=description,
        legend={"title": {"text": "Activism"}},
        legend_x=1,
        legend_y=1,
    )

    fig.show()


def main(path_A, path_B, folder_path):
    df_A = pd.read_csv(folder_path.format(path_A),
                       encoding="utf-8", on_bad_lines='skip')
    df_B = pd.read_csv(folder_path.format(path_B),
                       encoding="utf-8", on_bad_lines='skip')

    print(df_A.size, df_B.size)

    processed_df = remove_intersections(df_A, df_B)
    df_A = processed_df[0]
    df_B = processed_df[1]

    labels = ["right", "left"]

    df_A["group"] = labels[0]
    df_B["group"] = labels[1]
    df = pd.concat([df_A, df_B])

    df_BIAS = df.iloc[:, list(range(1, 6)) + [-1]]  # BIAS
    df_INTENSITY = df.iloc[:, list(range(6, 11)) + [-1]]  # INTENSITY

    n_A = len(df_A)
    n_B = len(df_B)

    boxplot(
        df_BIAS, labels, f"MF Bias Distributions: \n {labels[0]} (n={n_A}) \n {labels[1]} (n={n_B})")
    boxplot(df_INTENSITY, labels,
            f"MF Intensity Distributions: \n {labels[0]} (n={n_A}) \n {labels[1]} (n={n_B})")


if __name__ == "__main__":
    folder_path = "data\\scores\\frame-axis\\aggregated\\{}"

    account_A = "right-regular-aggregated.csv"
    account_B = "left-regular-aggregated.csv"

    main(account_A, account_B, folder_path)

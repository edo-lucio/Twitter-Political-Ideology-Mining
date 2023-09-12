import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from remove_intersection import remove_intersections


def train_knn_model(data):
    quantiles = pd.qcut(
        data["tweet_count"], 4, duplicates='drop', labels=[0, 1, 3])
    data.insert(len(list(data.columns))-2, "frequency_range", quantiles)

    # Create design matrix X and response variable y
    X = data.iloc[:, :-3]
    y = data.iloc[:, -1]

    # split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=data["frequency_range"])

    X_train.info()
    y_train.info()

    # create logistic regression model
    knn = KNeighborsClassifier(n_neighbors=30)

    # Train the classifier using the training data
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Calculate various performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    # Print the performance metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\nClassification Report:")
    print(classification_rep)
    print("\nConfusion Matrix:")
    print(confusion_mat)


def main(path_A, path_B, folder_path):
    df_A = pd.read_csv(folder_path.format(path_A),
                       encoding="utf-8", on_bad_lines='skip')
    df_B = pd.read_csv(folder_path.format(path_B),
                       encoding="utf-8", on_bad_lines='skip')

    processed_df = remove_intersections(df_A, df_B)
    df_A = processed_df[0]
    df_B = processed_df[1]

    df_A["category"] = 0
    df_B["category"] = 1

    df = pd.concat([df_A, df_B], ignore_index=True)
    df = df.drop("user_id", axis=1)
    df = df.iloc[:, np.r_[0:10, 20, 21, 22]]

    print(df["tweet_count_range"])

    train_knn_model(df)


if __name__ == "__main__":
    folder_path = "data\\scores\\frame-axis\\aggregated\\{}"

    account_A = "external-regular#climatechange-aggregated.csv"
    account_B = "internal-regular#climatechange-aggregated.csv"

    main(account_A, account_B, folder_path)

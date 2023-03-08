import pandas as pd
def remove_zero_rows(df):
    return
    df = df.loc[(df.iloc[:,1:]!=0).any(axis=1)]
    return df


if __name__ == "__main__":
    input_path = "data-collection\\data\\scores\\UKLabour-regular.csv"
    df = pd.read_csv(input_path, on_bad_lines='skip', encoding='utf-8')
    # df_1 = pd.read_csv("data-collection\\data\\tweets\\UKLabour-regular.csv", on_bad_lines='skip', encoding='utf-8')

    print(df.head())

    df = remove_zero_rows(df)
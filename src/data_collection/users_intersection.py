import pandas as pd


def get_intersection(paths, output_path):
    df_list = [pd.read_csv(
        path, encoding="utf-8", on_bad_lines="skip", header=None)[0] for path in paths]
    user_sets = [set(df) for df in df_list]
    intersection = set.intersection(*user_sets)

    intersection_df = pd.DataFrame(intersection)
    intersection_df.to_csv(path_or_buf=output_path, index=False, header=None)

    return intersection


if __name__ == "__main__":

    paths_list = [[
        "data\\users-list\\raw\\juststop_oil-followers-list.csv",
        "data\\users-list\\raw\\greenpeaceuk-followers-list.csv",

    ], [
        "data\\users-list\\raw\\juststop_oil-followers-list.csv",
        "data\\users-list\\raw\\xrebellionuk-followers-list.csv",
        "data\\users-list\\raw\\greenpeaceuk-followers-list.csv",
    ]]

    output_paths = ["data\\users-list\\raw\\test1-gpgp-climate-out-users.csv",
                    "data\\users-list\\raw\\test2-climate-out-users.csv"]

    for i in range(2):
        get_intersection(paths_list[i], output_paths[i])

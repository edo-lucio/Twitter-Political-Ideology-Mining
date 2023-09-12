import pandas as pd
from pathlib import Path


def remove_intersections(input_files):
    '''
    - takes a tuple of paths of the data to process
    - creates 2 csv files named after the twitter accounts as inputs
    - the ouput csv files contain the symmetric difference between the elements of the two input files  
    '''

    # read data
    data = [pd.read_csv(input_file, header=None)[0]
            for input_file in input_files]

    # create output file
    file_names = [file.split("\\")[-1].split(".")[0].split("-")[0]
                  for file in input_files]
    output_dir = Path(
        f"data\\users-list\\no-intersection-pairs\\{file_names[0] + '-' + file_names[1]}")
    output_dir.mkdir(exist_ok=True)
    output_files = [
        f"{output_dir}\\{file_name}-followers-list.csv" for file_name in file_names]

    # remove intersections
    processed_array1 = list(set(data[0]) - set(data[1]))
    processed_array2 = list(set(data[1]) - set(data[0]))

    dataframes = [pd.DataFrame(processed_array1),
                  pd.DataFrame(processed_array2)]

    dataframes[0].to_csv(path_or_buf=output_files[0], index=False, header=None)
    dataframes[1].to_csv(path_or_buf=output_files[1], index=False, header=None)


if __name__ == "__main__":
    # paths of the two csv to remove intersections from
    path_1 = "data\\users-list\\raw\\tradbritgroup-followers-list.csv"
    path_2 = "data\\users-list\\raw\\hopenothate-followers-list.csv"

    paths = (path_1, path_2)
    # will store the new sets into data-collection//data//users-list//no-intersection-pairs
    remove_intersections(paths)

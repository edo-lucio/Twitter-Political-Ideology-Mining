import pandas as pd

def remove_intersections(df_A, df_B):
    '''
    - takes 2 pandas df and remove intersections based on "user_id" column 
    '''

    # select ids column
    id_A = df_A["user_id"]
    id_B = df_B["user_id"]

    # remove intersections
    processed_id_A = list(set(id_A) - set(id_B))
    processed_id_B = list(set(id_B) - set(id_A))

    # subset
    df_A_processed = df_A[df_A['user_id'].isin(processed_id_A)]
    df_B_processed = df_B[df_B['user_id'].isin(processed_id_B)]

    return [df_A_processed, df_B_processed]


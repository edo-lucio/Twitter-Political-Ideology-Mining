from processing_tools.tools import remove_intersections

if __name__ == "__main__":
    # paths of the two csv to remove intersections from
    path_1 = "data-collection//data//users-list//UKLabour-followers-list.csv"
    path_2 = "data-collection//data//users-list//Conservatives-followers-list.csv"

    paths = (path_1, path_2)
    remove_intersections(paths) # will store the new sets into data-collection//data//users-list//no-intersection-pairs
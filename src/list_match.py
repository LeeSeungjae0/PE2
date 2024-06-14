def match_array_lengths(arr1, arr2):

    min_length = min(len(arr1), len(arr2))

    arr1 = arr1[:min_length]
    arr2 = arr2[:min_length]

    return arr1, arr2


def trim_lists_to_min_length(nested_lists):

    min_length = min(len(lst) for lst in nested_lists)

    trimmed_lists = [lst[:min_length] for lst in nested_lists]

    return trimmed_lists


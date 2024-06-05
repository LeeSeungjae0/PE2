def match_array_lengths(arr1, arr2):

    min_length = min(len(arr1), len(arr2))

    arr1 = arr1[:min_length]
    arr2 = arr2[:min_length]

    return arr1, arr2
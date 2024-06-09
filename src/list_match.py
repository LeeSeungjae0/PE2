def match_array_lengths(arr1, arr2):

    min_length = min(len(arr1), len(arr2))

    arr1 = arr1[:min_length]
    arr2 = arr2[:min_length]

    return arr1, arr2


def trim_lists_to_min_length(nested_lists):
    """
    주어진 리스트 내의 리스트들을 가장 짧은 리스트의 길이에 맞추어 자르는 함수.

    Args:
    nested_lists (list of lists): 길이가 서로 다른 리스트들이 포함된 리스트.

    Returns:
    list of lists: 가장 짧은 리스트의 길이에 맞추어 잘린 리스트들.
    """
    # 가장 짧은 리스트의 길이를 구합니다.
    min_length = min(len(lst) for lst in nested_lists)

    # 각 리스트를 가장 짧은 리스트의 길이에 맞추어 자릅니다.
    trimmed_lists = [lst[:min_length] for lst in nested_lists]

    return trimmed_lists


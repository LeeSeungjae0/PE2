# sj3 211, 260에 추가하면 됨


def balance_lists(list1, list2):
    # 길이가 긴 리스트에서 요소를 제거하여 길이를 맞춤
    while len(list1) != len(list2):
        if len(list1) > len(list2):
            list1.pop()
        else:
            list2.pop()
    return list1, list2


def match_array_lengths(arr1, arr2):
    # 두 배열의 길이를 비교
    min_length = min(len(arr1), len(arr2))

    # 두 배열의 길이를 작은 쪽에 맞추기
    arr1 = arr1[:min_length]
    arr2 = arr2[:min_length]

    return arr1, arr2


# 예시
#arr1 = [1, 2, 3]
#arr2 = [1, 2, 3, 4, 5, 6]

#print(adjust_array_length(arr1, 5, 0))  # 출력: [1, 2, 3, 0, 0]
#print(adjust_array_length(arr2, 3))  # 출력: [1, 2, 3]

# transmission_list, poly6 = balance_lists(transmission_list, poly6)
# wavelength_list, poly6 = balance_lists(wavelength_list, poly6)
# Plot the line connecting the two points
# flat_transmission = np.array(transmission_list) - np.array(poly6)


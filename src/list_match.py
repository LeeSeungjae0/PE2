# 211, 260에 추가하면 됨
def match_array_lengths(arr1, arr2):
    # 두 배열의 길이를 비교
    min_length = min(len(arr1), len(arr2))

    # 두 배열의 길이를 작은 쪽에 맞추기
    arr1 = arr1[:min_length]
    arr2 = arr2[:min_length]

    return arr1, arr2

# transmission_list, poly6 = balance_lists(transmission_list, poly6)
# wavelength_list, poly6 = balance_lists(wavelength_list, poly6)
# Plot the line connecting the two points
# flat_transmission = np.array(transmission_list) - np.array(poly6)
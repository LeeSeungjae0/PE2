# 211, 260에 추가하면 됨


def balance_lists(list1, list2):
    # 길이가 긴 리스트에서 요소를 제거하여 길이를 맞춤
    while len(list1) != len(list2):
        if len(list1) > len(list2):
            list1.pop()
        else:
            list2.pop()
    return list1, list2


# transmission_list, poly6 = balance_lists(transmission_list, poly6)
# wavelength_list, poly6 = balance_lists(wavelength_list, poly6)
# Plot the line connecting the two points
# flat_transmission = np.array(transmission_list) - np.array(poly6)


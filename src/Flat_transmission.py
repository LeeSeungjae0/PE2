from scipy.signal import find_peaks
import numpy as np
import list_match as lm
def process_flat_transmission(transmissions, polynomial):
    mid_transmission = (min(transmissions[0][1]) + max(transmissions[0][1])) / 2
    max_transmission_point, max_transmission_point2 = -50, -50
    for i, (dc_bias, wavelength_list, transmission_list) in enumerate(transmissions):
        flat_transmission = np.array(transmission_list) - np.array(polynomial(wavelength_list))
        if i != len(transmissions) - 1:
            peaks, _ = find_peaks(flat_transmission, distance=50)
            for peak_index in peaks:
                if min(wavelength_list) <= wavelength_list[peak_index] <= mid_transmission:
                    if flat_transmission[peak_index] > max_transmission_point:
                        max_transmission_point = flat_transmission[peak_index]
                        max_transmission_wavelength = wavelength_list[peak_index]
                if mid_transmission <= wavelength_list[peak_index] <= max(wavelength_list):
                    if flat_transmission[peak_index] > max_transmission_point2:
                        max_transmission_point2 = flat_transmission[peak_index]
                        max_transmission_wavelength2 = wavelength_list[peak_index]

    m = (max_transmission_point2 - max_transmission_point) / (max_transmission_wavelength2 - max_transmission_wavelength)
    b = max_transmission_point - m * max_transmission_wavelength
    peak_fit = m * np.array(transmissions[0][1]) + b

    for i, (dc_bias, wavelength_list, transmission_list) in enumerate(transmissions):
        poly_wavelength_array, peak_fit = lm.match_array_lengths(np.array(polynomial(wavelength_list)), peak_fit)
        transmission_array, peak_fit = lm.match_array_lengths(np.array(transmission_list), peak_fit)
        flat_meas_trans = transmission_array - poly_wavelength_array - (peak_fit if i != len(transmissions) - 1 else 0)
        wavelength_array, flat_meas_trans = lm.match_array_lengths( np.array(wavelength_list), flat_meas_trans)

    return wavelength_array, flat_meas_trans

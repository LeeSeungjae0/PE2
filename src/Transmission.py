from scipy.signal import find_peaks

def process_transmission_data(root):
    wavelength_sweeps = root.findall('.//WavelengthSweep')
    transmissions = []

    for wavelengthsweep in wavelength_sweeps:
        dc_bias = float(wavelengthsweep.get('DCBias'))
        wavelength_str = wavelengthsweep.find('.//L').text
        transmission_str = wavelengthsweep.find('.//IL').text
        wavelength_list = [float(w) for w in wavelength_str.split(',')]
        transmission_list = [float(t) for t in transmission_str.split(',')]
        transmissions.append((dc_bias, wavelength_list, transmission_list))

    return transmissions

def plot_transmission(ax, transmissions):
    ref_transmission_point = -50
    for i, (dc_bias, wavelength_list, transmission_list) in enumerate(transmissions):
        label = None if i == len(transmissions) - 1 else f'{dc_bias}V'
        ax.plot(wavelength_list, transmission_list, label=label)
        if i == len(transmissions) - 1:
            peaks, _ = find_peaks(transmission_list, distance=50)
            for peak_index in peaks:
                if transmission_list[peak_index] > ref_transmission_point:
                    ref_transmission_point = transmission_list[peak_index]

    return ref_transmission_point

def plot_reference(ax, reference_wave, reference_trans, r_squared_values):
    ax.plot(reference_wave, reference_trans, label='data')
    degrees = range(1, 7)
    max_transmission = np.max(reference_trans)
    min_transmission = np.min(reference_trans)
    y_pos = 0.5 * (max_transmission + min_transmission) - 0.3
    x_pos = reference_wave[0] + 0.5 * (reference_wave[-1] - reference_wave[0])

    for degree in degrees:
        coeffs, _, _, _ = np.linalg.lstsq(np.vander(reference_wave, degree + 1), reference_trans, rcond=None)
        polynomial = np.poly1d(coeffs)
        ax.plot(reference_wave, polynomial(reference_wave), label=f'{degree}th')
        mean_transmission = np.mean(reference_trans)
        total_variation = np.sum((reference_trans - mean_transmission) ** 2)
        residuals = np.sum((reference_trans - polynomial(reference_wave)) ** 2)
        r_squared = 1 - (residuals / total_variation)
        r_squared_values[degree] = r_squared
        ax.text(x_pos, y_pos, f'{degree}th R²: {r_squared:.4f}', fontsize=10, verticalalignment='center', horizontalalignment='center')
        y_pos -= 0.06 * (max_transmission - min_transmission)

    return polynomial

def plot_flat_transmission(ax, transmissions, polynomial):
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
        if i != len(transmissions) - 1:
            flat_meas_trans = np.array(transmission_list) - np.array(polynomial(wavelength_list)) - np.array(peak_fit)
        else:
            flat_meas_trans = np.array(transmission_list) - np.array(polynomial(wavelength_list))
        ax.plot(wavelength_list, flat_meas_trans, label=f'{dc_bias}V' if i != len(transmissions) - 1 else None)

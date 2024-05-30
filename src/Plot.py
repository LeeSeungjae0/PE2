import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.signal import find_peaks
import numpy as np
import list_match as lm


def plot_iv(ax, voltage_values, abs_current, final, R_squared, current_values):
    def log_formatter(x, pos):
        return "{:.0e}".format(x)

    y_text_position = 1e-5 if abs_current.max() >= 1e-3 and abs_current.min() <= 1e-10 else (
        1e-10 if abs_current.max() <= 2e-10 and abs_current.min() >= 6e-11 else min(abs_current) * 1.5)

    ax.scatter(voltage_values, abs_current, label='data')
    ax.plot(voltage_values, final, 'r-', label=f'fit (R²: {R_squared:.4f})')
    ax.set_xlim(-2, 1)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))
    ax.set_title('IV raw data & fitted data (log scale)')
    ax.set_ylabel('Absolute Current (A)')
    ax.set_xlabel('Voltage (V)')
    ax.grid(True)
    ax.legend(loc='upper left')
    ax.text(-1.9, y_text_position,
            f'  R²: {R_squared}\n-2V: {current_values[0]:.2e}\n-1V: {current_values[4]:.2e} \n 1V: {current_values[12]:.2e}',
            fontsize=10, horizontalalignment='left', verticalalignment='center')


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

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Transmission (dB)')
    ax.set_title('Transmission vs Wavelength')
    ax.grid(True)
    ax.legend(loc='lower right', bbox_to_anchor=(1.2, 0.47))

    return ref_transmission_point


def plot_reference(ax, reference_wave, reference_trans, r_squared_values):
    ax.plot(reference_wave, reference_trans, label='data')
    degrees = range(1, 7)
    max_transmission = np.max(reference_trans)
    min_transmission = np.min(reference_trans)
    y_pos = 0.5 * (max_transmission + min_transmission) - 0.3
    x_pos = reference_wave[0] + 0.5 * (reference_wave[-1] - reference_wave[0])
    best_r = 0
    for degree in degrees:
        coeffs, _, _, _ = np.linalg.lstsq(np.vander(reference_wave, degree + 1), reference_trans, rcond=None)
        polynomial1 = np.poly1d(coeffs)
        ax.plot(reference_wave, polynomial1(reference_wave), label=f'{degree}th')
        mean_transmission = np.mean(reference_trans)
        total_variation = np.sum((reference_trans - mean_transmission) ** 2)
        residuals = np.sum((reference_trans - polynomial1(reference_wave)) ** 2)
        r_squared = 1 - (residuals / total_variation)
        r_squared_values[degree] = r_squared
        ax.text(x_pos, y_pos, f'{degree}th R²: {r_squared:.4f}', fontsize=10, verticalalignment='center',
                horizontalalignment='center')
        y_pos -= 0.06 * (max_transmission - min_transmission)
        if best_r<=r_squared:
            best_r = r_squared
            polynomial = polynomial1
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

    m = (max_transmission_point2 - max_transmission_point) / (
                max_transmission_wavelength2 - max_transmission_wavelength)
    b = max_transmission_point - m * max_transmission_wavelength
    peak_fit = m * np.array(transmissions[0][1]) + b


    for i, (dc_bias, wavelength_list, transmission_list) in enumerate(transmissions):
        poly_wavelength_array, peak_fit = lm.match_array_lengths(np.array(polynomial(wavelength_list)), peak_fit)
        transmission_array, peak_fit = lm.match_array_lengths(np.array(transmission_list), peak_fit)
        flat_meas_trans = transmission_array - poly_wavelength_array - (peak_fit if i != len(transmissions) - 1 else 0)
        wavelength_array, flat_meas_trans = lm.match_array_lengths( np.array(wavelength_list), flat_meas_trans)
        ax.plot(wavelength_array, flat_meas_trans, label=f'{dc_bias}V' if i != len(transmissions) - 1 else None)


    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Flat Measured Transmission (dB)')
    ax.set_title('Flat Transmission spectra - as measured')
    ax.grid(True)
    ax.legend(loc='lower right', bbox_to_anchor=(1.2, 0.47))

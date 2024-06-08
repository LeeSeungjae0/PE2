import xml.etree.ElementTree as eT
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from decimal import Decimal

from Transmission import process_transmission_data
from Flat_transmission import process_flat_transmission
from Reference import extract_reference_data

def r_squared(y_true, y_pred):
    y_mean = np.mean(y_true)
    tss = np.sum((y_true - y_mean) ** 2)
    rss = np.sum((y_true - y_pred) ** 2)
    r2 = Decimal(1) - (Decimal(rss) / Decimal(tss))
    return r2

# ax1: 데이터 선형화, ax2: 근사, ax3: 근사화 모둠, ax4: delta neff
def linear(ax1, ax2, ax3, ax4, wavelength_array, flat_meas_trans):
    # 선형 전력 변환
    linear_minus_2 = 10 ** (flat_meas_trans[0] / 10) * 0.0005
    linear_minus_1_dot_5 = 10 ** (flat_meas_trans[1] / 10) * 0.0005
    linear_minus_1 = 10 ** (flat_meas_trans[2] / 10) * 0.0005
    linear_minus_0_dot_5 = 10 ** (flat_meas_trans[3] / 10) * 0.0005
    linear_0 = 10 ** (flat_meas_trans[4] / 10) * 0.0005
    linear_0_dot_5 = 10 ** (flat_meas_trans[5] / 10) * 0.0005

    ax1.scatter(wavelength_array[0], linear_minus_2, s=1, label='Measured -2V')
    ax1.scatter(wavelength_array[1], linear_minus_1_dot_5, s=1, label='Measured -1.5V')
    ax1.scatter(wavelength_array[2], linear_minus_1, s=1, label='Measured -1V')
    ax1.scatter(wavelength_array[3], linear_minus_0_dot_5, s=1, label='Measured -0.5V')
    ax1.scatter(wavelength_array[4], linear_0, s=1, label='Measured 0.0V')
    ax1.scatter(wavelength_array[5], linear_0_dot_5, s=1, label='Measured 0.5V')
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Intensity')
    ax1.set_title('Flat transmission spectra - linear')
    ax1.legend(loc='lower center', ncol=2, fontsize='small')

    def intensity(lamda, neff, delta, l, deltaL, I0):
        I = I0 * np.sin(((2 * np.pi / lamda) * deltaL * neff) / 2 + ((2 * np.pi / lamda) * l * delta / 2)) ** 2
        return I

    # 모델 생성
    model = Model(intensity)

    # 파라미터 설정
    params = model.make_params(neff=4.1, delta=0, l=500 * (10 ** -6), deltaL=40 * (10 ** -6), I0=0.0005)
    params['delta'].vary = False
    params['l'].vary = False
    params['deltaL'].vary = False
    params['I0'].vary = False
    params['neff'].vary = True

    x_nm = wavelength_array * (10 ** -9)
    # 피팅 수행
    result = model.fit(linear_0, params, lamda=x_nm[4])

    r2_2 = r_squared(linear_0, result.best_fit)
    neff_value = result.params['neff'].value

    # 결과 시각화
    ax2.scatter(wavelength_array[4], linear_0, s=5, label='Measured 0.0V')
    ax2.plot(wavelength_array[4], result.best_fit, label='Fitted 0.0V', color='red')

    ax2.set_xlabel('Wavelength [nm]')
    ax2.set_ylabel('Intensity')
    ax2.set_title('Flat transmission spectra - fitted')
    ax2.legend(loc='lower center', ncol=2, fontsize='small')

    delta_n = []
    # 모델 생성
    model2 = Model(intensity)

    # 파라미터 설정
    params = model2.make_params(neff=neff_value, delta=0, l=500 * (10 ** -6), deltaL=40 * (10 ** -6), I0=0.0005)
    params['delta'].vary = True
    params['l'].vary = False
    params['deltaL'].vary = False
    params['I0'].vary = False
    params['neff'].vary = False

    # 피팅 수행

    # -2V
    result2 = model2.fit(linear_minus_2, params, lamda=x_nm[0])
    r2_3 = r_squared(linear_minus_2, result2.best_fit)
    delta_neff_value = result2.params['delta'].value
    delta_n.append(delta_neff_value)

    # -1.5V
    result3 = model2.fit(linear_minus_1_dot_5, params, lamda=x_nm[1])
    r2_4 = r_squared(linear_minus_1_dot_5, result3.best_fit)
    delta_neff_value = result3.params['delta'].value
    delta_n.append(delta_neff_value)

    # -1V
    result4 = model2.fit(linear_minus_1, params, lamda=x_nm[2])
    r2_5 = r_squared(linear_minus_1, result4.best_fit)
    delta_neff_value = result4.params['delta'].value
    delta_n.append(delta_neff_value)

    # -0.5V
    result5 = model2.fit(linear_minus_0_dot_5, params, lamda=x_nm[3])
    r2_6 = r_squared(linear_minus_0_dot_5, result5.best_fit)
    delta_neff_value = result5.params['delta'].value
    delta_n.append(delta_neff_value)

    # 0V
    result6 = model2.fit(linear_0, params, lamda=x_nm[3])
    r2_7 = r_squared(linear_0, result6.best_fit)
    delta_neff_value = result6.params['delta'].value
    delta_n.append(delta_neff_value)

    # 0.5V
    result7 = model2.fit(linear_0_dot_5, params, lamda=x_nm[5])
    r2_8 = r_squared(linear_0_dot_5, result7.best_fit)
    delta_neff_value = result7.params['delta'].value
    delta_n.append(delta_neff_value)

    # 결과 시각화
    ax3.plot(wavelength_array[0], result2.best_fit, label='Fitted -2V')
    ax3.plot(wavelength_array[1], result3.best_fit, label='Fitted -1.5V')
    ax3.plot(wavelength_array[2], result4.best_fit, label='Fitted -1V')
    ax3.plot(wavelength_array[3], result5.best_fit, label='Fitted -0.5V')
    ax3.plot(wavelength_array[4], result6.best_fit, label='Fitted 0V')
    ax3.plot(wavelength_array[5], result7.best_fit, label='Fitted 0.5V')

    ax3.set_xlabel('Wavelength [nm]')
    ax3.set_ylabel('Intensity')
    ax3.set_title('Flat transmission spectra - fitted')
    ax3.legend(loc='lower center', ncol=2, fontsize='small')

    voltage = [-2, -1.5, -1, -0.5, 0, 0.5]

    ax4.plot(voltage, delta_n, label='delta', color='red')
    ax4.set_xlabel('Voltage')
    ax4.set_ylabel('delta_n')
    ax4.set_title('Delta n_eff')

    return

def plot_reference( reference_wave, reference_trans, r_squared_values):

    degrees = range(1, 7)
    max_transmission = np.max(reference_trans)
    min_transmission = np.min(reference_trans)
    y_pos = 0.5 * (max_transmission + min_transmission) - 0.3
    x_pos = reference_wave[0] + 0.5 * (reference_wave[-1] - reference_wave[0])
    best_r = 0
    for degree in degrees:
        coeffs, _, _, _ = np.linalg.lstsq(np.vander(reference_wave, degree + 1), reference_trans, rcond=None)
        polynomial1 = np.poly1d(coeffs)

        mean_transmission = np.mean(reference_trans)
        total_variation = np.sum((reference_trans - mean_transmission) ** 2)
        residuals = np.sum((reference_trans - polynomial1(reference_wave)) ** 2)
        r_squared = 1 - (residuals / total_variation)
        r_squared_values[degree] = r_squared

        y_pos -= 0.06 * (max_transmission - min_transmission)
        if best_r<=r_squared:
            best_r = r_squared
            polynomial = polynomial1
    return polynomial

tree = eT.parse(r'C:\Users\User\PycharmProjects\pythonProject1\PE2\dat\HY202103\D08\20190712_113254\HY202103_D08_(2,-1)_LION1_DCM_LMZC.xml')
root = tree.getroot()
reference_wave, reference_trans = extract_reference_data(root)
r_squared_values = {}
polynomial = plot_reference( reference_wave, reference_trans, r_squared_values)
transmissions = process_transmission_data(root)
wavelength_array, flat_meas_trans = process_flat_transmission(transmissions, polynomial)
#test
fig, axs = plt.subplots(1, 4)
linear(axs[0], axs[1], axs[2], axs[3], wavelength_array, flat_meas_trans)
plt.show()


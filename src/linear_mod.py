import xml.etree.ElementTree as eT
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from decimal import Decimal
from scipy.signal import find_peaks
from list_match import match_array_lengths
def linear(transmissions, axs):
    def r_squared(y_true, y_pred):
        y_mean = np.mean(y_true)
        tss = np.sum((y_true - y_mean) ** 2)
        rss = np.sum((y_true - y_pred) ** 2)
        r2 = Decimal(1) - (Decimal(rss) / Decimal(tss))
        return r2

    x_data_list = []
    y_data_list = []

    for dc_bias, wavelength_list, transmission_list in transmissions:
        wavelength_list, transmission_list = match_array_lengths(wavelength_list, transmission_list)
        x_data_list.append(wavelength_list)
        y_data_list.append(transmission_list)

    # Ensure x and y data lists are arrays
    x_data_list = [np.array(lst) for lst in x_data_list]
    y_data_list = [np.array(lst) for lst in y_data_list]

    x = np.array(x_data_list)
    y = np.array(y_data_list)

    # 데이터 정규화 다항식 고차수 근사 error해결
    x_mean = np.mean(x[6])
    x_std = np.std(x[6])
    x_normalized = (x[6] - x_mean) / x_std

    # 다항식 근사
    coefficients = np.polyfit(x_normalized, y[6], 7)
    poly = np.poly1d(coefficients)
    y_pred = poly(x_normalized)

    # find peak사용
    peaks = []
    for i in range(6):
        # data - 근사 ref 값
        y0 = y[i] - y_pred
        peaks_distance, _ = find_peaks(y0, distance=len(x[i]) / 2)
        x_ = x[i][peaks_distance]
        y0_ = y0[peaks_distance]
        coefficients = np.polyfit(x_, y0_, 1)
        poly = np.poly1d(coefficients)
        tra = poly(x[i])
        y0 = y0 - tra
        peaks.append(y0)

    peaks = np.array(peaks)

    # 선형 전력 변환
    linear_minus_2 = 10 ** (peaks[0] / 10) * 0.0005
    linear_minus_1_dot_5 = 10 ** (peaks[1] / 10) * 0.0005
    linear_minus_1 = 10 ** (peaks[2] / 10) * 0.0005
    linear_minus_0_dot_5 = 10 ** (peaks[3] / 10) * 0.0005
    linear_0 = 10 ** (peaks[4] / 10) * 0.0005
    linear_0_dot_5 = 10 ** (peaks[5] / 10) * 0.0005

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

    x_nm = x * (10 ** -9)
    result = model.fit(linear_0, params, lamda=x_nm[4])
    neff_value = result.params['neff'].value

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

    x_nm = x * (10 ** -9)

    result2 = model2.fit(linear_minus_2, params, lamda=x_nm[0])
    delta_neff_value = result2.params['delta'].value
    delta_n.append(delta_neff_value)

    # -1.5V
    result3 = model2.fit(linear_minus_1_dot_5, params, lamda=x_nm[1])
    delta_neff_value = result3.params['delta'].value
    delta_n.append(delta_neff_value)

    # -1V
    result4 = model2.fit(linear_minus_1, params, lamda=x_nm[2])
    delta_neff_value = result4.params['delta'].value
    delta_n.append(delta_neff_value)

    # -0.5V
    result5 = model2.fit(linear_minus_0_dot_5, params, lamda=x_nm[3])
    delta_neff_value = result5.params['delta'].value
    delta_n.append(delta_neff_value)
    delta_n.append(0)

    # 0.5V
    result6 = model2.fit(linear_0_dot_5, params, lamda=x_nm[5])
    delta_neff_value = result6.params['delta'].value
    delta_n.append(delta_neff_value)

    # 결과 시각화 - 첫 번째 그래프
    axs[1, 1].plot(x[0], result2.best_fit, label='Fitted -2V')
    axs[1, 1].plot(x[1], result3.best_fit, label='Fitted -1.5V')
    axs[1, 1].plot(x[2], result4.best_fit, label='Fitted -1V')
    axs[1, 1].plot(x[3], result5.best_fit, label='Fitted -0.5V')
    axs[1, 1].plot(x[4], result.best_fit, label='Fitted 0V')
    axs[1, 1].plot(x[5], result6.best_fit, label='Fitted 0.5V')

    axs[1, 1].set_xlabel('Wavelength [nm]')
    axs[1, 1].set_ylabel('Intensity')
    axs[1, 1].set_title('n_eff')
    axs[1, 1].legend(loc='lower center', ncol=2, fontsize='small')

    # 두 번째 그래프
    voltage = [-2, -1.5, -1, -0.5, 0, 0.5]
    axs[1, 2].plot(voltage, delta_n, label='delta', color='red')
    axs[1, 2].set_xlabel('Voltage')
    axs[1, 2].set_ylabel('delta_n')
    axs[1, 2].set_title('n_eff')
    axs[1, 2].legend(loc='best')
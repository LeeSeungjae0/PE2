import numpy as np
from lmfit import Model
from decimal import Decimal

def r_squared(y_true, y_pred):
    y_mean = np.mean(y_true)
    tss = np.sum((y_true - y_mean) ** 2)
    rss = np.sum((y_true - y_pred) ** 2)
    if tss == 0:
        return 1 if rss == 0 else 0
    r2 = 1 - (Decimal(rss) / Decimal(tss))
    return r2

# ax1: 데이터 선형화, ax2: 근사, ax3: 근사화 모둠, ax4: delta neff
def make_linear(ax1, ax2, ax3, ax4, wavelength_array, flat_meas_trans):
    # 선형 전력 변환
    r2_linear = []

    linear_minus_2 = 10 ** (flat_meas_trans[0] / 10) * 0.0005
    linear_minus_1_dot_5 = 10 ** (flat_meas_trans[1] / 10) * 0.0005
    linear_minus_1 = 10 ** (flat_meas_trans[2] / 10) * 0.0005
    linear_minus_0_dot_5 = 10 ** (flat_meas_trans[3] / 10) * 0.0005
    linear_0 = 10 ** (flat_meas_trans[4] / 10) * 0.0005
    linear_0_dot_5 = 10 ** (flat_meas_trans[5] / 10) * 0.0005

    ax1.plot(wavelength_array[0], linear_minus_2, label='-2.0V')
    ax1.plot(wavelength_array[1], linear_minus_1_dot_5, label='-1.5V')
    ax1.plot(wavelength_array[2], linear_minus_1, label='-1.0V')
    ax1.plot(wavelength_array[3], linear_minus_0_dot_5, label='-0.5V')
    ax1.plot(wavelength_array[4], linear_0, label='0.0V')
    ax1.plot(wavelength_array[5], linear_0_dot_5, label='0.5V')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Intensity (W/m^2)')
    ax1.set_title('Flat transmission spectra - linear')
    ax1.grid(True)
    ax1.legend(loc='lower right', bbox_to_anchor=(1.3, 0.47))

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
    ax2.scatter(wavelength_array[4], linear_0, s=5, label='data')
    ax2.plot(wavelength_array[4], result.best_fit, label='fit', color='red')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Intensity (W/m^2)')
    ax2.set_title('Flat transmission spectra - fitted 0.0V')
    ax2.grid(True)
    ax2.legend(loc='lower right', bbox_to_anchor=(1.3, 0.47))

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
    ax3.plot(wavelength_array[0], result2.best_fit, label='-2.0V')
    ax3.plot(wavelength_array[1], result3.best_fit, label='-1.5V')
    ax3.plot(wavelength_array[2], result4.best_fit, label='-1.0V')
    ax3.plot(wavelength_array[3], result5.best_fit, label='-0.5V')
    ax3.plot(wavelength_array[4], result6.best_fit, label='0.0V')
    ax3.plot(wavelength_array[5], result7.best_fit, label='0.5V')

    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel('Intensity (W/m^2)')
    ax3.set_title('Flat transmission spectra - fitted')
    ax3.grid(True)
    ax3.legend(loc='lower right', bbox_to_anchor=(1.3, 0.47))

    voltage = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5]
    x = np.arange(len(delta_n))
    coefficients = np.polyfit(x, delta_n, 2)
    polynomial = np.poly1d(coefficients)
    y_fit = polynomial(x)
    r2=r_squared(delta_n,y_fit)

    ax4.scatter(voltage, delta_n, label='delta')
    ax4.plot(voltage, y_fit, 'r-', label=f'fit (R²: {r2:.4f})')
    ax4.set_xlabel('Voltage(V)')
    ax4.set_ylabel('delta_n')
    ax4.set_title('Delta n_eff')
    ax4.grid(True)
    ax4.legend(loc='upper right')

    r2_linear.append(r2_2)      # fitted 0V
    r2_linear.append(r2_3)      # -2V
    r2_linear.append(r2_4)      # -1.5V
    r2_linear.append(r2_5)      # -1V
    r2_linear.append(r2_6)      # -0.5V
    r2_linear.append(r2_7)      # 0V
    r2_linear.append(r2_8)      # 0.5V

    return r2_linear

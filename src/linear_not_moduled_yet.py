import xml.etree.ElementTree as eT
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from decimal import Decimal
from scipy.signal import find_peaks

# 개선 방향 모듈화 peak로 flat하게 만든 transmission data 추출 후 그대로 박아서 만둘수 있도록 D08의 list길이차이 고치는 방식아직 적용안했으므로 넣지 말것

# XML 파일 파싱 절대경로 복사하여 붙여놓기 연습용임 아직 모듈화 전
tree = eT.parse(r'C:\Users\User\PycharmProjects\pythonProject1\PE2\dat\HY202103\D07\20190715_190855\HY202103_D07_(0,-4)_LION1_DCM_LMZC.xml')
root = tree.getroot()

# 데이터 리스트 초기화
x_data_list = []
y_data_list = []

def r_squared(y_true, y_pred):
    y_mean = np.mean(y_true)
    tss = np.sum((y_true - y_mean) ** 2)
    rss = np.sum((y_true - y_pred) ** 2)
    r2 = Decimal(1) - (Decimal(rss) / Decimal(tss))
    return r2

# 'L' 태그의 데이터를 가져오기
for wl in root.iter('L'):
    wavel = wl.text.replace(",", " ").split()
    x = [float(num) for num in wavel]
    x_data_list.append(x)

x = np.array(x_data_list)

# 'IL' 태그의 데이터를 가져오기
for tr in root.iter('IL'):
    trans = tr.text.replace(",", " ").split()
    y = [float(num) for num in trans]
    y_data_list.append(y)

y = np.array(y_data_list)

# 데이터 정규화 다항식 고차수 근사 error해결
x_mean = np.mean(x[6])
x_std = np.std(x[6])
x_normalized = (x[6] - x_mean) / x_std

# 다항식 근사
coefficients = np.polyfit(x_normalized, y[6], 7)
poly = np.poly1d(coefficients)
y_pred = poly(x_normalized)

# 데이터 - 근사값
y6 = y[6] - y_pred

# find peak사용
peaks = []
for i in 0,1,2,3,4,5:
    # data - 근사 ref 값
    y0 = y[i] - y_pred
    peaks_distance, _ = find_peaks(y0, distance=len(x[i])/2)
    x_= x[i][peaks_distance]
    y0_= y0[peaks_distance]
    coefficients = np.polyfit(x_, y0_, 1)
    poly = np.poly1d(coefficients)
    tra = poly(x[i])
    y0 = y0 - tra
    peaks.append(y0)

peaks = np.array(peaks)
# R^2 계산
r2 = r_squared(y[6], y_pred)
print("R^2:", r2)

# 선형 전력 변환
linear_minus_2 = 10**(peaks[0]/10) * 0.0005
linear_minus_1_dot_5 = 10**(peaks[1]/10) * 0.0005
linear_minus_1 = 10**(peaks[2]/10) * 0.0005
linear_minus_0_dot_5 = 10**(peaks[3]/10) * 0.0005
linear_0 = 10**(peaks[4]/10) * 0.0005
linear_0_dot_5 = 10**(peaks[5]/10) * 0.0005

# plt.scatter(x[0], linear_minus_2, s=1, label='Measured -2V')
# plt.scatter(x[1], linear_minus_1_dot_5, s=1, label='Measured -1.5V')
# plt.scatter(x[2], linear_minus_1, s=1, label='Measured -1V')
# plt.scatter(x[3], linear_minus_0_dot_5, s=1, label='Measured -0.5V')
plt.scatter(x[4], linear_0, s=1, label='Measured 0.0V')
# plt.scatter(x[5], linear_0_dot_5, s=1, label='Measured 0.5V')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Intensity')
plt.title('Flat transmission spectra - as measured')
plt.legend(loc='lower center', ncol=2, fontsize='small')
plt.show()

def intensity(lamda, neff, delta, l, deltaL, I0):
    I = I0 * np.sin(((2*np.pi/lamda) * deltaL * neff) / 2 + ((2*np.pi/lamda) * l * delta / 2))**2
    return I

# 모델 생성
model = Model(intensity)

# 파라미터 설정
params = model.make_params(neff=4.1, delta=0, l=500*(10**-6), deltaL=40*(10**-6), I0=0.0005)
params['delta'].vary = False
params['l'].vary = False
params['deltaL'].vary = False
params['I0'].vary = False
params['neff'].vary = True

x_nm = x*(10**-9)
# 피팅 수행
result = model.fit(linear_0, params, lamda=x_nm[4])

r2_2 = r_squared(linear_0, result.best_fit)
print("R^2:", r2_2)
neff_value = result.params['neff'].value
print("Optimized neff value:", neff_value)

# 결과 시각화
plt.scatter(x[4], linear_0, s=5, label='Measured 0.0V')
plt.plot(x[4], result.best_fit, label='Fitted 0.0V', color='red')

plt.xlabel('Wavelength [nm]')
plt.ylabel('Intensity')
plt.title('Flat transmission spectra - as measured')
plt.legend(loc='lower center', ncol=2, fontsize='small')
plt.show()

delta_n = []
# 모델 생성
model2 = Model(intensity)

# 파라미터 설정
params = model2.make_params(neff=neff_value, delta=0, l=500*(10**-6), deltaL=40*(10**-6), I0=0.0005)
params['delta'].vary = True
params['l'].vary = False
params['deltaL'].vary = False
params['I0'].vary = False
params['neff'].vary = False

x_nm = x*(10**-9)
# 피팅 수행
result2 = model2.fit(linear_minus_2, params, lamda=x_nm[0])
r2_3 = r_squared(linear_minus_2, result2.best_fit)
delta_neff_value = result2.params['delta'].value
print("Optimized delta_neff value:", delta_neff_value)
print("R^2:", r2_3)
delta_n.append(delta_neff_value)

result3 = model2.fit(linear_minus_1_dot_5, params, lamda=x_nm[1])
r2_4 = r_squared(linear_minus_1_dot_5, result3.best_fit)
delta_neff_value = result3.params['delta'].value
print("Optimized delta_neff value:", delta_neff_value)
print("R^2:", r2_4)
delta_n.append(delta_neff_value)

result4 = model2.fit(linear_minus_1, params, lamda=x_nm[2])
r2_5 = r_squared(linear_minus_1, result4.best_fit)
delta_neff_value = result4.params['delta'].value
print("Optimized delta_neff value:", delta_neff_value)
print("R^2:", r2_5)
delta_n.append(delta_neff_value)

result5 = model2.fit(linear_minus_0_dot_5, params, lamda=x_nm[3])
r2_6 = r_squared(linear_minus_0_dot_5, result5.best_fit)
delta_neff_value = result5.params['delta'].value
print("Optimized delta_neff value:", delta_neff_value)
print("R^2:", r2_6)
delta_n.append(delta_neff_value)

print("Optimized delta_neff value:", 0)
print("R^2:", r2_2)
delta_n.append(0)

result6 = model2.fit(linear_0_dot_5, params, lamda=x_nm[5])
r2_7 = r_squared(linear_0_dot_5, result6.best_fit)
delta_neff_value = result6.params['delta'].value
print("Optimized delta_neff value:", delta_neff_value)
print("R^2:", r2_7)
delta_n.append(delta_neff_value)

# 결과 시각화
plt.plot(x[0], result2.best_fit, label='Fitted -2V')
plt.plot(x[1], result3.best_fit, label='Fitted -1.5V')
plt.plot(x[2], result4.best_fit, label='Fitted -1V')
plt.plot(x[3], result5.best_fit, label='Fitted -0.5V')
plt.plot(x[4], result.best_fit, label='Fitted 0V')
plt.plot(x[5], result6.best_fit, label='Fitted 0.5V')

plt.xlabel('Wavelength [nm]')
plt.ylabel('Intensity')
plt.title('Flat transmission spectra - as measured')
plt.legend(loc='lower center', ncol=2, fontsize='small')
plt.show()

voltage = [-2, -1.5, -1, -0.5, 0, 0.5]


plt.plot( voltage, delta_n, label='delta', color='red')
plt.xlabel('Voltage')
plt.ylabel('delta_n')
plt.title('Flat transmission spectra - as measured')
plt.show()

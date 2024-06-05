import xml.etree.ElementTree as eT
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from decimal import Decimal

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
y0 = y[0] - y_pred
y1 = y[1] - y_pred
y2 = y[2] - y_pred
y3 = y[3] - y_pred
y4 = y[4] - y_pred
y5 = y[5] - y_pred
y6 = y[6] - y_pred

# R^2 계산
r2 = r_squared(y[6], y_pred)
print("R^2:", r2)

# 선형 전력 변환
linear_minus_2 = 10**(y0/10) * 0.0005
linear_minus_1_dot_5 = 10**(y1/10) * 0.0005
linear_minus_1 = 10**(y2/10) * 0.0005
linear_minus_0_dot_5 = 10**(y3/10) * 0.0005
linear_0 = 10**(y4/10) * 0.0005
linear_0_dot_5 = 10**(y5/10) * 0.0005

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
print(x_nm)
# 피팅 수행
result = model.fit(linear_0, params, lamda=x_nm[4])

print(result.fit_report())
r2_2 = r_squared(linear_0, result.best_fit)
print("R^2:", r2_2)

# 결과 시각화
plt.scatter(x[4], linear_0, s=5, label='Measured 0.0V')
plt.plot(x[4], result.best_fit, label='Fitted 0.0V', color='red')

plt.xlabel('Wavelength [nm]')
plt.ylabel('Intensity')
plt.title('Flat transmission spectra - as measured')
plt.legend(loc='lower center', ncol=2, fontsize='small')
plt.show()

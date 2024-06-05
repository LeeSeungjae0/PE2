import xml.etree.ElementTree as eT
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from decimal import Decimal

# XML 파일 파싱 절대경로 복사하여 붙여놓기 연습용임 아직 모듈하 전
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

# 다항식 근사
coefficients = np.polyfit(x[6], y[6], 7)
poly = np.poly1d(coefficients)
y_pred = poly(x[6])

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


plt.scatter(x[4], linear_0, s=5, label='Measured 0.0V')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Intensity')
plt.title('Flat transmission spectra - as measured')
plt.legend(loc='lower center', ncol=2, fontsize='small')
plt.show()

def intensity(neff, delta, l, deltaL, lamda, I0):
    I = I0 * np.sin(((2*np.pi/lamda) * deltaL * neff) / 2 + ((2*np.pi/lamda) * l * delta / 2))
    return I

# 모델 생성
model = Model(intensity)

# 파라미터 설정
params = model.make_params(neff=0, delta=0, l=0, deltaL=0, I0=0)
params['neff'].set(value=4.1)
params['delta'].set(value=0, vary=False)
params['l'].set(value=500*(10**-6), vary=False)
params['deltaL'].set(value=40*(10**-6), vary=False)
params['I0'].set(value=0.0005, vary=False)

# 피팅 수행
result = model.fit(linear_0, params, lamda=x[4])

print(result.fit_report())

# 결과 시각화
plt.scatter(x[4], linear_0, s=5, label='Measured 0.0V')
plt.plot(x[4], result.best_fit, label='Fitted 0.0V', color='red')

plt.xlabel('Wavelength [nm]')
plt.ylabel('Intensity')
plt.title('Flat transmission spectra - as measured')
plt.legend(loc='lower center', ncol=2, fontsize='small')
plt.show()

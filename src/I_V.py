from lmfit import Parameters, Minimizer
import numpy as np
from decimal import Decimal

def process_iv_data(root):
    voltage_str = root.find('.//Voltage').text
    voltage_values = np.array([float(v) for v in voltage_str.split(',')])
    current_str = root.find('.//Current').text
    current_values = np.array([float(v) for v in current_str.split(',')])
    abs_current = np.abs(current_values)

    def mob(params, x, data=None):
        Is = params['Is']
        Vt = params['Vt']
        n = params['n']
        if abs_current[0]*1000 > abs_current[12]:
            poly_coeff = np.polyfit(x[x < 2], data[x < 2], deg=12)
            model_negative = np.polyval(poly_coeff, x[x < 2])
            model_positive = Is * (np.exp(x[x >= 2] / (n * Vt)) - 1)
            model = np.concatenate((model_negative, model_positive))
        else:
            poly_coeff = np.polyfit(x[x < 0], data[x < 0], deg=6)
            model_negative = np.polyval(poly_coeff, x[x < 0])
            model_positive = Is * (np.exp(x[x >= 0] / (n * Vt)) - 1)
            model = np.concatenate((model_negative, model_positive))
        if data is None:
            return model
        else:
            return model - data

    pars = Parameters()
    pars.add('Is', value=10 ** -8)
    pars.add('Vt', value=0.026)
    pars.add('n', value=1, vary=False)

    fitter = Minimizer(mob, pars, fcn_args=(voltage_values, current_values))
    result = fitter.minimize()
    final = abs_current + result.residual

    RSS = np.sum(result.residual ** 2)
    mean_current = np.mean(current_values)
    TSS = np.sum((current_values - mean_current) ** 2)
    R_squared = 1 - (Decimal(RSS) / Decimal(TSS))

    return voltage_values, abs_current, final, R_squared, current_values

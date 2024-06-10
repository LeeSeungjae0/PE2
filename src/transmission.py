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

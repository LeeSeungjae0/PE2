def extract_reference_data(root):
    reference_wave = []
    reference_trans = []
    for wavelengthsweep in root.findall('.//WavelengthSweep'):
        wavelength_str = wavelengthsweep.find('.//L').text
        transmission_str = wavelengthsweep.find('.//IL').text
        wavelength_list = [float(w) for w in wavelength_str.split(',')]
        transmission_list = [float(t) for t in transmission_str.split(',')]
        reference_wave = wavelength_list
        reference_trans = transmission_list
    return reference_wave, reference_trans

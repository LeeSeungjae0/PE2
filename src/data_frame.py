import pandas as pd

def create_data_frame():
    data_dict = {key: [] for key in
                 ['Lot', 'Wafer', 'Mask', 'TestSite', 'Name', 'Date', 'Script ID', 'Script Version',
                  'Script Owner', 'Operator', 'Row', 'Column', 'ErrorFlag', 'Error description', 'Analysis Wavelength',
                  'Rsq of Ref. spectrum (Nth)', 'Max transmission of Ref. spec. (dB)', 'Rsq of IV', 'I at -1V [A]',
                  'I at 1V [A]', 'Graph Image']}
    return data_dict

def update_data_frame(data_dict, root, r_squared, ref_transmission_point, R_squared, current_values, voltage_values, abs_current, transmissions):
    test_site_info = root.find('.//TestSiteInfo')
    data_dict['Lot'].append(test_site_info.get('Batch'))
    data_dict['Wafer'].append(test_site_info.get('Wafer'))
    data_dict['Mask'].append(test_site_info.get('Maskset'))
    data_dict['TestSite'].append(test_site_info.get('TestSite'))

    data_dict['Operator'].append(root.get('Operator'))

    modulator = root.find('.//ElectroOpticalMeasurements/ModulatorSite/Modulator')
    data_dict['Name'].append(modulator.get('Name'))

    data_dict['Script ID'].append('process LMZ')
    data_dict['Script Version'].append('0.1')
    data_dict['Script Owner'].append('A2')

    date_stamp = root.find('.//PortCombo').get('DateStamp')
    data_dict['Date'].append(date_stamp)

    data_dict['Row'].append(test_site_info.get('DieRow'))
    data_dict['Column'].append(test_site_info.get('DieColumn'))

    if float(r_squared) < 0.95:
        data_dict['ErrorFlag'].append('1')
        data_dict['Error description'].append('Ref. spec. Error')
    elif abs(current_values[0])*1000 > abs(current_values[12]):
        data_dict['ErrorFlag'].append('2')
        data_dict['Error description'].append('IV. Error')
    else:
        data_dict['ErrorFlag'].append('0')
        data_dict['Error description'].append('No Error')

    data_dict['Analysis Wavelength'].append(int(transmissions[0][1][0]))

    data_dict['Rsq of Ref. spectrum (Nth)'].append(format(float(r_squared), '.4f'))
    data_dict['Max transmission of Ref. spec. (dB)'].append(format(float(ref_transmission_point), '.4f'))
    data_dict['Rsq of IV'].append(format(float(R_squared), '.4f'))

    i_at_minus_1v = None
    i_at_1v = None
    for voltage, current in zip(voltage_values, abs_current):
        if voltage == -1:
            i_at_minus_1v = '{:.4e}'.format(float(current))
        elif voltage == 1:
            i_at_1v = '{:.4e}'.format(float(current))

    data_dict['I at -1V [A]'].append(i_at_minus_1v)
    data_dict['I at 1V [A]'].append(i_at_1v)

    return data_dict

def save_data_frame(data_dict, xlsx_file_path):
    df = pd.DataFrame(data_dict)

    # ErrorFlag 열의 값이 문자열 '0'인 경우를 포함하여 필터링
    error_counts = df[df['ErrorFlag'] != '0'].groupby(['Wafer', 'TestSite'])['ErrorFlag'].count()

    # Calculate total number of entries for each wafer and test site
    total_counts_all = df.groupby(['Wafer', 'TestSite'])['ErrorFlag'].count()


    # Get detailed error descriptions distribution for each wafer and test site
    error_descriptions = df[df['ErrorFlag'] != '0'].groupby(['Wafer', 'TestSite', 'Error description']).size()

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Total Errors': error_counts,
        'Total Entries': total_counts_all,

    }).reset_index()

    # Filter the original DataFrame to include only rows where ErrorFlag is not '0'
    errors_only_df = df[df['ErrorFlag'] != '0']

    with pd.ExcelWriter(xlsx_file_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')

        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Write the detailed error descriptions to the second sheet
        error_descriptions = error_descriptions.reset_index(name='Count')
        error_descriptions.to_excel(writer, sheet_name='Error Descriptions', index=False)

        # Write the errors-only DataFrame to a third sheet
        errors_only_df.to_excel(writer, sheet_name='Errors Only', index=False)
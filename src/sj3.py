import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from lmfit import Parameters, Minimizer
from decimal import Decimal
from matplotlib.ticker import FuncFormatter
from scipy.signal import find_peaks

# Function to process XML files in a given directory
def process_xml_files(directory1, directory2, current_directory):
    # Directory to save results (CSV and PNG files)
    output_directory = os.path.join(current_directory, '..', 'res', 'HY202103', directory1, directory2)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Path to the CSV file
    csv_file_path = os.path.join(output_directory, f'{directory1}_{directory2}.csv')

    # Clear existing content of the CSV file
    open(csv_file_path, 'w').close()

    # Get the XML directory
    xml_directory = os.path.join(current_directory, '..', 'dat', 'HY202103', directory1, directory2)

    # Ensure the XML directory exists
    if not os.path.isdir(xml_directory):
        print(f"The directory {xml_directory} does not exist. Please enter a valid directory path.")
        return

    # Parse XML files in the directory
    for filename in os.listdir(xml_directory):
        if filename.endswith('LMZC.xml') or filename.endswith('LMZO.xml'):
            print('---', filename, '---')
            # Parse XML file and get the root element
            xml_file_path = os.path.join(xml_directory, filename)
            tree = elemTree.parse(xml_file_path)
            root = tree.getroot()
            # Parse data and perform analysis here
            # Parsing voltage and current data
            voltage_str = root.find('.//Voltage').text
            voltage_values = np.array([float(v) for v in voltage_str.split(',')])
            current_str = root.find('.//Current').text
            current_values = np.array([float(v) for v in current_str.split(',')])
            abs_current = np.abs(current_values)

            def mob(params, x, data=None):
                Is = params['Is']
                Vt = params['Vt']
                n = params['n']
                poly_coeff = np.polyfit(x[x < 0], data[x < 0], deg=6)
                model_negative = np.polyval(poly_coeff, x[x < 0])
                model_positive = Is * (np.exp(x[x >= 0] / (n * Vt)) - 1)
                model = np.concatenate((model_negative, model_positive))
                if data is None:
                    return model
                else:
                    return model - data

            # Set up the initial parameter values
            pars = Parameters()
            pars.add('Is', value=10 ** -8)
            pars.add('Vt', value=0.026)
            pars.add('n', value=1, vary=False)

            fitter = Minimizer(mob, pars, fcn_args=(voltage_values, current_values))
            result = fitter.minimize()
            final = abs_current + result.residual

            # Calculate R-squared
            RSS = np.sum(result.residual ** 2)
            mean_current = np.mean(current_values)
            TSS = np.sum((current_values - mean_current) ** 2)
            R_squared = 1 - (Decimal(RSS) / Decimal(TSS))

            # Initialize figure and axes for subplots
            fig, axs = plt.subplots(2, 3, figsize=(18, 8))

            def log_formatter(x, pos):
                return "{:.0e}".format(x)

            # Determine the y position for the text dynamically based on the current values
            if abs_current.max() >= 1e-3 and abs_current.min() <= 1e-10:
                y_text_position = 1e-5  # For a broad range of currents
            elif abs_current.max() <= 2e-10 and abs_current.min() >= 6e-11:
                y_text_position = 1e-10  # For a narrow range of low currents
            else:
                y_text_position = min(abs_current) * 1.5  # Default case

            # First subplot: IV raw data & fitted data
            axs[1, 0].scatter(voltage_values, abs_current, label='data')
            axs[1, 0].plot(voltage_values, final, 'r-', label=f'fit (R²: {R_squared:.4f})')
            axs[1, 0].set_xlim(-2, 1)
            axs[1, 0].set_yscale('log')
            axs[1, 0].yaxis.set_major_formatter(FuncFormatter(log_formatter))
            axs[1, 0].set_title('IV raw data & fitted data (log scale)')
            axs[1, 0].set_ylabel('Absolute Current (A)')
            axs[1, 0].set_xlabel('Voltage (V)')
            axs[1, 0].grid(True)
            axs[1, 0].legend(loc='upper left')
            axs[1, 0].text(-1.9, y_text_position,
                           f'  R²: {R_squared}\n-2V: {current_values[0]:.2e}\n-1V: {current_values[4]:.2e} \n 1V: {current_values[12]:.2e}',
                           fontsize=10, horizontalalignment='left', verticalalignment='center')

            ref_transmission_point = -50
            # Second subplot: Transmission vs Wavelength
            for i, wavelengthsweep in enumerate(root.findall('.//WavelengthSweep')):
                dc_bias = float(wavelengthsweep.get('DCBias'))
                # Do not display legend for the last DCBias
                if i == len(root.findall('.//WavelengthSweep')) - 1:
                    label = None
                else:
                    label = f'{dc_bias}V'
                # Extract wavelength and transmission data
                wavelength_str = wavelengthsweep.find('.//L').text
                transmission_str = wavelengthsweep.find('.//IL').text

                # Convert strings to lists
                wavelength_list = [float(w) for w in wavelength_str.split(',')]
                transmission_list = [float(t) for t in transmission_str.split(',')]

                if i == len(root.findall('.//WavelengthSweep')) - 1:
                    # Find peaks in transmission data
                    peaks, _ = find_peaks(transmission_list, distance=50)  # Adjust distance parameter as needed

                    # Iterate through peaks and find the one within the specified wavelength range
                    for peak_index in peaks:
                        if transmission_list[peak_index] > ref_transmission_point:
                            ref_transmission_point = transmission_list[peak_index]

                axs[0, 0].plot(wavelength_list, transmission_list, label=label)

            # Set labels and title for the second subplot
            axs[0, 0].set_xlabel('Wavelength (nm)')
            axs[0, 0].set_ylabel('Transmission (dB)')
            axs[0, 0].set_title('Transmission vs Wavelength')
            axs[0, 0].grid(True)
            axs[0, 0].legend(loc='lower right', bbox_to_anchor=(1.2, 0.47))

            # reference setup
            reference_wave = wavelength_list
            reference_trans = transmission_list
            axs[0, 1].plot(reference_wave, reference_trans, label=f'data')

            # Polynomial degree
            degrees = range(1, 7)

            # Store R-squared values
            r_squared_values = {}

            # Fit polynomials and plot
            for degree in degrees:
                # Fit polynomial with no rank warning
                coeffs, _, _, _ = np.linalg.lstsq(np.vander(reference_wave, degree + 1), reference_trans, rcond=None)
                polynomial = np.poly1d(coeffs)

                axs[0, 1].plot(reference_wave, polynomial(reference_wave), label=f'{degree}th')

                # Calculate R-squared
                mean_transmission = np.mean(reference_trans)
                total_variation = np.sum((reference_trans - mean_transmission) ** 2)
                residuals = np.sum((transmission_list - polynomial(reference_wave)) ** 2)
                r_squared = 1 - (residuals / total_variation)

                # Store R-squared values
                r_squared_values[degree] = r_squared

                # Print polynomial and R-squared value
                polynomials = []
                for i in range(degree, 0, -1):
                    polynomials.append(f"{coeffs[degree - i]:.16f}X^{i}")
                polynomials.append(f"{coeffs[degree]:.16f}")
                polynomial_str = '+'.join(polynomials)

            # Position for R-squared values
            max_transmission = np.max(reference_trans)
            min_transmission = np.min(reference_trans)
            y_pos = 0.5 * (max_transmission + min_transmission) - 0.3
            x_pos = reference_wave[0] + 0.5 * (reference_wave[-1] - reference_wave[0])
            for degree, r_squared in r_squared_values.items():
                axs[0, 1].text(x_pos, y_pos, f'{degree}th R²: {r_squared:.4f}', fontsize=10, verticalalignment='center', horizontalalignment='center')
                y_pos -= 0.06 * (max_transmission - min_transmission)

            # Plot settings
            axs[0, 1].set_xlabel('Wavelength (nm)')
            axs[0, 1].set_ylabel('Transmission (dB)')
            axs[0, 1].set_title('Transmission spectra - Processed and fitting')
            axs[0, 1].grid(True)
            axs[0, 1].legend(loc='lower right', bbox_to_anchor=(1.19, 0.39))

            poly6 = polynomial(reference_wave)
            max_transmission_point, max_transmission_point2 = -50, -50
            mid_transmission = (min(wavelength_list)+max(wavelength_list))/2

            for i, wavelengthsweep in enumerate(root.findall('.//WavelengthSweep')):
                # Extract wavelength and transmission data
                wavelength_str = wavelengthsweep.find('.//L').text
                transmission_str = wavelengthsweep.find('.//IL').text

                # Convert strings to lists
                wavelength_list = [float(w) for w in wavelength_str.split(',')]
                transmission_list = [float(t) for t in transmission_str.split(',')]

                # Plot the line connecting the two points
                flat_transmission = np.array(transmission_list) - np.array(poly6)

                # Exclude the last graph
                if i != len(root.findall('.//WavelengthSweep')) - 1:
                    # Find peaks in transmission data
                    peaks, _ = find_peaks(flat_transmission, distance=50)  # Adjust distance parameter as needed

                    # Iterate through peaks and find the one within the specified wavelength range
                    for peak_index in peaks:
                        if min(wavelength_list) <= wavelength_list[peak_index] <= mid_transmission:
                            # Update maximum transmission point if the peak is higher
                            if flat_transmission[peak_index] > max_transmission_point:
                                max_transmission_point = flat_transmission[peak_index]
                                max_transmission_wavelength = wavelength_list[peak_index]

                if i != len(root.findall('.//WavelengthSweep')) - 1:
                    # Find peaks in transmission data
                    peaks, _ = find_peaks(flat_transmission, distance=50)  # Adjust distance parameter as needed

                    # Iterate through peaks and find the one within the specified wavelength range
                    for peak_index in peaks:
                        if mid_transmission <= wavelength_list[peak_index] <= max(wavelength_list):
                            # Update maximum transmission point if the peak is higher
                            if flat_transmission[peak_index] > max_transmission_point2:
                                max_transmission_point2 = flat_transmission[peak_index]
                                max_transmission_wavelength2 = wavelength_list[peak_index]

            # Calculate the line connecting the two maximum points
            m = (max_transmission_point2 - max_transmission_point) / (
                    max_transmission_wavelength2 - max_transmission_wavelength)
            b = max_transmission_point - m * max_transmission_wavelength
            peak_fit = m * np.array(wavelength_list) + b

            for i, wavelengthsweep in enumerate(root.findall('.//WavelengthSweep')):
                dc_bias = float(wavelengthsweep.get('DCBias'))
                # Do not display legend for the last DCBias
                if i == len(root.findall('.//WavelengthSweep')) - 1:
                    label = None
                else:
                    label = f'{dc_bias}V'
                # Extract wavelength and transmission data
                wavelength_str = wavelengthsweep.find('.//L').text
                transmission_str = wavelengthsweep.find('.//IL').text

                # Convert strings to lists
                wavelength_list = [float(w) for w in wavelength_str.split(',')]
                transmission_list = [float(t) for t in transmission_str.split(',')]

                # Plot the line connecting the two points
                if i != len(root.findall('.//WavelengthSweep')) - 1:
                    flat_meas_trans = np.array(transmission_list) - np.array(poly6) - np.array(peak_fit)
                else:
                    flat_meas_trans = np.array(transmission_list) - np.array(poly6)
                # Plot the graph
                axs[0, 2].plot(wavelength_list, flat_meas_trans, label=label)

            # Set labels and title for the second subplot
            axs[0, 2].set_xlabel('Wavelength (nm)')
            axs[0, 2].set_ylabel('Flat Mearsured Transmission (dB)')
            axs[0, 2].set_title('Flat Transmission spectra -as measured')
            axs[0, 2].grid(True)
            axs[0, 2].legend(loc='lower right', bbox_to_anchor=(1.2, 0.47))

            # Initialize the dictionary to store necessary information
            data_dict = {key: [] for key in
                         ['Lot', 'Wafer', 'Mask', 'TestSite', 'Name', 'Date', 'Script ID', 'Script Version',
                          'Script Owner',
                          'Operator', 'Row', 'Column', 'ErrorFlag', 'Error description', 'Analysis Wavelength',
                          'Rsq of Ref. spectrum (Nth)', 'Max transmission of Ref. spec. (dB)', 'Rsq of IV',
                          'I at -1V [A]',
                          'I at 1V [A]', 'Graph Image']}

            # Extract Row and Column information from XML
            test_site_info = root.find('.//TestSiteInfo')

            # Extract Lot, Wafer, Mask, TestSite information from XML
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
            else:
                data_dict['ErrorFlag'].append('0')
                data_dict['Error description'].append('No Error')

            data_dict['Analysis Wavelength'].append('1550')

            data_dict['Rsq of Ref. spectrum (Nth)'].append(format(float(r_squared), '.4f'))
            data_dict['Max transmission of Ref. spec. (dB)'].append(format(float(ref_transmission_point), '.4f'))
            data_dict['Rsq of IV'].append(format(float(R_squared), '.4f'))

            values_to_display = [-1, 1]
            for voltage, current in zip(voltage_values, abs_current):
                if voltage == -1:
                    data_dict['I at -1V [A]'].append('{:.4e}'.format(float(current)))
                elif voltage == 1:
                    data_dict['I at 1V [A]'].append('{:.4e}'.format(float(current)))
                    # Add the graph image file name to the dictionary

            filename = filename.replace('.xml', '')
            # Save the figure as an image in the output directory
            image_filename = f'{filename}.png'
            image_path = os.path.join(output_directory, image_filename)
            file_path = os.path.abspath(image_path).replace('\\', '/')  # 수정된 부분
            filename_no_ext, _ = os.path.splitext(filename)

            data_dict['Graph Image'].append(f'=HYPERLINK("{file_path}", "{filename_no_ext}")')

            # Convert extracted information into a DataFrame
            df = pd.DataFrame(data_dict)

            # Print the result
            pd.set_option('display.max_columns', None)

            # Append data to the existing CSV file with UTF-8 encoding
            with open(csv_file_path, 'a', newline='', encoding='utf-8') as f:
                df.to_csv(f, mode='a', header=f.tell() == 0, index=False)

            # Adjust layout to prevent overlap
            plt.suptitle(filename)
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3)
            axs[1, 1].axis('off')
            axs[1, 2].axis('off')

            # Save the figure as an image in the output directory
            plt.savefig(image_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free up memory

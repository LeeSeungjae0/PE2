import os
import matplotlib.pyplot as plt
from parsing import parse_xml_files
from iv import process_iv_data
from transmission import process_transmission_data
from reference import extract_reference_data
from data_frame import update_data_frame
from plot import plot_iv, plot_transmission, plot_reference, plot_flat_transmission
from linear import make_linear

def main(directory0, directory1, directory2, current_directory, data_dict, testsite, graph_image, timestamp):
    xml_files, xml_directory = parse_xml_files(directory0, directory1, directory2, current_directory,testsite)
    if xml_files is None:
        print("Failed to parse XML files. Please check the directory paths and try again.")
        return

    for filename, root in xml_files:
        voltage_values, abs_current, final, R_squared, current_values = process_iv_data(root)
        transmissions = process_transmission_data(root)
        reference_wave, reference_trans = extract_reference_data(root)

        fig, axs = plt.subplots(2, 4, figsize=(20, 8))

        plot_iv(axs[0, 3], voltage_values, abs_current, final, R_squared, current_values)
        ref_transmission_point = plot_transmission(axs[0, 0], transmissions)
        r_squared_values = {}
        polynomial = plot_reference(axs[0, 1], reference_wave, reference_trans, r_squared_values)
        wavelength_array, flat_meas_trans = plot_flat_transmission(axs[0, 2], transmissions, polynomial)
        make_linear(axs[1, 0], axs[1, 1], axs[1, 2], axs[1, 3], wavelength_array, flat_meas_trans)
        data_dict = update_data_frame(data_dict, root, r_squared_values[6], ref_transmission_point, R_squared,
                                      current_values, voltage_values, abs_current, transmissions)

        filename = filename.replace('.xml', '')
        image_output_directory = os.path.join(current_directory, 'res', timestamp, directory0, directory1, directory2)
        os.makedirs(image_output_directory, exist_ok=True)
        image_filename = f'{filename}.png'
        image_path = os.path.join(image_output_directory, image_filename)
        file_path = os.path.abspath(image_path).replace('\\', '/')
        filename_no_ext, _ = os.path.splitext(filename)
        if graph_image:
            data_dict['Graph Image'].append(f'=HYPERLINK("{file_path}", "{filename_no_ext}")')
        else:
            data_dict['Graph Image'].append('None')

        image_path = os.path.join(image_output_directory, f'{filename_no_ext}.png')
        plt.suptitle(filename)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        if graph_image:
            plt.savefig(image_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        print('---', filename, '---')

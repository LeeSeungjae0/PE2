import matplotlib.pyplot as plt

# Importing functions from the separated modules
from Parsing import parse_xml_files
from I_V import process_iv_data
from Transmission import process_transmission_data
from Reference import extract_reference_data
from Flat_transmission import process_flat_transmission
from Data_Frame import create_data_frame, update_data_frame, save_data_frame
from Plot import plot_iv, plot_transmission, plot_reference, plot_flat_transmission




def main(directory1, directory2, current_directory):
    xml_files, csv_file_path, output_directory = parse_xml_files(directory1, directory2, current_directory)

    if xml_files is None:
        print("Failed to parse XML files. Please check the directory paths and try again.")
        return

    for filename, root in xml_files:
        voltage_values, abs_current, final, R_squared, current_values = process_iv_data(root)
        transmissions = process_transmission_data(root)
        reference_wave, reference_trans = extract_reference_data(root)

        fig, axs = plt.subplots(2, 3, figsize=(18, 8))

        plot_iv(axs[1, 0], voltage_values, abs_current, final, R_squared, current_values)
        ref_transmission_point = plot_transmission(axs[0, 0], transmissions)
        r_squared_values = {}
        polynomial = plot_reference(axs[0, 1], reference_wave, reference_trans, r_squared_values)
        peak_fit = process_flat_transmission(transmissions, polynomial)
        plot_flat_transmission(axs[0, 2], transmissions, polynomial)

        data_dict = create_data_frame()
        data_dict = update_data_frame(data_dict, root, r_squared_values[6], ref_transmission_point, R_squared,
                                      current_values, voltage_values, abs_current)

        filename = filename.replace('.xml', '')
        # Save the figure as an image in the output directory
        image_filename = f'{filename}.png'
        image_path = os.path.join(output_directory, image_filename)
        file_path = os.path.abspath(image_path).replace('\\', '/')  # 수정된 부분
        filename_no_ext, _ = os.path.splitext(filename)

        data_dict['Graph Image'].append(f'=HYPERLINK("{file_path}", "{filename_no_ext}")')
        save_data_frame(data_dict, csv_file_path)

        filename_no_ext = os.path.splitext(filename)[0]
        image_path = os.path.join(output_directory, f'{filename_no_ext}.png')
        plt.suptitle(filename)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        axs[1, 1].axis('off')
        axs[1, 2].axis('off')
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print('---', filename, '---')
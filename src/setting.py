import os
from main import main
from data_frame import create_data_frame, save_data_frame
from datetime import datetime

def set_up(directory0, directory1, directory2, current_directory, testsite, xlsx_file, graph_image):
    base_directory = os.path.join(current_directory, 'dat', directory0)
    data_dict = create_data_frame()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print (timestamp)
    if directory1.lower() == 'all':
        if not os.path.isdir(base_directory):
            print(f"The base directory {base_directory} does not exist.")
        else:
            for dir1 in os.listdir(base_directory):
                dir1_path = os.path.join(base_directory, dir1)
                if not os.path.isdir(dir1_path):
                    print(f"The directory {dir1_path} does not exist or is not a directory.")
                    continue
                for dir12 in os.listdir(dir1_path):
                    xml_directory = os.path.join(dir1_path, dir12)
                    if os.path.isdir(xml_directory):
                        main(directory0, dir1, dir12, current_directory, data_dict, testsite, graph_image, timestamp)
    else:
        if directory2.lower() == 'all':
            base_directory2 = os.path.join(current_directory, 'dat', directory0, directory1)
            if not os.path.isdir(base_directory2):
                print(f"The base directory {base_directory2} does not exist.")
            else:
                for dir2 in os.listdir(base_directory2):
                    xml_directory = os.path.join(base_directory2, dir2)
                    if os.path.isdir(xml_directory):
                        main(directory0, directory1, dir2, current_directory, data_dict, testsite, graph_image, timestamp)
        else:
            main(directory0, directory1, directory2, current_directory, data_dict, testsite, graph_image, timestamp)

    if xlsx_file:
        output_directory = os.path.join(current_directory, 'res',timestamp, 'xlsx')
        os.makedirs(output_directory, exist_ok=True)
        xlsx_file_path = os.path.join(output_directory, f'analysis_result.xlsx')
        save_data_frame(data_dict, xlsx_file_path)
    print('Data analysis is complete.')

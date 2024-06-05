import xml.etree.ElementTree as elemTree
import os
from datetime import datetime

def parse_xml_files(directory0, directory1, directory2, current_directory, teststie):
    base_output_directory = os.path.join(current_directory, 'res', directory0, 'xlsx')
    os.makedirs(base_output_directory, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    xlsx_file_path = os.path.join(base_output_directory, f'{timestamp}.xlsx')

    xml_directory = os.path.join(current_directory, 'dat', directory0, directory1, directory2)
    if not os.path.isdir(xml_directory):
        print(f"The directory {xml_directory} does not exist. Please enter a valid directory path.")
        return None, None, None

    xml_files = []
    for filename in os.listdir(xml_directory):
        if any(filename.endswith(f'{site}.xml') for site in teststie):
            xml_file_path = os.path.join(xml_directory, filename)
            tree = elemTree.parse(xml_file_path)
            root = tree.getroot()
            xml_files.append((filename, root))

    return xml_files, xlsx_file_path, xml_directory

import xml.etree.ElementTree as elemTree
import os

def parse_xml_files(directory1, directory2, current_directory):
    output_directory = os.path.join(current_directory, 'res', 'HY202103', directory1, directory2)
    os.makedirs(output_directory, exist_ok=True)

    csv_file_path = os.path.join(output_directory, f'{directory1}_{directory2}.csv')
    open(csv_file_path, 'w').close()

    xml_directory = os.path.join(current_directory, 'dat', 'HY202103', directory1, directory2)
    if not os.path.isdir(xml_directory):
        print(f"The directory {xml_directory} does not exist. Please enter a valid directory path.")
        return None, None, None

    xml_files = []
    for filename in os.listdir(xml_directory):
        if filename.endswith('LMZC.xml') or filename.endswith('LMZO.xml'):
            xml_file_path = os.path.join(xml_directory, filename)
            tree = elemTree.parse(xml_file_path)
            root = tree.getroot()
            xml_files.append((filename, root))

    return xml_files, csv_file_path, output_directory

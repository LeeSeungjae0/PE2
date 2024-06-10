import xml.etree.ElementTree as elemTree
import os

def parse_xml_files(directory0, directory1, directory2, current_directory, teststie):
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

    return xml_files, xml_directory

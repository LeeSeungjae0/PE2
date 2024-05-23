import os
import sys
from sj3 import process_xml_files

# Get the absolute path to the sj3 module
sj3_module_path = os.path.abspath("sj3.py")
sj3_module_directory = os.path.dirname(sj3_module_path)
sys.path.append(sj3_module_directory)

if __name__ == "__main__":
    # Get the current working directory
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Get the XML directory from user input
    xml_directory1 = input("Enter the path D07, D08, D23, D24, or 'all': ")

    if xml_directory1.lower() == 'all':
        base_directory = os.path.join(current_directory, '..', 'dat', 'HY202103')
        for dir1 in ['D07', 'D08', 'D23', 'D24']:
            dir1_path = os.path.join(base_directory, dir1)
            if not os.path.isdir(dir1_path):
                print(f"The directory {dir1_path} does not exist.")
                continue
            for dir2 in os.listdir(dir1_path):
                xml_directory = os.path.join(dir1_path, dir2)
                if os.path.isdir(xml_directory):
                    process_xml_files(dir1, dir2, current_directory)
    else:
        xml_directory2 = input("Enter the path to the directory containing the XML files: ")
        xml_directory = os.path.join(current_directory, '..', 'dat', 'HY202103', xml_directory1, xml_directory2)
        if not os.path.isdir(xml_directory):
            print(f"The directory {xml_directory} does not exist. Please enter a valid directory path.")
        else:
            process_xml_files(xml_directory1, xml_directory2, current_directory)

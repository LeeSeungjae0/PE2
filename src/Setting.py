import os
from main import main

def setting(directory1, directory2, current_directory):
    base_directory = os.path.join(current_directory, 'dat', 'HY202103')
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
                        main(dir1, dir12, current_directory)
    else:
        if directory2.lower() == 'all':
            base_directory2 = os.path.join(current_directory, 'dat', 'HY202103', directory1)
            if not os.path.isdir(base_directory2):
                print(f"The base directory {base_directory2} does not exist.")
            else:
                for dir2 in os.listdir(base_directory2):
                    xml_directory = os.path.join(base_directory2, dir2)
                    if os.path.isdir(xml_directory):
                        main(directory1, dir2, current_directory)
        else:
            main(directory1, directory2, current_directory)

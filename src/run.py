import os
from main import main

if __name__ == "__main__":
    current_directory = os.getcwd()
    directory1 = input("Enter the path D07, D08, D23, D24, or 'all': ")

    if directory1.lower() == 'all':
        base_directory = os.path.join(current_directory, '..', 'dat', 'HY202103')
        for dir1 in ['D07', 'D08', 'D23', 'D24']:
            dir1_path = os.path.join(base_directory, dir1)
            if not os.path.isdir(dir1_path):
                print(f"The directory {dir1_path} does not exist.")
                continue
            for dir2 in os.listdir(dir1_path):
                xml_directory = os.path.join(dir1_path, dir2)
                if os.path.isdir(xml_directory):
                    main(dir1, dir2, current_directory)
    else:
        directory2 = input("Enter the path to the directory containing the XML files: ")
        main(directory1, directory2, current_directory)

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from setting import setting

lot = 'HY202103'
wafer_number = 'all'                 # 'D07', 'D08', 'D23', 'D24', 'all'
measurement_date = 'all'             # ex) '20190715_190855' or 'all'

testsite = ['LMZO', 'LMZC']          # ['LMZO', 'LMZC']
save_xlsx_file = True                # True or False
save_graph_image = True              # True or False

if __name__ == "__main__":
    current_directory = os.getcwd()
    setting(lot, wafer_number, measurement_date, current_directory, testsite, save_xlsx_file, save_graph_image)
<h1 align="center"> 💻 Group 2's project 💻 </h1> 

---
# 📌Index 
[Who made](#Who-made)<br>
[Background Description](#Background-Description)<br>
[Program Functions](#Program-Functions)<br>
[How to use](#How-to-use)<br>
[Example photos from analysis results](#Example-photos-from-analysis-results)

---
# 📌Who made <a id="Who made"></a> 

- Dongmin Seo :
- Lina Kim : flfsks4242@hanyang.ac.kr
- Seungjae Lee :
- Youngwoong Son :

If you have any questions about this project, you can email us.

---
# 📌Background Description <a id="Background-Description"></a> 
### The goal for the project
-  This project aims to create software that analyzes data using PyCharm.
### The specific issue or challenge it addresses
- If you write the desired lot number and wafer number on run.py , the data is analyzed, and the analyzed graph and data frame are stored to provide you with an xlsx file. 
- You can view the error flag analyzed in the program through the xlsx file, and you can view the graph analyzed with the data as a picture.

---
# 📌Program Functions <a id="Program-Functions"></a> 

This section outlines the main functions and features of the program. It provides a detailed description of what the program does and how it performs its tasks. Key features may include:

- Core functionality
- Additional features
- Any special capabilities or tools integrated into the program

---
# 📌How to use <a id="How-to-use"></a> 
<h3> The code below is an example code for run.py.</h3> 

```python 
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

print('Data analysis is complete.')
```
<h3> 1. You have something to set up before you run 'run.py' .</h3> 

- **You save the data you want to analyze in the 'dat' file.**
- **You have to enter which lot to analyze.** <br>
You need to set the name of the lot you want in the variable 'lot'. In the example above, it is set to 'HY202103'.
- **You set the number of wafer.** <br> 
In the variable 'wafer_number', write the number of wafer you want to analyze. If you want to analyze all the wafers, type 'all'. <br>
The example code for run.py has entered 'all'.
- **You can enter the measurement date.** <br>
In the variable 'measurement_date', you enter the measurement date of the data to be analyzed. If you want to analyze all the data on that wafer, enter 'all'.<br>
The example code is entered with 'all'.
- **You can set up a test site.**<br>
LMZC or LMZO may be entered in the variable 'testsite'. <br> 
The example code contains LMZC and LMZO.
- **You can decide whether to save the data.** <br>
Entering True in the variables 'save_xlsx_file' and 'save_graph_image' saves data frames and graph photos.<br> 
The example code is set to 'True'.

<h3>2. When data analysis is finished by running run.py , 'Data analysis is complete.' will appear.</h3>
<h3>3. You can check the data analyzed in the 'res' file.</h3>

- Data is stored in a file with the date and time of analysis as the name. The analyzed data can be accumulated.<br>
- In the 'res' file, the picture of the graph is stored with the same name as the data file.<br>
- The xlsx file stores the data frame.<br>
- You can check the information in the list below on xlsx. <br>
  >'Lot, Wafer, Mask, TestSite, Name, Date, Script ID, Script Version, Script Owner, Operator, Row, Column, ErrorFlag, Error description, Analysis Wavelength, Rsq of Ref. spectrum (Nth), Max transmission of Ref. spec. (dB), Rsq of IV, I at -1V [A], I at 1V [A], Graph Image'.


<h3> 4. You open the analyzed xlsx file and check the data. </h3>

- You can see the graph by double-clicking the link in 'Graph Image'.<br>
- Each error flag means :  <br>
  - 0 : No error <br>
  - 1: Error in reference spectrum. <br>
  - 2: The IV fitting graph has an error.

---
# 📌Example photos from analysis results <a id="Example-photos-from-analysis-results"></a> 
[Graph Image]()
[xlsx Image]()
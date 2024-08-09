# quickstatandeda

quickstatandeda is a Python library for quick and automatic exploratory data analysis and preliminary statistics analysis. The outputs of the main `edaFeatures()` function are a folder of visualizations and a html file that contains all analyses. This library is built based on mainstream libraries like numpy, pandas, scipy, statsmodel, matplotlib, and seaborn. 

Make sure the data types of your input dataframe are correctly converted! Use `pd.to_datetime()` and `astype()` functions to convert the data type. Here is a simple example:

```python
import pandas as pd
x = pd.read_csv('xxx.csv')

x['string_column'] = x['string_column'].astype('string')
x['int_column'] = x['int_column'].astype('int')
x['float_column'] = x['float_column'].astype('float')
x['date_time_column'] = pd.to_datetime(x['date_time_column'])
```

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install quickstatandeda. If there are some version conflicts, try creating a new virtual environment or use `pip install --upgrade <package_name>` to upgrade the required package. 

## Usage

Here is a simple example: 

```python
import pandas as pd
from quickstatandeda import edaFeatures

x = pd.read_csv('xxx.csv')
y = 'target_column'
id = 'id_column_for_paired_t_test'
save_path = 'path_to_save_the_output_files'
significant_level = 0.05
file_name = 'name_of_the_output_html_file'

edaFeatures(x, y, id, save_path, significant_level, file_name)
```

The outputs are structured as following:

```
├── <file_name>.html
├── _visuals
│   ├── <plot1>.png
│   ├── <plot2>.png
│   ├── <plot3>.png
│   └── ...
```

A visuals folder is created automatically to save all the visuals used in the html output file. 

## Contributing

If you find a bug 🐛 or want to make some major or minor changes, please open an issue in the GitHub repository to discuss. Please feel free to fork the project, make any changes, and submit and pull request if you want to make some major changes. 

## License

[MIT](https://choosealicense.com/licenses/mit/)
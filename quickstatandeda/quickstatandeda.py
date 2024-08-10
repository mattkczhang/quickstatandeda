import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2
import itertools
import os
from scipy.stats import ttest_ind, ttest_rel
from scipy.stats import wilcoxon, mannwhitneyu

sns.set_style('white')
sns.set_context("paper", 
                rc={"font.size":8,
                    "axes.titlesize":10,
                    "axes.labelsize":8,
                    'xtick.labelsize':6,
                    'ytick.labelsize':6,
                    'legend.fontsize':7})   

os.makedirs('visuals', exist_ok=True)

########################
# Supporting Functions #
########################

def getModelResults(model):
    """Takes in the model from statsmodel.api and return the model metrics

    Args:
        model (stats.model.api): input model

    Returns:
        dict: model metrics 
    """
    return {
        'AIC': model.aic,
        'BIC': model.bic,
        'R-squared': model.rsquared,
        'Adjusted R-sqaured': model.rsquared_adj,
        'Log-likelihood': model.llf,
        'P-value': model.pvalues[-1]
    }


def allCombinations(lst):
    """Generate all the combinations of each elements in the list

    Args:
        lst (list): input list

    Returns:
        list: list with all the possible combinations
    """
    output = []
    for i in range(1,len(lst)+1):
        output.extend([j for j in itertools.combinations(lst,i)])
    return output


def findOutliers(df, col, ):
    """Find outlier records bsed on one column/feature

    Args:
        df (pd.DataFrame): input dataframe
        col (str): column name

    Returns:
        pd.DataFrame: outlier records
    """
    records = df.copy()
    median = records[col].median()    
    deviation_from_med = records[col] - median
    mad = deviation_from_med.abs().median()
    records['modified_z_score'] = deviation_from_med/(0.6745*mad)
    return records[records['modified_z_score'].abs() > 3.5]

def findBestModels(input):
    """Find the best models based on different criteria

    Args:
        input (pd.DataFrame): a summary dataframe containing all possible models

    Returns:
        pd.DataFrame: the best models from the input summary dataframe
    """
    output = pd.DataFrame()
    for i in ['AIC','BIC']:
        min_i = min(input[i])
        best_model = input[input[i] == min_i]
        best_model.insert(loc=0, column='Criterion', value='Best '+i)
        output = pd.concat([output, best_model])
    for i in ['R-squared','Adjusted R-sqaured','Log-likelihood']:
        max_i = max(input[i])
        best_model = input[input[i] == max_i]
        best_model.insert(loc=0, column='Criterion', value='Best '+i)
        output = pd.concat([output, best_model])
    return output

def saveInfoToHtml(sum_stats, visuals, regressions, save_path, file_name):
    """Generate an HTML file that showcases the exploratory data analysis
    
    Args:
        sum_stats (dict): dictionary where keys are the section header and values are the summary statistics tables
        visuals (dict): dictionary where keys are the section header and values are the file names of the visuals
        regressions (dict): dictionary where keys are the section header and values are the regression tables
        save_path (str): path to save the HTML file and read the visuals
        file_name (str): file name of the final HTML file
    """
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Quick Statistics and Exploratory Data Analysis Report</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" integrity="sha512-KyZXEAg3QhqLMpG8r+Knujsl5+5hb7O4R0zMQ3f2kZdBc6sP9vO4R0zMQ3f2kZdBc6sP9vO4fVQ8tJaT5fs7iU1z8K6J4t4d1K6Zn6A/FA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
        <style>
            /* Global Styles */
            html, body {{
                margin: 0;
                padding: 0;
                max-width: 100%;
                overflow-x: hidden; /* Prevent horizontal scroll */
                font-family: Arial, sans-serif;
                line-height: 1.6;
            }}

            /* Header Styles */
            header {{
                background-color: #000000; /* Black background */
                color: white;
                padding: 20px 10px;
                text-align: center;
                position: sticky;
                top: 0;
                width: 100%;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                z-index: 1000;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }}

            header .logo {{
                display: flex;
                align-items: center;
            }}

            header .logo img {{
                height: 50px;
                margin-right: 10px;
            }}

            header h1 {{
                margin: 0;
                font-size: 2.5em;
                font-weight: normal;
                display: inline-block;
            }}

            header nav a {{
                color: white;
                text-decoration: none;
                margin: 0 15px;
                font-size: 1.1em;
                transition: color 0.3s ease;
            }}

            header nav a:hover {{
                color: #ddd;
            }}
            
            /* Dynamic Content Styles */
            .plot {{
                text-align: center;
                margin: 10px 0;
            }}
            .plot img {{
                max-width: 100%; /* Ensures the image is responsive */
                height: auto; /* Maintains aspect ratio */
            }}
            .table-responsive {{
                overflow-x: auto;
            }}
            .table {{
                width: 100%;
                max-width: 100%;
                margin-bottom: 1rem;
                background-color: transparent;
            }}

            /* Subsection Title Styles */
            ol {{
                padding-left: 0;
                list-style: none; /* Remove default list styling */
            }}
            ol li {{
                font-size: 1.2em;
                font-weight: bold;
                padding: 10px 0;
                margin-bottom: 15px;
                background-color: #ffffff;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                border-radius: 5px;
                text-align: left;
                padding-left: 20px;
            }}

            /* Button Styles */
            .top-button {{
                margin-top: 20px;
                padding: 10px 15px;
                background-color: #000000; /* Black button */
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }}

            .top-button:hover {{
                background-color: #333333;
            }}

            /* Container Styles */
            .container {{
                max-width: 100%;
                margin: auto;
                padding: 20px;
                box-sizing: border-box;
            }}

            /* Section Styles */
            section {{
                padding: 60px 20px;
                text-align: center;
                background-color: #f4f4f9;
                border-bottom: 1px solid #ddd;
                min-height: 150px; /* Add min-height for visibility */
            }}

            section:nth-child(even) {{
                background-color: #e2e2eb;
            }}

            section h2 {{
                font-size: 2em;
                margin-bottom: 10px;
            }}

            /* Footer Styles */
            footer {{
                background-color: #000000; /* Black background */
                color: white;
                text-align: center;
                padding: 15px;
                position: relative;
                bottom: 0;
                width: 100%;
            }}

            /* Responsive Styles */
            @media (max-width: 768px) {{
                header h1 {{
                    font-size: 2em;
                }}

                header nav a {{
                    font-size: 1em;
                    margin: 0 10px;
                }}

                header {{
                    flex-direction: column;
                    align-items: center;
                }}

                header nav {{
                    margin-top: 10px;
                }}

                section {{
                    padding: 40px 10px;
                }}

                section h2 {{
                    font-size: 1.8em;
                }}

                ol li {{
                    font-size: 1.1em;
                    padding: 8px 15px;
                }}
            }}
        </style>
    </head>
    <body>

        <!-- Header Section -->
        <header id="top">
            <div class="logo">
                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQKvslzr0H2trJJ-BvlhfF8WJYu0n1fwrvjrg&s" alt="Logo">
                <h1>Preliminary Study</h1>
            </div>
            <nav>
                <a href="#summary"><i class="fas fa-chart-pie"></i> Summary Statistics</a>
                <a href="#eda"><i class="fas fa-chart-line"></i> Exploratory Data Analysis</a>
                <a href="#regression"><i class="fas fa-chart-bar"></i> Preliminary Regression Analysis</a>
            </nav>
        </header>

        <!-- Main Content -->
        <main class="container">
            <section id="summary">
                <h2>Summary Statistics</h2>
                <ol>
                """ 
    for i in sum_stats.keys():
        html_template += f"""
                    <li>{i}
                    <div class="table-responsive">
                        {sum_stats[i].to_html(justify='center', classes='table table-striped', border=2)}
                    </div>
                    </li>
            """
    html_template += """
                </ol>
                <a href="#top" class="top-button">Back to Top</a>
            </section>

            <section id="eda">
                <h2>Exploratory Data Analysis</h2>
                <ol>
                """
    for i in visuals.keys():
        html_template += f"""
                        <li>{i}
                        <div class="plot">
                            <img src="{save_path}visuals/{visuals[i]}" alt="EDA Visual" class="img-fluid">
                        </div>
                        </li>
                    """
    html_template += """
                </ol>  
                <a href="#top" class="top-button">Back to Top</a>
            </section>

            <section id="regression">
                <h2>Preliminary Regression Analysis</h2>
                <ol>
                """
    if len(regressions.keys()) > 0:
        for i in regressions.keys():
            html_template += f"""
                            <li>{i}
                            <div class="table-responsive">
                                {regressions[i].to_html(index=False, justify='center', classes='table table-striped', border=2)}
                            </div>
                            </li>
                        """
    else:
        html_template += f"""
                        <p> Target feature y is not specified
                        </p>
                                """
    html_template += """
                </ol>  
                <a href="#top" class="top-button">Back to Top</a>         
            </section>
        </main>

        <!-- Footer -->
        <footer>
            <p>&copy; Quick Statistics and Exploratory Data Analysis Report</p>
        </footer>

    </body>
    </html>
    """
    
    # Save HTML content to file
    with open(save_path + file_name + '.html', 'w') as file:
        file.write(html_template)

##################
# Main Functions #
##################

def forwardSelection(x, y):
    """Forward selection of linear regression

    Args:
        x (pd.DataFrame): features
        y (pd.Series): target

    Returns:
        pd.DataFrame: table that showcases the steps of forward selection
    """
    if len(x) != len(y):
        print('The number of rows of features and target is not matched. Check out their length!')
        return

    predictors = {}
    for i in x.columns.tolist():
        predictors['+'+i] = x[i]
    
    trainData = pd.DataFrame(y)
    trainData.insert(0, 'intercept', 1.0)
    trainData = trainData[['intercept']]

    model = sm.OLS(y, trainData[['intercept']], missing='drop').fit()
    results0 = getModelResults(model)

    step_summary = pd.DataFrame([[
        'Intercept', results0['AIC'], results0['BIC'], results0['R-squared'], 
        results0['Adjusted R-sqaured'], results0['Log-likelihood'],results0['P-value'], np.nan]])
    
    max_p = 1
    model0 = model
    name = ''

    while predictors != {}:
        for key in predictors:
            trainData_temp = trainData.join(predictors[key])
            model = sm.OLS(y, trainData_temp, missing='drop').fit()
            results1 = getModelResults(model)

            if np.isnan(results1['Adjusted R-sqaured']) or np.isnan(results1['P-value']):
                step_summary.reset_index(inplace=True)
                step_summary.columns = ['Step', 'Predictor Entered', 'AIC', 'BIC', 'R-squared', 'Adjusted R-sqaured',
                       'Log-likelihood', 'P-value', 'F-test significance'] 
                return step_summary

            f_test_p = model.compare_f_test(model0)[1]
            # print(key, f_test_p)       
            if f_test_p < max_p:
                max_p = f_test_p
                name = key
                results = results1
        if max_p > 0.05:
            break

        step_summary = pd.concat([step_summary, pd.DataFrame([[name, results['AIC'], results['BIC'], results['R-squared'], 
                                                                   results['Adjusted R-sqaured'], results['Log-likelihood'],results['P-value'], max_p]])], ignore_index = True)
        trainData = trainData.join(predictors[name])
        predictors.pop(name)
        max_p = 1
        model0 = model

    step_summary.reset_index(inplace=True)
    step_summary.columns = ['Step', 'Predictor Entered', 'AIC', 'BIC', 'R-squared', 'Adjusted R-sqaured',
                       'Log-likelihood', 'P-value', 'F-test significance']    
    return step_summary


def backwardSelection(x, y):
    """Backward selection of linear regression

    Args:
        x (pd.DataFrame): features
        y (pd.Series): target

    Returns:
        pd.DataFrame: table that showcases the steps of backward selection
    """
    if len(x) != len(y):
        print('The number of rows of features and target is not matched. Check out their length!')
        return

    predictors = {}
    for i in x.columns.tolist():
        predictors[i] = x[i]
    
    trainData = pd.DataFrame(x)
    trainData.insert(0, 'intercept', 1.0)

    model = sm.OLS(y, trainData, missing='drop').fit()
    results0 = getModelResults(model)

    step_summary = pd.DataFrame([[
        'Full Model', results0['AIC'], results0['BIC'], results0['R-squared'], 
        results0['Adjusted R-sqaured'], results0['Log-likelihood'],results0['P-value'], np.nan]])
    
    max_p = 0
    model0 = model
    name = ''

    while predictors != {}:
        for key in predictors:
            trainData_temp = trainData.drop(columns=[key])
            model = sm.OLS(y, trainData_temp, missing='drop').fit()
            results1 = getModelResults(model)

            if np.isnan(results1['Adjusted R-sqaured']) or np.isnan(results1['P-value']):
                step_summary.reset_index(inplace=True)
                step_summary.columns = ['Step', 'Predictor Entered', 'AIC', 'BIC', 'R-squared', 'Adjusted R-sqaured',
                       'Log-likelihood', 'P-value', 'F-test significance']    
                return step_summary

            f_test_p = model0.compare_f_test(model)[1]
            # print(key, f_test_p)       
            if f_test_p > max_p:
                max_p = f_test_p
                name = key
                results = results1
        if max_p > 0.05:
            break

        step_summary = pd.concat([step_summary, pd.DataFrame([['-'+name, results['AIC'], results['BIC'], results['R-squared'], 
                                                                   results['Adjusted R-sqaured'], results['Log-likelihood'],results['P-value'], max_p]])], ignore_index = True)
        trainData = trainData.drop(columns=[name])
        predictors.pop(name)
        max_p = 0
        model0 = model

    step_summary.reset_index(inplace=True)
    step_summary.columns = ['Step', 'Predictor Entered', 'AIC', 'BIC', 'R-squared', 'Adjusted R-sqaured',
                       'Log-likelihood', 'P-value', 'F-test significance']    
    return step_summary


def allPossibleSelection(x, y):
    """All possible selection of linear regression

    Args:
        x (pd.DataFrame): features
        y (pd.Series): target

    Returns:
        pd.DataFrame: table that showcases all the possible combinations of features and the corresponding model metrics
    """
    if len(x) != len(y):
        print('The number of rows of features and target is not matched. Check out their length!')
        return

    combs = allCombinations(x.columns)
    step_summary = pd.DataFrame()
    for i in combs:
        trainData = pd.DataFrame(x[list(i)])
        trainData.insert(0, 'intercept', 1.0)
        model = sm.OLS(y, trainData, missing='drop').fit()
        results = getModelResults(model)
        step_summary = pd.concat([step_summary, 
                                  pd.DataFrame([[str(list(i)).replace('[','').replace(']','').replace(', ', ' + ').replace("'",""), 
                                                 results['AIC'], results['BIC'], results['R-squared'], 
                                                 results['Adjusted R-sqaured'], results['Log-likelihood'],
                                                 results['P-value']]])], ignore_index=True)

    step_summary.reset_index(inplace=True)
    step_summary.columns = ['Index', 'Predictors', 'AIC', 'BIC', 'R-squared', 'Adjusted R-sqaured',
                       'Log-likelihood', 'P-value']    
    
    return step_summary


def edaFeatures(x, y = None, id=None, save_path = '', significant_level = 0.05, file_name = 'EDA'):
    """Generate a HTML based exploratory data analysis report

    Args:
        x (pd.DataFrame): features (can include target feature)
        y (pd.Series|str, optional): target feature. Defaults to None.
        id (str, optional): observation identifiers for paired t test. Defaults to None.
        save_path (str, optional): path to save the visuals and the HTML report. Defaults to ''.
        significant_level (float, optional): significant level for t test. Defaults to 0.05.
        file_name (str, optional): file name of the HTML report. Defaults to 'EDA'.
    """
    if type(y) == pd.Series and len(x) != len(y):
            print('The number of rows of features and target is not matched. Check out their length!')
            return

    # prepare the variables
    if save_path != '' and save_path[-1] != '/':
        save_path += '/'
    numeric_features = x.select_dtypes(exclude=['object', 'datetime64[ns]']).columns.tolist()
    categorical_features = x.dtypes[x.dtypes=='object'].index.tolist()
    datetime_features = x.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    if id != None:
        if id in numeric_features:
            numeric_features.remove(id)
        elif id in categorical_features:
            categorical_features.remove(id)
    num_num = len(numeric_features)
    num_cat = len(categorical_features)
    num_datetime = len(datetime_features)

    sum_stats = {}
    visuals = {}
    regressions = {}

    # summary stat
    if num_num > 0:
        Shapiro_Wilk_results = []
        Anderson_Darling_results = []
        nan_count = []
        datatypes = []
        for i in numeric_features:
            Shapiro_Wilk_results.append(sc.stats.shapiro(x[i], nan_policy='omit').pvalue)
            ad_test = sc.stats.anderson(x[i][~np.isnan(x[i])])
            if ad_test.statistic <= ad_test.critical_values[np.where(ad_test.significance_level==significant_level*100)[0][0]]:
                ad_res = 'is from a normal distribution at ' + str(significant_level)
            else:
                ad_res = 'is not from a normal distribution at ' + str(significant_level)
            Anderson_Darling_results.append(ad_res)
            nan_count.append(x[i].isna().sum())
            datatypes.append(str(x[i].dtype))

        sum_stat_numeric = x[numeric_features].describe()
        sum_stat_numeric.loc['number of nan'] = nan_count
        sum_stat_numeric.loc['Shprio Wilk p value'] = Shapiro_Wilk_results
        sum_stat_numeric.loc['Anderson Darling result'] = Anderson_Darling_results
        sum_stat_numeric.loc['data type'] = datatypes
        # print(sum_stat_numeric.to_markdown(tablefmt="grid"))
        sum_stats['Numeric Features'] = sum_stat_numeric
    if num_cat > 0:
        unique_values = []
        nan_count = []
        datatypes = []
        for i in categorical_features:
            unique_values.append(str(dict(x[i].value_counts())).replace('{','').replace('}',''))
            nan_count.append(x[i].isna().sum())
            datatypes.append(str(x[i].dtype))
        sum_stat_categorical = x[categorical_features].describe()
        sum_stat_categorical.loc['number of nan'] = nan_count
        sum_stat_categorical.loc['unique values'] = unique_values
        sum_stat_categorical.loc['data type'] = datatypes
        # print(sum_stat_categorical.to_markdown(tablefmt="grid"))
        sum_stats['Categorical Features'] = sum_stat_categorical
    if num_datetime > 0:
        max_time = []
        min_time = []
        time_diff = []
        datatypes = []
        nan_count = []
        for i in datetime_features:
            max_time.append(max(x[i]))
            min_time.append(min(x[i]))
            time_diff.append(str(max(x[i])-min(x[i])))
            nan_count.append(x[i].isna().sum())
            datatypes.append(str(x[i].dtype))
        sum_stat_datetime = x[datetime_features].describe()[:2]
        sum_stat_datetime.loc['latest date time'] = max_time
        sum_stat_datetime.loc['earliest date time'] = max_time
        sum_stat_datetime.loc['date time range'] = time_diff
        sum_stat_datetime.loc['number of nan'] = nan_count
        sum_stat_datetime.loc['data type'] = datatypes
        sum_stats['Date Time Features'] = sum_stat_datetime

    # correlation coefficient matrix
    if num_num > 0:
        corr_matrix = x[numeric_features].corr()
        # print(corr_matrix.to_markdown(tablefmt="grid"))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="crest")
        plt.savefig(save_path+'visuals/correlation_heatmap.png', dpi=500)
        visuals['Heatmap of Correlation Matrix'] = 'correlation_heatmap.png'
        plt.clf()
        # sum_stats['Correlation Coefficient Matrix'] = corr_matrix

    # outliers detection
    for i in numeric_features:
        if len(x[i].value_counts()) > 2:
            outlierRecords = findOutliers(x, i)
            if len(outlierRecords) > 0:
                sum_stats['Outlier Records of Feature '+i] = outlierRecords

    # missing value heatmap
    sns.heatmap(x.isnull(), cbar=False)
    plt.savefig(save_path+'visuals/missing_value_heatmap.png', dpi=500)
    visuals['Heatmap of Missing Values'] = 'missing_value_heatmap.png'
    plt.clf()

    if num_num > 0:
        # qq plot
        for i in numeric_features:
            if len(x[i].value_counts()) > 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sc.stats.probplot(x[i], dist="norm", plot=ax, fit=False)
                ax.get_lines()[0].set_markerfacecolor('black')  
                ax.get_lines()[0].set_markeredgecolor('black') 

                # Add 45-degree reference line
                ax.plot([x[i].min(), x[i].max()], [x[i].min(), x[i].max()], 'k--', lw=2)

                # Customize plot appearance
                ax.set_xlabel('Theoretical Quantiles')
                ax.set_ylabel('Sample Quantiles')

                plt.savefig(save_path+'visuals/'+i+'_qqplot.png', dpi=500)
                visuals['Q-Q Plot of Feature '+i] = i+'_qqplot.png'
                plt.clf()

        # lineplot 
        if num_datetime > 0:
            c = 0
            h = 0
            m = 0
            ncol_per_row = 2
            row_multiplier = num_datetime // ncol_per_row + 1
            fig, axs = plt.subplots(num_num*row_multiplier, ncol_per_row, figsize=(10*ncol_per_row, 5*num_num*row_multiplier))
            while c < num_num*row_multiplier and m < num_datetime:
                if num_num*row_multiplier == 1:
                    ax_temp = axs[h]
                elif num_datetime == 1:
                    ax_temp = axs[c,h]
                else:
                    ax_temp = axs[c,h]
                sns.lineplot(data=x, x=datetime_features[m], 
                                y=np.repeat(numeric_features,row_multiplier).tolist()[c], 
                                color = '#49acf2', ax=ax_temp)
                sns.scatterplot(data=x, x=datetime_features[m], 
                                y=np.repeat(numeric_features,row_multiplier).tolist()[c], 
                                color = '#ebac59', ax=ax_temp)
                sns.despine(right = True)
                if h < ncol_per_row-1:
                    if m < num_datetime-1:
                        h += 1
                        m += 1
                    elif m == num_datetime-1:
                        m = 0
                        h = 0
                        c += 1
                else:
                    if m < num_datetime-1:
                        h = 0
                        c += 1
                        m += 1
                    elif m == num_datetime-1:
                        m = 0
                        h = 0
                        c += 1
            row = row_multiplier-1
            col = num_datetime % ncol_per_row
            while row < num_num*row_multiplier:
                for i in range(col, ncol_per_row):
                    if num_num*row_multiplier == 1:
                        fig.delaxes(axs[i])
                    else:
                        fig.delaxes(axs[row,i])
                row += row_multiplier
            plt.savefig(save_path+'visuals/lineplot_all_numeric_vs_datetime.png', dpi=500)
            visuals['Lineplot On All Numeric Features Paired with Date Time Features'] = 'lineplot_all_numeric_vs_datetime.png'
            plt.clf()

        # clustermap
        if num_num > 1:
            sns.clustermap(x[numeric_features].dropna())
            plt.savefig(save_path+'visuals/cluster_map.png', dpi=500)
            visuals['Cluster Map On All Numeric Features'] = 'cluster_map.png'
            plt.clf()

        # pairplot
        sns.pairplot(x[numeric_features], kind='reg',
                    plot_kws={'line_kws':{'color':'#82ad32'},
                            'scatter_kws': {'alpha': 0.5, 's':3,
                                            'color': '#197805'}},
                    diag_kws= {'color': '#82ad32'})
        plt.savefig(save_path+'visuals/pairplot_numeric.png', dpi=500)
        visuals['Pairplot On All Numeric Features'] = 'pairplot_numeric.png'
        plt.clf()

    if num_cat > 0:
        # countplot
        if num_cat > 1:
            c = 0
            h = 0
            fig, axs = plt.subplots(num_cat,num_cat-1, figsize=(3*num_cat, 6*(num_cat-1)))
            for ax in axs.flatten():
                if c == h:
                    h+=1
                sns.countplot(data=x, x=categorical_features[c], 
                            hue=categorical_features[h], ax=ax)
                if h < num_cat-1:
                    h += 1
                else:
                    h = 0
                    c += 1 
            plt.savefig(save_path+'visuals/countplot_categorical.png', dpi=500)
        else:
            sns.countplot(data=x, x=categorical_features[0])
            plt.savefig(save_path+'visuals/countplot_categorical.png', dpi=500)
        visuals['Countplot On All Categorical Features'] = 'countplot_categorical.png'
        plt.clf()

        # boxplot & stripplot
        c = 0
        h = 0
        m = 0
        ncol_per_row = 3
        row_multiplier = num_num // ncol_per_row + 1
        fig, axs = plt.subplots(num_cat*row_multiplier, ncol_per_row, figsize=(10*ncol_per_row, 10*num_cat*row_multiplier))
        while c < num_cat*row_multiplier and m < num_num:
            if num_cat*row_multiplier == 1:
                ax_temp = axs[h]
            elif num_num == 1:
                ax_temp = axs[c,h]
            else:
                ax_temp = axs[c,h]                    
            sns.boxplot(data=x, x=np.repeat(categorical_features,row_multiplier).tolist()[c], 
                                    y=numeric_features[m], color = '#49acf2', ax=ax_temp)
            sns.stripplot(data=x, x=np.repeat(categorical_features,row_multiplier).tolist()[c], 
                                    y=numeric_features[m], color = '#ebac59', ax=ax_temp)
            sns.despine(right = True)
            if h < ncol_per_row-1:
                if m < num_num-1:
                    h += 1
                    m += 1
                elif m == num_num-1:
                    h = 0
                    m = 0
                    c += 1
            else:
                if m < num_num-1:
                    h = 0
                    m += 1
                    c += 1
                elif m == num_num-1:
                    h = 0
                    m = 0
                    c += 1
        row = row_multiplier-1
        col = num_num % ncol_per_row
        while row < num_cat*row_multiplier:
            for i in range(col, ncol_per_row):
                if num_cat*row_multiplier == 1:
                    fig.delaxes(axs[i])
                else:
                    fig.delaxes(axs[row,i])
            row += row_multiplier
        plt.savefig(save_path+'visuals/boxplot_all_numeric_vs_categorical.png', dpi=500)
        visuals['Boxplot On All Categorical Features Paired with Numeric Features'] = 'boxplot_all_numeric_vs_categorical.png'
        plt.clf()

    # t test 
    # paired
    if id != None:
        paired_t_test_parametric = pd.DataFrame(columns=numeric_features)
        paired_t_test_nonparametric = pd.DataFrame(columns=numeric_features)
        for i in categorical_features:
            if len(x[i].unique()) == 2:
                row_para = []
                row_nonpara = []
                add_row = True
                for j in numeric_features:
                    tab_temp = x.pivot(index=id,columns=i,values=j).reset_index()
                    if len(tab_temp.dropna()) > 0:
                        col_temp = tab_temp.columns
                        row_para.append(ttest_rel(tab_temp[col_temp[1]], tab_temp[col_temp[1]],
                                                nan_policy = 'omit').pvalue)
                        row_nonpara.append(wilcoxon(tab_temp[col_temp[2]], tab_temp[col_temp[1]],
                                                    nan_policy = 'omit').pvalue)
                    else:
                        add_row = False
                if add_row:
                    paired_t_test_parametric.loc[i] = row_para
                    paired_t_test_nonparametric[i] = row_nonpara
        sum_stats['Parametric Paired T Test'] = paired_t_test_parametric
        sum_stats['Non-parametric Paired T Test'] = paired_t_test_nonparametric

    #two-sample
    if len(categorical_features) > 0:
        two_sample_t_test_parametric  = pd.DataFrame(columns=numeric_features)
        two_sample_t_test_nonparametric = pd.DataFrame(columns=numeric_features)
        for i in categorical_features:
            if len(x[i].unique()) == 2:
                row_para = []
                row_nonpara = []
                for j in numeric_features:
                    unique_values = x[i].unique()
                    row_para.append(ttest_ind(x[x[i]==unique_values[0]][j], 
                                            x[x[i]==unique_values[0]][j],
                                            nan_policy = 'omit').pvalue)
                    row_nonpara.append(mannwhitneyu(x[x[i]==unique_values[0]][j], 
                                                    x[x[i]==unique_values[0]][j],
                                                    nan_policy = 'omit').pvalue)
                two_sample_t_test_parametric.loc[i] = row_para
                two_sample_t_test_nonparametric.loc[i] = row_nonpara
        if len(two_sample_t_test_parametric) > 0:
            sum_stats['Parametric Two-Sample T Test'] = two_sample_t_test_parametric
        if len(two_sample_t_test_nonparametric) > 0:
            sum_stats['Non-parametric Two-Sample T Test'] = two_sample_t_test_nonparametric

    # regression
    if y != None:
        target = None
        if type(y) == str and y in x.columns:
            target = x.dropna()[y]
        elif type(y) == pd.core.series.Series:
            target = y
        if type(target) == pd.core.series.Series:
            forwardSelection_tab = forwardSelection(x.dropna()[numeric_features].copy().drop(columns=target.name),target)
            backwardSelection_tab = backwardSelection(x.dropna()[numeric_features].copy().drop(columns=target.name),target)
            allPossibleSelection_tab = allPossibleSelection(x.dropna()[numeric_features].copy().drop(columns=target.name), target)
            if forwardSelection_tab is not None:
                regressions['Forward Selection'] = forwardSelection_tab
            if backwardSelection_tab is not None:
                regressions['Backward Selection'] = backwardSelection_tab.drop(columns=['P-value'])
            if allPossibleSelection_tab is not None:
                bestModel_tab = findBestModels(allPossibleSelection_tab)
                regressions['All Possible Selection'] = allPossibleSelection_tab.drop(columns=['P-value'])
                regressions['Best Models'] = bestModel_tab.drop(columns=['P-value','Index'])

    saveInfoToHtml(sum_stats, visuals, regressions, save_path, file_name)


import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2
import itertools
import os
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel
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
    """Generate a HTML file that showcases the exploratory data analysis
    
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
        <title>Large Table and Visuals</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
        <style>
            body {{
                margin: 20px;
            }}
            .container {{
                max-width: 1200px;
                margin: auto;
            }}
            .plot {{
                text-align: center;
                margin: 10px 0;
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
            ol li{{
                font-size:20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Exploratory Data Analysis</h1>
            <h2>Summary Statistics</h2>
            <ol>
            """ 
    for i in sum_stats.keys():
        html_template += """
                <li>""" + i + f"""</li>
                <div class="table-responsive">
                    {sum_stats[i].to_html(justify = 'center', classes='table table-striped', border=2)}
                </div>
        """
    html_template += """
            </ol>
            <h2>Visualizations of Correlation, Missing Values and Univariate and Bivairate Relationship </h2>
            <ol>
    """
    for i in visuals.keys():
        html_template += """
            <li>""" + i + """</li>
            <div class="plot">
                <img src=""""" + save_path + 'visuals/' + visuals[i] + """ alt="Correlation Heatmap" class="img-fluid">
            </div>
        """
    html_template += """
            </ol>
            <h2>Preliminary Regression Analysis </h2>
            <ol>
    """
    for i in regressions.keys():
        html_template += """
            <li>""" + i + f"""</li>
            <div class="table-responsive">
                {regressions[i].to_html(index=False, justify = 'center', classes='table table-striped', border=2)}
            </div>
        """
    html_template += """
            </ol>
        </div>
    </body>
    </html>
    """

    # Save HTML content to file
    with open(save_path+file_name+'.html', 'w') as file:
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
    if id != None:
        if id in numeric_features:
            numeric_features.remove(id)
        elif id in categorical_features:
            categorical_features.remove(id)
    num_num = len(numeric_features)
    num_cat = len(categorical_features)

    sum_stats = {}
    visuals = {}
    regressions = {}

    # summary stat
    if num_num > 0:
        Shapiro_Wilk_results = []
        Anderson_Darling_results = []
        nan_count = []
        for i in numeric_features:
            Shapiro_Wilk_results.append(sc.stats.shapiro(x[i], nan_policy='omit').pvalue)
            ad_test = sc.stats.anderson(x[i][~np.isnan(x[i])])
            if ad_test.statistic <= ad_test.critical_values[np.where(ad_test.significance_level==significant_level*100)[0][0]]:
                ad_res = 'is from a normal distribution at ' + str(significant_level)
            else:
                ad_res = 'is not from a normal distribution at ' + str(significant_level)
            Anderson_Darling_results.append(ad_res)
            nan_count.append(x[i].isna().sum())

        sum_stat_numeric = x[numeric_features].describe()
        sum_stat_numeric.loc['number of nan'] = nan_count
        sum_stat_numeric.loc['Shprio Wilk p value'] = Shapiro_Wilk_results
        sum_stat_numeric.loc['Anderson Darling result'] = Anderson_Darling_results
        # print(sum_stat_numeric.to_markdown(tablefmt="grid"))
        sum_stats['Numeric Features'] = sum_stat_numeric
    if num_cat > 0:
        unique_values = []
        nan_count = []
        for i in categorical_features:
            unique_values.append(str(dict(x[i].value_counts())).replace('{','').replace('}',''))
            nan_count.append(x[i].isna().sum())
        sum_stat_categorical = x[categorical_features].describe()
        sum_stat_categorical.loc['number of nan'] = nan_count
        sum_stat_categorical.loc['unique values'] = unique_values
        # print(sum_stat_categorical.to_markdown(tablefmt="grid"))
        sum_stats['Categorical Features'] = sum_stat_categorical

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
                visuals['Q-Q plot of Feature '+i] = i+'_qqplot.png'
                plt.clf()

        # pairplot
        sns.pairplot(x[numeric_features], kind='reg',
                    plot_kws={'line_kws':{'color':'#82ad32'},
                            'scatter_kws': {'alpha': 0.5, 's':3,
                                            'color': '#197805'}},
                    diag_kws= {'color': '#82ad32'})
        plt.savefig(save_path+'visuals/pairplot_numeric.png', dpi=500)
        visuals['Pairplot On All numeric Features'] = 'pairplot_numeric.png'
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
        ncol_per_row = 4
        row_multiplier = num_num // ncol_per_row + 1
        fig, axs = plt.subplots(num_cat*row_multiplier, ncol_per_row, figsize=(10*num_cat, 2*(num_num)))
        while c < num_cat*row_multiplier and m < num_num:
            ax_temp = axs[c,h]
            sns.boxplot(data=x, x=np.repeat(categorical_features,row_multiplier).tolist()[c], 
                                    y=numeric_features[m], color = '#49acf2', ax=ax_temp)
            sns.stripplot(data=x, x=np.repeat(categorical_features,row_multiplier).tolist()[c], 
                                    y=numeric_features[m], color = '#ebac59', ax=ax_temp)
            sns.despine(right = True)
            if h < ncol_per_row-1:
                h += 1
                m += 1
            else:
                h = 0
                c += 1
                m += 1
            if m == num_num-1:
                m = 0
                h = 0
                c += 1
        row = row_multiplier-1
        col = ncol_per_row - num_num % ncol_per_row
        while row <= num_cat*row_multiplier:
            for i in range(col, ncol_per_row):
                fig.delaxes(axs[row,i])
            row += row_multiplier
        plt.savefig(save_path+'visuals/boxplot_all_numeric_vs_categorical.png', dpi=500)
        visuals['Boxplot On All Categorical Features Paired with numeric Features'] = 'boxplot_all_numeric_vs_categorical.png'
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
            target = x[y]
        elif type(y) == pd.core.series.Series:
            target = y
        if type(target) == pd.core.series.Series:
            forwardSelection_tab = forwardSelection(x[numeric_features].copy().drop(columns=target.name),target)
            backwardSelection_tab = backwardSelection(x[numeric_features].copy().drop(columns=target.name),target)
            allPossibleSelection_tab = allPossibleSelection(x[numeric_features].copy().drop(columns=target.name), target)
            bestModel_tab = findBestModels(allPossibleSelection_tab)
            regressions['Forward Selection'] = forwardSelection_tab
            regressions['Backward Selection'] = backwardSelection_tab.drop(columns=['P-value'])
            regressions['All Possible Selection'] = allPossibleSelection_tab.drop(columns=['P-value'])
            regressions['Best Models'] = bestModel_tab.drop(columns=['P-value','Index'])

    saveInfoToHtml(sum_stats, visuals, regressions, save_path, file_name)


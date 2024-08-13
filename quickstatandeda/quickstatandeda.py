import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import ttest_ind, ttest_rel
from scipy.stats import wilcoxon, mannwhitneyu
from autoregression import *

sns.set_style('white')
sns.set_context("paper", 
                rc={"font.size":8,
                    "axes.titlesize":10,
                    "axes.labelsize":8,
                    'xtick.labelsize':6,
                    'ytick.labelsize':6,
                    'legend.fontsize':7})   

def edaFeatures(x : pd.DataFrame, y : str = None, 
                id : str =None, save_path : str = '', 
                significant_level : float = 0.05, file_name : str = 'EDA'):
    """Generate a HTML based exploratory data analysis report

    Args:
        x (pd.DataFrame): features (can include target feature)
        y (pd.Series|str, optional): target feature. Defaults to None.
        id (str, optional): observation identifiers for paired t test. Defaults to None.
        save_path (str, optional): path to save the visuals and the HTML report. Defaults to ''.
        significant_level (float, optional): significant level for t test. Defaults to 0.05.
        file_name (str, optional): file name of the HTML report. Defaults to 'EDA'.
    """

    os.makedirs('visuals', exist_ok=True)

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
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="crest")
        plt.savefig(save_path+'visuals/correlation_heatmap.png', dpi=500)
        visuals['Heatmap of Correlation Matrix'] = 'correlation_heatmap.png'
        plt.clf()

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
        target = x.dropna()[y]
        # if type(y) == str and y in x.columns:
        #     target = x.dropna()[y]
        # elif type(y) == pd.core.series.Series and len(y) == len(x.dropna()):
        #     target = y
        # if type(target) == pd.core.series.Series:
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


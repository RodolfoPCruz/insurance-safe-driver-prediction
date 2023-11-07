import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.pyplot as plt


from scipy.stats import chi2_contingency

def chi_squared_test_cramers_features_output(data,alpha):
    
    '''
    This function calculates the chi-squared test between each categorical feature in the dataframe and the 
    output variable (column named 'target'). The test aims to determine whether they are independent or not. 
    The level of association between each variable and the 'target ' column, the cramer'v  value and the cramer'v degrees 
    of freedom are calculated.
    
    data is a dataframe containing categorial features and the output variable
    alpha is the significance for the test of hypothesis
    
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    The output is a dataframe with the following columns:
    
    chi2       - calculated values of the chi squared test
    p          - p value
    dof        - degrees of freedom
    conclusion - the result of the chi-squared test indicating whether the variables are independent
    cramers_v  - cramer'v value
    cramers_df - cramers'v degrees of freedom
    '''
    
    features=data.columns
    features=features.delete(np.argwhere(features=='target'))
    results={'chi2':[],'p':[],'dof':[],'conclusion':[],'cramers_v':[],'cramers_df':[]}
    for i in features:
        x=data[[i,'target']].copy()
        x.dropna(inplace=True)
        columns=x.columns
        x=pd.crosstab(x[columns[0]],x[columns[1]])
        chi2, p, dof, con_table = chi2_contingency(x)
        n=x.values.sum()
        r, k = x.shape
        phi2 = chi2 / n
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        cramers_dof=min(x.shape)-1
        cramers_v=np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
        results['chi2']      =results['chi2']+[chi2]
        results['p']         =results['p']   +[p]
        results['dof']       =results['dof']+[dof]
        results['cramers_v'] =results['cramers_v']+[cramers_v]
        results['cramers_df']=results['cramers_df']+[cramers_dof]

        if p<alpha/2:
            conclusion='Dependente'
        else:
            conclusion='Independente'
        results['conclusion']=results['conclusion']+[conclusion]
    results=pd.DataFrame(results,index=features)
    results.sort_values(by=['cramers_v'],ascending=False,inplace=True)
    return results
        

def association_categorical_features(data):
    '''
    data is a dataframe containing categorical features and an output variable that is also categorical
    
    The function outputs three matrices. The first one contains the p values for the chi-squared tests 
    between all two pairs of features. The second comprises Cramer's values to measure the power of 
    association between the pairs of features, and the last one includes  Cramer's degrees of freedom.
    '''
    columns=data.columns
    n_columns=len(data.columns)
    cramers_matrix     =np.ones(shape=[n_columns,n_columns])
    chi_squared_matrix=np.ones(shape=[n_columns,n_columns])
    cramers_df_matrix =np.ones(shape=[n_columns,n_columns])
    #itertools.combinations - it is an alternative to using two for loops
    for i in range(n_columns):
        for j in range(i+1,n_columns):
            x=data[[columns[i],columns[j]]].copy()
            x.dropna(inplace=True)
            names=x.columns
            x=pd.crosstab(x[names[0]],x[names[1]])
            chi2, p, dof, con_table = chi2_contingency(x)
            chi_squared_matrix[i,j]=p
            chi_squared_matrix[j,i]=p

            n=x.values.sum()
            r, k = x.shape
            phi2 = chi2 / n
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            cramers_dof=min(x.shape)-1
            cramers_v=np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
            cramers_matrix[i,j]=cramers_v
            cramers_matrix[j,i]=cramers_v
            cramers_df_matrix[i,j]=cramers_dof
            cramers_df_matrix[j,j]=cramers_dof
    chi_squared_matrix=pd.DataFrame(chi_squared_matrix,index=columns,columns=columns)
    cramers_df_matrix=pd.DataFrame(cramers_df_matrix,index=columns,columns=columns)
    cramers_matrix=pd.DataFrame(cramers_matrix,index=columns,columns=columns)
    return chi_squared_matrix,cramers_matrix,cramers_df_matrix


def heat_map_significance(confusion_matrix,alpha,size=[12,8]):
    #Plotting the heat map for a specified level of significance. The calculated p values are compared to the 
    #significance level to determine whether the features are independent.
    #alpha is the required significance
    cmap = sns.mpl_palette("Set1", 2)
    legend_handles = [Patch(color=cmap[True], label='Dependente'),  # red
                  Patch(color=cmap[False], label='Independente')]  # green
    plt.legend(handles=legend_handles, ncol=2, bbox_to_anchor=[0.5, 1.02], loc='lower center', 
           fontsize=8, handlelength=.8)
    sns.heatmap(confusion_matrix<alpha/2,cmap=cmap,cbar=False)
    
def cramers_level(df,cr):
    '''
    The functions converts the cramers values and 
    cramers degrees of freedom to one of the four levels of association between the features: 
    negligible,small,medium,and large (Cohen's rules of thumb)
    
    df -degrees of freedom
    cr -crames's value 
    '''
    if df==1:
        if cr<=0.1:
            i='negligenciavel'
        elif 0.1<cr<=0.3:
            i='pequeno'
        elif 0.3<cr<=0.5:
            i='medio'
        else:
            i='grande'

    if df==2:
        if cr<=0.07:
            i='negligenciavel'
        elif 0.07<cr<=0.21:
            i='pequeno'
        elif 0.21<cr<=0.35:
            i='medio'
        else:
            i='grande'

    if df==3:
        if cr<=0.06:
            i='negligenciavel'
        elif 0.06<cr<=0.17:
            i='pequeno'
        elif 0.17<cr<=0.29:
            i='medio'
        else:
            i='grande'

    if df==4:
        if cr<=0.05:
            i='negligenciavel'
        elif 0.05<cr<=0.15:
            i='pequeno'
        elif 0.15<cr<=0.25:
            i='medio'
        else:
            i='grande'

    if df>=5:
        if cr<=0.05:
            i='negligenciavel'
        elif 0.05<cr<=0.13:
            i='pequeno'
        elif 0.13<cr<=0.22:
            i='medio'
        else:
            i='grande'

    return i

def create_cm_cramers_level(cramers_value,cramers_df):
    '''
    This function creates dataframe containing the crames"s levels of association between the 
    categorical variables.To interpret the Cramer's values the Cohen's rule of thumb was used.
    
    cramers_value - matrix containing the calculated crame's values between each pair of features
    cramers_df    - cramer's dregrees of freedom
    
    '''
    level=cramers_value.copy().astype('string')
    n=level.shape[0]
    for i in range(n):
        for j in range(n):
            df=cramers_df.iloc[i,j]
            cv=cramers_value.iloc[i,j]
            level.iloc[i,j]=cramers_level(df,cv)
    return level

    
def association_to_number(x):
    #Function to convert the dataframe containg the cramer's levels of association to numbers. 
    #The conversion  is necessary to create a heatmap 

    if x=='grande':
        return 255
    if x=='medio':
        return 128
    if x== 'pequeno':
        return 64 
    if x== 'negligenciavel':
        return 0


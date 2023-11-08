import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from time import sleep
from tqdm import tqdm,trange
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

def train_baseline(x,y,models_dict,sampler=None,n_splits=5):
    '''
    x           - input data
    y           - output data
    models_dict - dicionário com os modelos que serão treinados.    
    sampler     - instância de um objeto que fará o balanceamentos do dataset
    n_splits    - numero de splits que será usada na validação cruzada.
    '''
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    results={'model':[],'scores':[]}
    
    
    for key in tqdm(models_dict,desc='Models'):
        scores={'accuracy':[],'precision':[],'recall':[],'roc_auc':[]}
        for train_index, test_index in skf.split(x, y):
            
            
            
            x_train,x_test=x.iloc[train_index],x.iloc[test_index]
            y_train,y_test=y.iloc[train_index],y.iloc[test_index]
            
            if sampler:
            #model=make_pipeline(sampler,
            #           models_dict[str(key)][0])
                x_train,y_train=sampler.fit_resample(x_train,y_train)
                models_dict[str(key)][0].fit(x_train,y_train)
            else :
                models_dict[str(key)][0].fit(x_train,y_train)
            y_pred_proba=models_dict[str(key)][0].predict_proba(x_test)
            y_pred=np.argmax(y_pred_proba,axis=1)
            scores['accuracy']=scores['accuracy']  +[accuracy_score(y_test,y_pred)]
            scores['precision']=scores['precision']+[precision_score(y_test,y_pred)]
            scores['recall']=scores['recall']      +[recall_score(y_test,y_pred)]
            scores['roc_auc']=scores['roc_auc']    +[roc_auc_score(y_test,y_pred_proba[:,1])]
        results['model'] =results['model']+[str(key)]
        results['scores']=results['scores']+[scores]
    return results


def update_dictionary(dict_complete,dataset,dict_brief=None,model_name="LGBM"):
    '''
    The models were trained using cross-validation, which means that the scores were calculated for each split. 
    The purpose of this function is to calculate the mean of the scores. 

    If more than one model was trained, it is necessary to specify the model's name. 
    The dictionary contains a key with all the trained models in this case.
    
    dict_complete - a dictionary containing all the scores of the cross validation procedure
    dataset       - description of the dataset used to train the models
    model name    - the name of the model. The function return the results for only one model.
    dict_brief    - a dictionary containing the mean of the scores. If a dict is passed to the function, it will
    be updated. If not, a new one is created.
    
    The function returns a dictionary containing the mean of the scores for the metrics precision, 
    recall and roc auc
    '''
    if dict_brief is None:
        dict_brief={'dataset':[],'mean_precision':[],'mean_recall':[],'mean_roc':[]}

    dict_brief['dataset']       = dict_brief['dataset']+[dataset]

    if 'model' in dict_complete.keys():
        n=dict_complete['model'].index(model_name)
        dict_brief['mean_precision']= dict_brief['mean_precision']+[np.mean(dict_complete['scores'][n]['precision'])]
        dict_brief['mean_recall']   = dict_brief['mean_recall']+[np.mean(dict_complete['scores'][n]['recall'])]
        dict_brief['mean_roc']      = dict_brief['mean_roc']+[np.mean(dict_complete['scores'][n]['roc_auc'])]
    
    else:
        dict_brief['mean_precision']= dict_brief['mean_precision']+[np.mean(dict_complete['test_precision'])]
        dict_brief['mean_recall']   = dict_brief['mean_recall']+[np.mean(dict_complete['test_recall'])]
        dict_brief['mean_roc']      = dict_brief['mean_roc']+[np.mean(dict_complete['test_roc_auc'])]
    return dict_brief


def train_model(model,x,y,scoring,sampler=None):
    '''
    model       -model that will be trained (instance of an object)
    scoring     - metrics that will be measured
    sampler     - instance of an object that will balance the dataset   
    '''
    if sampler:
        model_pipeline=make_pipeline(sampler,
                           model)
        scores=cross_validate(model_pipeline,x,y,scoring=scoring)
    else :
        scores=cross_validate(model,x,y,scoring=scoring)
    return scores

def generating_new_samples(x,y,codings_size,auto_encoder,decoder):
        '''
        Função será usada para gerar novas amostras da classe minoritária
	codings_size - shape of the sample that the encoder outpus
        model        - the decoder that will generate the new samples
        '''
        n_samples_per_class=pd.Series.value_counts(y) #get the number of samples from each class
        target_value_minority_class=n_samples_per_class.idxmin() #get target value of the minority class
        n_min,n_max=n_samples_per_class.min(),n_samples_per_class.max() #number of samples from each class
        n_samples=n_max-n_min #number of samples that will be generated
        
        #Separating the samples from both classes
        indexes_class_1=np.where(y==1)[0]
        indexes_class_0=np.where(y==0)[0]

        x_class_1=x.iloc[indexes_class_1]
        x_class_0=x.iloc[indexes_class_0]
        y_class_1=y.iloc[indexes_class_1]
        y_class_0=y.iloc[indexes_class_0]
        
        x_class_1_train,x_class_1_test=train_test_split(x_class_1, test_size=0.33, random_state=42)
        #Training the autoencoder
        history=auto_encoder.fit(x_class_1_train,x_class_1_train,epochs=200,batch_size=128,
                                   validation_data=[x_class_1_test,x_class_1_test])
        condings_new_samples=tf.random.normal(shape=[n_samples,codings_size])
        #Creating new samples
        new_samples=pd.DataFrame(decoder(condings_new_samples).numpy(),columns=x.columns)
        y_new=pd.Series(n_samples*[target_value_minority_class])
        x=pd.concat([x,new_samples],ignore_index=True,axis=0)
        y=pd.concat([y,y_new],axis=0)
        x,y=shuffle(x, y, random_state=0)
        x.reset_index(inplace=True,drop=True)
        y.reset_index(inplace=True,drop=True)
        return x,y

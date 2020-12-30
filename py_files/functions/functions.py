"""
This file contains all the helper functions for the analysis
"""
# imports -----------------------------------------------------------------------------------
# mathematics & data manipulation tools
import pandas as pd
import numpy as np

# import plotting tools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# import scipy
import scipy.stats as stats

# ml models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

# import itertools
import itertools

# progress bar
from ipywidgets import IntProgress

# request for scrapping
import requests

# scikit-image for displaying
from skimage import io

# import regex
import re


# functions --------------------------------------------------------------------------------
# plot distribution
def plot_distribution(X_train, Y_train, title):
    """
    Takes the features and criterium and plots distribution plots for it

    params:
    -----------
    X_train: pd.DataFrame; the features
    Y_train: pd.DataFrame; the cirterium
    title: str; the name of the plot

    returns:
    -----------
    nothing
    """
    # concat for plotting
    tempDf = pd.concat([X_train, Y_train], axis=1)

    # create dist plot
    distPlot = make_subplots(rows = 3, cols = 4, subplot_titles=tempDf.columns)

    # starting row
    rowIdx = 1
    colIdx = 1

    for i, col in enumerate(tempDf.columns):
        # build correct colIdx
        if colIdx > 4:
            colIdx = 1
        
        # add data (differenciate by scale type)
        if len(tempDf[col].value_counts().index) > 2: # its probably one of the continous variables
            stat = tempDf[col].value_counts(bins=[0,.2,.4,.6,.8,1], normalize=True).sort_index()
        else: # its probably one of the binary data
            stat = tempDf[col].value_counts(normalize=True).sort_index().rename({0:"no", 1:"yes"})
        
        distPlot.add_trace(go.Bar(x=stat.index.astype(str).tolist(), y=stat.values), row=rowIdx, col=colIdx)
        
        # build correct rowIdx
        if (i+1)%4 == 0:
            rowIdx += 1
        
        colIdx += 1

    distPlot.update_layout(title_text=title, showlegend=False)
    distPlot.show()


    return

# check for normal distribution
def check_normal_distribution(variable):
    """
    check the normal distribution of a variable on 3 different tests: shapiro-wilk test,
    d'agostinos K2 & anderson-darling

    params:
    -----------
    variable: pd.Series; the variable to be checked

    returns:
    -----------
    nothing
    """
    # shapiro-wilk test; --> H0 testing
    print("shapiro-wilk: ", stats.shapiro(variable))
    print()

    # d'agostino's K² --> H0 testing
    print("K²: ", stats.normaltest(variable))
    print()

    # anderson-darling --> critical values should be higher than test values
    anderson = stats.anderson(variable)
    print("anderson:")
    print("test stat: ", round(anderson[0],1))
    print("Is the test value smaller than the critical value for:")
    for crit, lvl in zip(anderson[1], anderson[2]):
        print("sig. lvl", lvl, ":", anderson[0] < crit)

    return

# function for correlation plotting
def find_corr(binary, continuous, X_train, Y_train):
    """
    Calculation of the correlations between features and criterium. Does also plot the graphs
    
    params:
    -----------
    binary: list; contains the names of the binary coded features
    continuous: list; contains the names of the continous scaled features
    X_train: pd.DataFrame; the features
    Y_train: pd.DataFrame; the criterium
    
    returns:
    -----------
    nothing
    """
    corr = {}

    # calculate the correlations for the binary elements
    for col in binary:
        corr.update({col:stats.pointbiserialr(X_train[col], Y_train)})

    # calculate the correlation for the continuous elements
    for col in continuous:
        corr.update({col:stats.pearsonr(X_train[col], Y_train)})

    # transform into a nice table & display the table
    corr = pd.DataFrame(corr, index=["correlation", "p-value"])
    display(corr.round(2))

    # visual inspection of the found correlations
    distPlot = make_subplots(rows = 3, cols = 4, subplot_titles=X_train.columns)
    print()

    # starting row
    rowIdx = 1
    colIdx = 1

    # y for trendline
    Y = Y_train.values.reshape(-1,1)

    for i, col in enumerate(X_train.columns):
        # build correct colIdx
        if colIdx > 4:
            colIdx = 1

        # calculate trend
        X = X_train[col].values.reshape(-1,1)
        lr = LinearRegression().fit(X, Y)
        y = lr.predict(X)

        # add data    
        distPlot.add_trace(go.Scatter(mode="markers", x=X_train[col], y=Y_train), row=rowIdx, col=colIdx)
        distPlot.add_trace(go.Scatter(mode="lines", x=X_train[col], y=pd.DataFrame(y).iloc[:,0], line=dict(color="red")),
                           row=rowIdx, col=colIdx)

        # build correct rowIdx
        if (i+1)%4 == 0:
            rowIdx += 1

        colIdx += 1

    distPlot.update_layout(title_text="Scatterplot of the features with the criterium",
                           showlegend=False,
                           height=800
                          )
    distPlot.update_yaxes(range=[-.5, 1.5])
    distPlot.show()
    
    return

# calculate multicolinearity
def multicolinearity(X_train):
    """
    calculate the VIF score for each of the features in the trainset

    params:
    ----------
    X_train: pd.DataFrame; the trainset with the features

    returns:
    ----------
    VIF score as pd.DataFrame
    """
    # get list of features
    features = X_train.columns.tolist()
    print("features are: ")
    print(features)
    print()

    VIF = {}

    # for each feature
    for i in range(len(features)):
        # grab first feature
        target = features[0]
        
        # build df
        X = X_train[features[1:]].values
        Y = X_train[target].values.reshape(-1,1)
        
        # calculate regression
        lr = LinearRegression().fit(X,Y)
        y = lr.predict(X)
        r2 = r2_score(Y, y)
        
        # calculate VIF
        VIF.update({target:1/(1-r2)})
        
        # update features -> target to last position
        features.remove(target)
        features.append(target)
        
    # transform into nice table & show results
    results = pd.DataFrame(VIF, index=["VIF coef"]).T.sort_values("VIF coef", ascending=False)

    return results

# grid search method
def grid_search(X_train, Y_train, plannedModels):
    """
    performes a greadSearch for each defined parameter
    
    params:
    ----------------
    X_train: pd.DataFrame; feature data
    Y_train: pd.DataFrame; predictor data
    plannedModels: dict; dict with info about model name, the model object and the parameter space
    
    returns:
    ----------------
    performance of each model under certain parameters
    """
    results = {}
    
    # for each model
    for model in plannedModels:
        
        # get features-predictor
        """
        features = df.iloc[:, 1:-2].values
        predictor = df.iloc[:, -2].values.ravel()
        """
        features = X_train.values
        predictor = Y_train.values.ravel()

        # initiate grid search
        estimator = plannedModels.get(model).get("class")
        parameters = plannedModels.get(model).get("parameters")
        search = GridSearchCV(estimator=estimator, param_grid=parameters, cv=3,
                              n_jobs=3, return_train_score=True, verbose=2) # 3 means: 2/3 train, 1/3 test

        # run the search
        search.fit(features, predictor)
        
        # extract model results and add infos
        modelResults = search.cv_results_
        modelResults.update({"n_features":features.shape[1]})
        modelResults.update({"n_total":features.shape[0]})

        # add modelResults
        results.update({model:search.cv_results_})
        
    return results

# function to transform the results dict
def make_clean(results):
    """
    function convertes the results dict from the gridSearch into a nicely formatted df
    
    params:
    -----------
    results: dict; results dict from the gridSearch function
    
    returns:
    -----------
    pd.DataFrame with all relevant search infos
    """
    formatted = []
    
    # for each key (aka model) in the results
    for key in results:
        df = pd.DataFrame(results.get(key))
        
        # add the model name
        df["model"] = key
        
        # add adjusted r2
        for adjust in ["mean_test_score", "mean_train_score"]:
            df["adjusted_"+adjust] = 1-(1-df[adjust])*(df["n_total"]-1)/(df["n_total"]-df["n_features"]-1)
            
        # add the delta between test and train score
        df["delta"] = df["mean_train_score"] - df["mean_test_score"]
            
        formatted.append(df)
        
    # concat
    formatted = pd.concat(formatted, axis=0)
    
    return formatted

# sort girdSearch results
def sort_search_results(score, threshold=.3):
    """
    function sorts the clean score by different relevant bits of info and returns the sorted arrays

    params:
    ------------
    score: pd.DataFrame; contains the results from the grid search (after makeClean)
    threshold: float; indicates the threshold for the min size of an test score

    returns:
    ------------
    list; [deltaOptimized list, testOptimized list, trainOptimized list]
    """
    # show for each learner the best settings
    grouped = score.groupby("model")

    deltaOptimized = []
    testOptimized = []
    trainOptimized = []

    # for each learner
    for key in grouped.groups.keys():
        temp = grouped.get_group(key)
        temp.sort_values("mean_train_score", ascending=True, inplace=True)
        
        # add to train Optimized ranking
        trainOptimized.append(temp.iloc[-1, :])
        
        # filter all learners with low test_score (in the end its the importent stat.)
        temp = temp[temp["mean_test_score"] > threshold]
        #display(temp.sort_values(["delta", "mean_test_score"]))
        
        # kick each learner who doesn't meet the criteria of at least .3 test score
        if temp.shape[0] > 0:
            temp.sort_values("delta", ascending=False, inplace=True)
            deltaOptimized.append(temp.iloc[-1,:])
            temp.sort_values("mean_test_score", ascending=True, inplace=True)
            testOptimized.append(temp.iloc[-1,:])
        else:
            pass
        
    # build dfs
    deltaOptimized = pd.concat(deltaOptimized, axis=1).T.sort_values("mean_test_score", ascending=False)
    testOptimized = pd.concat(testOptimized, axis=1).T.sort_values("mean_test_score", ascending=False)
    trainOptimized = pd.concat(trainOptimized, axis=1).T.sort_values("mean_train_score", ascending=False)

    return [deltaOptimized, testOptimized, trainOptimized]

# score plot function
def score_plot(data, testScore, delta, params, title="Diagramm"):
    """
    params:
    ----------
    data: pd.DataFrame; contains data for the plot
    testScore: str; name of the column with the score value
    delta: str; name of the column with the delta score
    params: str; name of the column with the parameters
    title: str; the name of the plot
    
    returns:
    ----------
    nothing
    """
    # create figure
    fig = go.Figure()

    # add traces
    fig.add_trace(go.Scatter(mode="lines", x=data["model"], y=data[testScore],
                         name="model performance", text=data[params].astype(str)))
    fig.add_trace(go.Scatter(mode="markers+lines", x=data["model"], y=data[delta],
                             name="delta", text=data[params].astype(str)))

    # update layout & print it
    fig.update_layout(title_text=title).show()
    
    return

# shapley value calculation
def dominance(x, y, model):
    """
    takes the predictors and the criteria and calculates for the specified model the importance for each
    features
    
    params:
    ------------
    x: pd.DataFrame; contains all the features
    y: pd.DataFrame; contains the creteria
    model: sklearn model; the specified model
    
    returns:
    ------------
    dict {"featureName":importance, "featureName2":importance, ....}
    """
    
    # build all combinations
    combiResults = {}
    
    print("calculating R2 for each model")
    f=IntProgress(min=0, max=x.shape[1])
    display(f)
    
    # for each length of predictor combination
    for i in range(x.shape[1]):
        # build each combination for the total length
        combi = itertools.combinations(x.columns,i+1)

        # for each element in the combi list, build the model
        for c in combi:
            #print(list(c))
            #print(model)
            model.fit(x[list(c)], y)
            #y = model.predict(df[list(c)])
            #r2 = r2_score(df["winpercent"], y)
            #combiResults.update({c:r2})
            combiResults.update({c:model.score(x[list(c)], y)})
        
        # incremet progressbar
        f.value += 1
        
    print("all R2s calculated. Building final dominance score.")
    #print(len(combiResults))

    
    # calculate each dominace score for each predictor
    dominanceScore = {}
    
    f=IntProgress(min=0, max=x.shape[1])
    display(f)
    
    print()
    
    for col in x.columns:
        # filter keys swith the target column/predictor in it
        target = [key for key in combiResults if col in key]
        
        # get for each target/predictor the remainders
        remainder = [set(entry)-set([col]) for entry in target]
        
        # build results dict
        difference = {}
        for z in range(x.shape[1]):
            difference.update({z+1:[]})
            
        # iterate through entry lenght and build the differences
        for t,r in zip(target,remainder):
            # if the remainder lenght == 0 its basically the base model vs. the predictor
            if len(r) == 0:
                r2 = combiResults.get(tuple([col]))
                difference.get(len(t)).append(r2)
            # else its the remainder, adding the target predictor
            else:
                getKey = [key for key in combiResults if set(key) == r]
                if len(getKey) == 1:
                    r2 = combiResults.get(tuple(t)) - combiResults.get(getKey[0])
                    difference.get(len(t)).append(r2)
                else:
                    # there was an failure, print it!
                    print("failure at:", set(key), "Did not found the r2 score")
            
        # build the mean difference for the predictor and apply it to the final results
        colScore = 0
        for key in difference:
            colScore += np.array(difference.get(key)).mean()
            
        dominanceScore.update({col:colScore/x.shape[1]})
        
        #increment progressbar
        f.value += 1
        
    print("All calculations done")
    
    return dominanceScore

#scrapper function
def scrapping(candyName):
    """
    function scrapps google for a picture and displays it
    
    params:
    -----------
    candyname: str; name of the to be searched candy
    
    returns:
    -----------
    nothing
    """
    # get url
    print("candy:", candyName)
    url = "https://www.google.com/search?q=" + candyName + "candy&hl=de&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjC3Lmb4fLtAhVPqxoKHUu9BtgQ_AUoAXoECAwQAw&biw=1272&bih=1265"

    # get url source
    r = requests.get(url)

    # search for the first image
    m = re.findall(r'src="(.*?)"', r.text)

    # load image
    loaded = io.imread(m[1])
    
    # print
    go.Figure(go.Image(z=loaded)).update_layout(width=300, height=300).show()
    
    return


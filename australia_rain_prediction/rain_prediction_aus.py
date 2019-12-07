import pandas as pd
import numpy as np # used for handling numbers
from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler # used for feature scaling
import matplotlib.pyplot as c_plt
import seaborn as sns
import seaborn as sns1
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.combine import SMOTETomek
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
from matplotlib import pyplot as s_plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.dummy import DummyClassifier
from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
import rules


import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

def randomized_search(X_train,y_train):
    grid_params = {
        'n_neighbors' : [1,3,5],
        'weights' : ['uniform', 'distance'],
        'metric' : ['euclidean' , 'manhattan']
    }

    gs = RandomizedSearchCV(
        KNeighborsClassifier(),
        grid_params,
        verbose =1,
        cv =10,
        n_jobs=-1
    )
    gs_results = gs.fit(X_train,y_train)
    print(gs_results.best_score_)
    print(gs_results.best_estimator_)
    print(gs_results.best_params_)

# Function to graphically represent ouput lables
def plotOutputLabel(dataframe,title):
    class_dist = dataframe.groupby('RainTomorrow').size()
    class_label = pd.DataFrame(class_dist, columns=['Size'])
    plt.figure(figsize=(10, 2))
    sns.barplot(x=class_label.index, y='Size', data=class_label)
    plt.title(title)
    plt.show()


def correlation(df,title):
    ax = c_plt.axes()
    ax.set_title(title)
    data_correlate = df.corr()
    #c_plt.figure(figsize=(10, 10))
    sns.heatmap(data_correlate, linewidth=3, linecolor='black', ax = ax)
    c_plt.show()
#delete rows that have missing values in wind direction  winddir9am and winddir3pm
def preprocessing(df):
    df = df.drop(df.index[df.WindGustDir == 'NA'])
    df = df.drop(df.index[df.WindDir9am == 'NA'])
    df = df.drop(df.index[df.WindDir3pm == 'NA'])

    #replace mean value for mintemp and max temp based on location and month
    df["WindSpeed9am"] = df["WindSpeed9am"].fillna(df.groupby('Location')['WindSpeed9am'].mean())
    df["WindSpeed3pm"] = df["WindSpeed3pm"].fillna(df.groupby('Location')['WindSpeed3pm'].mean())
    df["Humidity9am"] = df["Humidity9am"].fillna(df.groupby('Location')['Humidity9am'].mean())
    df["Humidity3pm"]= df["Humidity3pm"].fillna(df.groupby('Location')['Humidity3pm'].mean())
    df["Pressure9am"] = df["Pressure9am"].fillna(df.groupby('Location')['Pressure9am'].mean())
    df["Pressure3pm"] = df["Pressure3pm"].fillna(df.groupby('Location')['Pressure3pm'].mean())
    df["Temp9am"] = df["Temp9am"].fillna(df.groupby('Location')['Temp3pm'].mean())
    df["Temp3pm"] = df["Temp3pm"].fillna(df.groupby('Location')['Temp3pm'].mean())
    df['WindGustDir'] = df['WindGustDir'].replace(np.nan,'0')
    df['WindDir9am'] = df['WindDir9am'].replace(np.nan,'0')
    df['WindDir3pm'] = df['WindDir3pm'].replace(np.nan,'0')

    #remove the columns evaporation and sunshine risk_mm and cloud3pm and cloud9am as more null values
    df.drop(["Evaporation", "Sunshine","Cloud3pm","Cloud9am"] , axis=1, inplace=True)

    #droping columns date location and risk-mm
    df = df.drop(columns = ['Date', 'Location', 'RISK_MM'])

    #Converting the RainToday and RainTomorrow data into binary
    df['RainToday'].replace('No', 0, inplace = True)
    df['RainToday'].replace('Yes', 1, inplace = True)
    df['RainTomorrow'].replace('No', 0, inplace = True)
    df['RainTomorrow'].replace('Yes', 1, inplace = True)
    df = df.dropna()

    print("Skewness")
    print(df.skew(axis=0, skipna=True))

    print("kurtosis")
    print(df.kurtosis())

    df.plot.hist(alpha=0.5, bins=15, grid=True, legend=None)
    plt.xlabel("Feature value")
    plt.title("Histogram")
    plt.show()

    return (df)

def onehotencoder(df):
    # onehotencoder for categorical data
    df1 = pd.DataFrame()
    df1['WindGustDir'] = df['WindGustDir']
    df1['WindDir9am'] = df['WindDir9am']
    df1['WindDir3pm'] = df['WindDir3pm']
    encoder = OneHotEncoder()
    encoder.fit(df1)
    x = encoder.transform(df1).toarray()
    for j in range(len(encoder.get_feature_names())):
        df[encoder.get_feature_names()[j]] = [i[j] for i in x]
    df = df.drop(columns=['WindGustDir'])
    df = df.drop(columns=['WindDir9am'])
    df = df.drop(columns=['WindDir3pm'])
    #drop the x0_0 as they have missing values
    df = df.drop(columns=['x0_0', 'x1_0', 'x2_0'])
    return (df)

def normalization(df):
    ####normalization
    df2=pd.DataFrame(df)
    scaler = MinMaxScaler()
    scaler.fit(df)
    df=pd.DataFrame(scaler.transform(df),columns=df2.columns)
    df= df.head(2000)
    return (df)

def split_data(df):
    features = pd.DataFrame(df)
    features = features.drop(columns=['RainTomorrow'])
    output_label = pd.DataFrame(df['RainTomorrow'])
    return(features,output_label)

#cross validation
def cross_validation(df_full):
    features,output_label= split_data(df_full)
    X_train, X_test, y_train, y_test = train_test_split(features,output_label, test_size=0.33, random_state=42)
    return(X_train, X_test, y_train, y_test)

def featured_cross_validation(features_new,output_label):
    X_train, X_test, y_train, y_test = train_test_split(features_new,output_label, test_size=0.33, random_state=42)
    return(X_train, X_test, y_train, y_test)

def algorithms_compare(features, output_label):
    # prepare models
    seed = 7
    models = []
    models.append(('NB', GaussianNB()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('DM', DummyClassifier(strategy='most_frequent')))
    models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
    models.append(('decision',tree.DecisionTreeClassifier()))
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    for name, model in models:
        # kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model,features, output_label, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

#oversampling
def oversampling(df,title):
    print("oversampling")
    features, output_label = split_data(df)
    Features_resampled, output_label_resampled = SMOTE().fit_resample(features, output_label )
    print(type(Features_resampled))
    df_full = pd.concat([pd.DataFrame(Features_resampled,columns=features.columns),pd.DataFrame(output_label_resampled, columns= output_label.columns)], axis=1)
    return(df_full)
    #cross_validation(df_full)

def feature_selection(df):
    # Feature selection
    features,output_label = split_data(df)
    selector = SelectKBest(chi2, k=7)
    selector.fit(features,output_label)
    features_new = selector.transform(features)
    print(features.columns[selector.get_support(indices=True)]) #top 3 columns
    return(features_new,output_label)
    # df_full = pd.concat([pd.DataFrame(features_new, columns=features_new.columns),pd.DataFrame(output_label, columns=output_label.columns)], axis=1)
    # return(df_full)

#balance sampling
def balance_sampling(df,title):
    features, output_label = split_data(df)
    smote_tomek = SMOTETomek(random_state=0)
    Features_resampled, output_label_resampled = smote_tomek.fit_resample(features, output_label )
    df_full = pd.concat([pd.DataFrame(Features_resampled,columns=features.columns),pd.DataFrame(output_label_resampled, columns= output_label.columns)], axis=1)
    return(df_full)
    #cross_validation(df_full)

def under_sampling(df,title):
    features, output_label = split_data(df)
    ncr = NeighbourhoodCleaningRule()
    X_undersampled, y_undersampled = ncr.fit_resample(features, output_label)
    df_full = pd.concat([pd.DataFrame(X_undersampled, columns=features.columns), pd.DataFrame(y_undersampled, columns=output_label.columns)], axis=1)
    return(df_full)


def navies_bayes( X_train, X_test, y_train, y_test):
    plt.figure(0).clf()
    navies_bayes_classifier = GaussianNB()
    navies_bayes_classifier = navies_bayes_classifier.fit(X_train, y_train)
    y_prect = navies_bayes_classifier.predict(X_test)
    # print(confusion_matrix(y_prect, y_test))
    # print(accuracy_score(y_prect, y_test))
    print("Naive Bayes Algorithm")
    printAccuracies(y_prect, y_test)
    navies_bayes_classifier_score = accuracy_score(y_test, y_prect)
    # print('the confusion matrix obtain from navies bayes  is :', confusion_matrix(y_prect, y_test))
    # print('the accuracy score obtain from navies bayes  is :', navies_bayes_classifier_score)
    fpr, tpr, thresh = metrics.roc_curve(y_test, y_prect)
    auc = metrics.roc_auc_score(y_test, y_prect)
    plt.plot(fpr, tpr, label="Naive Bayes, auc=" + str(auc))
    plt.legend(loc=0)

   # roc_draw(y_test, y_prect)




#random forest classifier
def random_forest( X_train, X_test, y_train, y_test):
    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.fit(X_train, y_train)
    y_prect = random_forest_classifier.predict(X_test)
    random_forest_classifier_score = accuracy_score(y_test, y_prect)
    # print('the confusion matrix obtain from random forest classifier is :', confusion_matrix(y_prect, y_test))
    # print('the accuracy score obtain from random forest classifier is :', random_forest_classifier_score)
    print("Random Forests Algorithm")
    printAccuracies(y_prect, y_test)
    fpr, tpr, thresh = metrics.roc_curve(y_test, y_prect)
    auc = metrics.roc_auc_score(y_test, y_prect)
    plt.plot(fpr, tpr, label="Random forest, auc=" + str(auc))
    plt.legend(loc=0)

    # return (y_test, y_prect)

#decision tree
def decision_tree( X_train, X_test, y_train, y_test):
    decision_tree_classifier = tree.DecisionTreeClassifier()
    decision_tree_classifier.fit(X_train, y_train)
    y_prect = decision_tree_classifier.predict(X_test)
    decision_tree_classifier_score = accuracy_score(y_test, y_prect)
    # print('the confusion matrix obtain from decision tree classifier is :', confusion_matrix(y_prect, y_test))
    # print('the accuracy score obtain from decision tree classifier is :', decision_tree_classifier_score)
    print("Decision Tree Algorithm")
    printAccuracies(y_prect, y_test)
    fpr, tpr, thresh = metrics.roc_curve(y_test, y_prect)
    auc = metrics.roc_auc_score(y_test, y_prect)
    plt.plot(fpr, tpr, label="Decision Tree, auc=" + str(auc))
    plt.legend(loc=0)


def choose_strategy_dummy_classifier(X_train, X_test, y_train, y_test):
    strategies = ['most_frequent','stratified', 'uniform']
    max_strategie = ''
    score_max = 0
    test_scores = []
    for s in strategies:
        if s == 'constant':
            dclf = DummyClassifier(strategy=s, random_state=0, constant='M')
        else:
            dclf = DummyClassifier(strategy=s, random_state=0)
        dclf.fit(X_train, y_train)
        score = dclf.score(X_test, y_test)
        test_scores.append(score)
        if score > score_max:
            score_max = score
            max_strategie = s
            print(s)
    ax = sns.stripplot(strategies, test_scores)
    ax.set(xlabel='Strategy', ylabel='Test Score')
    s_plt.title("Comparing stratagies")
    s_plt.show()
    return(max_strategie)

def rule_based_scopeRules(X_train, X_test, y_train, y_test, title):
    clf = rules.SkopeRules(max_depth_duplication=None,
                     n_estimators=30,
                     precision_min=0.2,
                     recall_min=0.01)


    clf.fit(X_train,y_train)
    y_score = clf.score_top_rules(X_test)# Get a risk score for each test example
    #print(y_score)
    print("scope rules")
    print("===============================================")
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall curve for scope rules ')
    plt.show()



def dummy_classifier(X_train, X_test, y_train, y_test,max_strategie):
    #create the dummy with the most score strategy
    dummy_classifier = DummyClassifier(strategy= max_strategie,  random_state = 0)
    dummy_classifier.fit(X_train, y_train)
    y_prect = dummy_classifier.predict(X_test)
    dummy_classifier_score = accuracy_score(y_test, y_prect)
    # print('the confusion matrix obtain from dummy classifier is :', confusion_matrix(y_prect, y_test))
    # print('the accuracy score obtain from dummy classifier is :', dummy_classifier_score)
    print("Dummy Classifier Algorithm")
    printAccuracies(y_prect, y_test)
    fpr, tpr, thresh = metrics.roc_curve(y_test, y_prect)
    auc = metrics.roc_auc_score(y_test, y_prect)
    plt.plot(fpr, tpr, label="Rule Based  , auc=" + str(auc))

    plt.legend(loc=0)


def knn( X_train, X_test, y_train, y_test, title):
    k_neighbors = KNeighborsClassifier(n_neighbors=10)
    k_neighbors.fit(X_train, y_train)
    y_prect = k_neighbors.predict(X_test)
    k_neighbors_score = accuracy_score(y_test, y_prect)
    # print('the confusion matrix obtain from KNN classifier is :', confusion_matrix(y_prect, y_test))
    # print('the accuracy score obtain from KNN classifier is :', k_neighbors_score)
    print("K-NN Algorithm")
    printAccuracies(y_prect, y_test)
    fpr, tpr, thresh = metrics.roc_curve(y_test, y_prect)
    auc = metrics.roc_auc_score(y_test, y_prect)
    plt.plot(fpr, tpr, label="KNN , auc=" + str(auc))
    plt.title(title)
    plt.legend(loc=0)
    plt.show()
    # return (y_test, y_prect)

def accuracy_graph(X_train, X_test, y_train, y_test, max_strategie,title):
    classifiers =[GaussianNB(),RandomForestClassifier(), DummyClassifier(strategy= max_strategie),KNeighborsClassifier(n_neighbors=3),tree.DecisionTreeClassifier()]
    log_cols = ["Classifier", "Accuracy"]
    log = pd.DataFrame(columns=log_cols)

    for clf in classifiers:
        clf.fit(X_train, y_train)
        name = clf.__class__.__name__
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        log_entry = pd.DataFrame([[name, acc * 100]], columns=log_cols)
        log = log.append(log_entry)
        print("=" * 30)
    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
    plt.xlabel('Accuracy %')
    plt.title(title)
    plt.show()


def plotTTest(a,b):
    print(stats.ttest_rel(a,b))

def printAccuracies(y_test,predicted):
    precision, recall, fscore, support = score(y_test, predicted)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, predicted))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

def main():
    data = pd.read_csv('weather.csv')
    df = pd.DataFrame(data)
    df2 = pd.DataFrame(data)
    print("dataframe object is ::::")
    print(df)
    #class lable distribution
    plotOutputLabel(df, "For Raw data")
    #corelation
    print("correlation for raw data")
    title = "Corelation for raw data"
    correlation(df,title)
    #preprocessing
    df = preprocessing(df)
    #corelation
    print("correlation for data after preprocessing")
    title = "correlation for data after preprocessing"
    correlation(df, title)
    #onehot encoder
    df = onehotencoder(df)
    #normalization
    df = normalization(df)
    #oversampling call all algorithms


    print("Over Sampling Results")
    title = " After Over Sampling "
    df_oversampled = oversampling(df,title)
    plotOutputLabel(df_oversampled, title)
    #cross validation
    X_train, X_test, y_train, y_test = cross_validation(df_oversampled)
    #scope rules calling
    rule_based_scopeRules(X_train, X_test, y_train, y_test, title)

    #RandonizedCV
    #randomized_search(X_train, y_train)

    max_strategie = choose_strategy_dummy_classifier(X_train, X_test, y_train, y_test)
    navies_bayes(X_train, X_test, y_train, y_test)
    random_forest(X_train, X_test, y_train, y_test)
    decision_tree(X_train, X_test, y_train, y_test)
    dummy_classifier(X_train, X_test, y_train, y_test, max_strategie)
    knn(X_train, X_test, y_train, y_test,title)
    accuracy_graph(X_train, X_test, y_train, y_test, max_strategie,title)

    # #accuracy in boxplots
    # features, output_label = split_data(df_oversampled)
    # #algorithms_compare(features, output_label)
    #
    print("Balanced Sampling results")
    #balance sampling
    title = "After Balance Sampling"
    df_balancesampled = balance_sampling(df,title)
    plotOutputLabel(df_balancesampled, title)

    # cross validation
    X_train, X_test, y_train, y_test = cross_validation(df_balancesampled)
    #scope rules calling
    rule_based_scopeRules(X_train, X_test, y_train, y_test, title)
    max_strategie = choose_strategy_dummy_classifier(X_train, X_test, y_train, y_test)
    navies_bayes(X_train, X_test, y_train, y_test)
    random_forest(X_train, X_test, y_train, y_test)
    decision_tree(X_train, X_test, y_train, y_test)
    dummy_classifier(X_train, X_test, y_train, y_test, max_strategie)
    knn(X_train, X_test, y_train, y_test, title)
    accuracy_graph(X_train, X_test, y_train, y_test, max_strategie,title)
    # dummy_classifier(X_train, X_test, y_train, y_test)


    print("Under Sampling results")
    #undersampling
    title = " After Under Sampling "
    df_undersampled = under_sampling(df,title)
    plotOutputLabel(df_undersampled, title)
    # cross validation
    X_train, X_test, y_train, y_test = cross_validation(df_undersampled)
    #scope rules calling
    rule_based_scopeRules(X_train, X_test, y_train, y_test, title)
    max_strategie = choose_strategy_dummy_classifier(X_train, X_test, y_train, y_test)
    navies_bayes(X_train, X_test, y_train, y_test)
    random_forest(X_train, X_test, y_train, y_test)
    decision_tree(X_train, X_test, y_train, y_test)
    dummy_classifier(X_train, X_test, y_train, y_test, max_strategie)
    knn(X_train, X_test, y_train, y_test, title)
    accuracy_graph(X_train, X_test, y_train, y_test, max_strategie,title)
    #feature sampling

    print("After Feature Selection Results")
    #df_featured = feature_selection(df)
    title = "After feature selection"
    features_new,output_label= feature_selection(df)
    X_train, X_test, y_train, y_test = featured_cross_validation(features_new,output_label)
    #scope rules calling
    rule_based_scopeRules(X_train, X_test, y_train, y_test, title)
    max_strategie = choose_strategy_dummy_classifier(X_train, X_test, y_train, y_test)
    navies_bayes(X_train, X_test, y_train, y_test)
    random_forest(X_train, X_test, y_train, y_test)
    decision_tree(X_train, X_test, y_train, y_test)
    dummy_classifier(X_train, X_test, y_train, y_test, max_strategie)
    knn(X_train, X_test, y_train, y_test, title)
    # #features split and output to compare them
    #accuracy in graph
    accuracy_graph(X_train, X_test, y_train, y_test,max_strategie,title)


    table_of_accuracies = pd.DataFrame()
    table_of_accuracies['naive'] = cross_val_score(GaussianNB(), features_new, output_label, cv=10)
    table_of_accuracies['random'] = cross_val_score(RandomForestClassifier(), features_new, output_label, cv=10)
    table_of_accuracies['decision'] = cross_val_score(tree.DecisionTreeClassifier(), features_new, output_label, cv=10)
    table_of_accuracies['dummy'] = cross_val_score(DummyClassifier(), features_new, output_label, cv=10)
    table_of_accuracies['knn'] = cross_val_score(KNeighborsClassifier(), features_new, output_label, cv=10)

    print(table_of_accuracies)

    print("Pair t test for naive and random")
    print(stats.ttest_rel(table_of_accuracies['naive'], table_of_accuracies['random']))
    print("Pair t test for naive and decision")
    print(stats.ttest_rel(table_of_accuracies['naive'], table_of_accuracies['decision']))
    print("Pair t test for naive and dummy")
    print(stats.ttest_rel(table_of_accuracies['naive'], table_of_accuracies['dummy']))
    print("Pair t test for naive and knn")
    print(stats.ttest_rel(table_of_accuracies['naive'], table_of_accuracies['knn']))
    print("Pair t test for random and dummy")
    print(stats.ttest_rel(table_of_accuracies['random'], table_of_accuracies['dummy']))
    print("Pair t test for random and knn")
    print(stats.ttest_rel(table_of_accuracies['random'], table_of_accuracies['knn']))
    print("Pair t test for random and decision")
    print(stats.ttest_rel(table_of_accuracies['random'], table_of_accuracies['decision']))
    print("Pair t test for dummy and knn")
    print(stats.ttest_rel(table_of_accuracies['dummy'], table_of_accuracies['knn']))
    print("Pair t test for dummy and decision")
    print(stats.ttest_rel(table_of_accuracies['dummy'], table_of_accuracies['decision']))
    print("Pair t test for K-NN and decision")
    print(stats.ttest_rel(table_of_accuracies['knn'], table_of_accuracies['decision']))
if __name__ == "__main__":
    main()




















#!/usr/bin/env python
# coding: utf-8

# In[47]:


# General imports
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data processing
from sklearn.model_selection import train_test_split
import imblearn

# Models
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# Fairlearn algorithms and utils
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, GridSearch, EqualizedOdds, DemographicParity

# Metrics
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio, equalized_odds_difference, demographic_parity_difference
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from scipy.stats import uniform


# AIF360 algorithms and utils
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.datasets import StandardDataset

from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# In[48]:


bab_df = pd.read_csv("turnover_babushkin_cleaned.csv")

categorical_names = ["GENDER", "INDUSTRY", "PROFESSION", "HIRE_SOURCE", "MANAGER_GENDER", "COMMUTE_TYPE"]
categorical_ixs = [bab_df.columns.get_loc(name) for name in categorical_names]

mappings = {}
for c in bab_df.columns:
    if c in categorical_names:
        if not all([x.isnumeric() for x in bab_df[c].values if type(x) == str]):
            categories = set(bab_df[c].values)
            if np.nan in categories:
                categories.remove(np.nan)
            mapping = dict(zip(categories, range(len(categories))))
            if c == "GENDER":
                print(mapping)
            
            mappings[c] = mapping
            bab_df[c] = bab_df[c].map(mapping)
    bab_df[c] = bab_df[c].astype('float64')
    
print(bab_df["LABELS"].value_counts())

bab_df["Senior"] = bab_df["AGE"].apply(lambda x : 0 if (x <= 26) else (1 if x > 40 else 2))
print(bab_df[["Senior", "GENDER", "LABELS"]].value_counts())

y = bab_df["LABELS"]
bab_df.drop("LABELS", axis=1, inplace=True)

X = bab_df
X


# ## Base classifiers

# In[23]:


def performance_report(n, model, sensitive_features, with_print=True):
    """Fit a model n times on the dataset and report its performance
    
    n: amount of iterations
    model: the classifier capable of making predictions
    probas: whether the model has a predict_proba() method, if so will output roc_auc scores
    with_print: whether to write the performance
    thresh: whether the model is a ThresholdOptimizer()"""
    
    f1_scores = []
    accuracy_scores = []
    roc_auc_scores = []
    eqodd_ratio_scores = []
    dempar_ratio_scores = []
    eqodd_group1_scores = []
    dempar_group1_scores = []
    eqodd_group2_scores = []
    dempar_group2_scores = []
    sens1, sens2 = sensitive_features
    for _ in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
        
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        test_probas = model.predict_proba(X_test)[:,1]
        
        
        roc_auc_scores.append(roc_auc_score(y_test, test_probas))    
        f1_scores.append(f1_score(y_test, test_preds))
        accuracy_scores.append(accuracy_score(y_test, test_preds))
        eqodd_ratio_scores = equalized_odds_ratio(y_test, test_preds, sensitive_features=X_test[sensitive_features])
        dempar_ratio_scores.append(demographic_parity_ratio(y_test, test_preds, sensitive_features=X_test[sensitive_features]))
        eqodd_group1_scores.append(equalized_odds_difference(y_test, test_preds, sensitive_features=X_test[sens1]))
        eqodd_group2_scores.append(equalized_odds_difference(y_test, test_preds, sensitive_features=X_test[sens2]))
        dempar_group1_scores.append(demographic_parity_difference(y_test, test_preds, sensitive_features=X_test[sens1]))
        dempar_group2_scores.append(demographic_parity_difference(y_test, test_preds, sensitive_features=X_test[sens2]))            
            
    if with_print:
        print(f"Performance statistics for {model} averaged over {n} times")
        print(f"F1-score: {np.mean(f1_scores)}")
        print(f"Accuracy score: {np.mean(accuracy_scores)}")
        print(f"ROC-AUC score: {np.mean(roc_auc_scores)}")
        print(f"Fairness statistics for {model} averaged over {n} times")
        print(f"Intersectional Equalized Odds ratio: {np.mean(eqodd_ratio_scores)}")
        print(f"Intersectional Demographic Parity ratio: {np.mean(dempar_ratio_scores)}")
        print(f"Equalized Odds difference for {sens1}: {np.mean(eqodd_group1_scores)}")
        print(f"Demographic Parity difference for {sens1}: {np.mean(dempar_group1_scores)}")
        print(f"Equalized Odds difference for {sens2}: {np.mean(eqodd_group2_scores)}")
        print(f"Demographic Parity difference for {sens2}: {np.mean(dempar_group2_scores)}")
        print("")
    
    return np.mean(f1_scores), np.mean(accuracy_scores), np.mean(roc_auc_scores), np.mean(eqodd_ratio_scores), np.mean(dempar_ratio_scores)


# In[24]:


# Train Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
log = LogisticRegression(solver='liblinear')
search = RandomizedSearchCV(estimator=log, param_distributions={"penalty":["l1", "l2"], 'C':uniform(loc=0, scale=2)}, n_iter=50, scoring="roc_auc", verbose=2, cv=3).fit(X_train, y_train)

log_clf = search.best_estimator_
f1_log, acc_log, rocauc_log, eqodd_log, dempar_log = performance_report(3, log_clf, ["GENDER", "Senior"])


# In[25]:


tree = RandomForestClassifier()
tree_params = {"n_estimators":[100, 250, 500, 1000], "criterion":["gini", "entropy"], 
               "max_features":["sqrt", "log2"], "max_depth":[25, 50, 75, 100, None],
               "min_samples_split":[2,5,10]}
search = RandomizedSearchCV(estimator=tree, param_distributions=tree_params, n_iter=50, scoring="roc_auc", cv=3, verbose=2).fit(X_train, y_train)
tree_clf = search.best_estimator_
f1_tree, acc_tree, rocauc_tree, eqodd_tree, dempar_tree = performance_report(3, tree_clf, ["GENDER", "Senior"])


# ## Threshold Optimization

# In[26]:


def performance_report_thresh(n, model, sensitive_features):
    f1_scores = []
    accuracy_scores = []
    roc_auc_scores = []
    eqodd_ratio_scores = []
    dempar_ratio_scores = []
    eqodd_group1_scores = []
    dempar_group1_scores = []
    eqodd_group2_scores = []
    dempar_group2_scores = []
    sens1, sens2 = sensitive_features
    for _ in range(n):
        # fit model with parameters
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
        model.fit(X_train, y_train, sensitive_features=X_train[sensitive_features])
        y_probas = model._pmf_predict(X_test, sensitive_features=X_test[sensitive_features])
        y_preds = model.predict(X_test, sensitive_features=X_test[sensitive_features])
        
        # store scores
        accuracy_scores.append(accuracy_score(y_test, y_preds))
        roc_auc_scores.append(roc_auc_score(y_test, y_probas[:,1]))
        eqodd_ratio_scores.append(equalized_odds_ratio(y_test, y_preds, sensitive_features=X_test[sensitive_features], method="between_groups"))
        dempar_ratio_scores.append(demographic_parity_ratio(y_test, y_preds, sensitive_features=X_test[sensitive_features], method="between_groups"))
        eqodd_group1_scores.append(equalized_odds_difference(y_test, y_preds, sensitive_features=X_test[sens1]))
        eqodd_group2_scores.append(equalized_odds_difference(y_test, y_preds, sensitive_features=X_test[sens2]))
        dempar_group1_scores.append(demographic_parity_difference(y_test, y_preds, sensitive_features=X_test[sens1]))
        dempar_group2_scores.append(demographic_parity_difference(y_test, y_preds, sensitive_features=X_test[sens2]))  
        f1_scores.append(f1_score(y_test, y_preds))

    print(f"Performance statistics for {model} averaged over {n} times")
    print(f"F1-score: {np.mean(f1_scores)}")
    print(f"Accuracy score: {np.mean(accuracy_scores)}")
    print(f"ROC-AUC score: {np.mean(roc_auc_scores)}")
    print(f"Fairness statistics for {model} averaged over {n} times")
    print(f"Intersectional Equalized Odds ratio: {np.mean(eqodd_ratio_scores)}")
    print(f"Intersectional Demographic Parity ratio: {np.mean(dempar_ratio_scores)}")
    print(f"Equalized Odds difference for {sens1}: {np.mean(eqodd_group1_scores)}")
    print(f"Demographic Parity difference for {sens1}: {np.mean(dempar_group1_scores)}")
    print(f"Equalized Odds difference for {sens2}: {np.mean(eqodd_group2_scores)}")
    print(f"Demographic Parity difference for {sens2}: {np.mean(dempar_group2_scores)}")
    print("")
    


# In[27]:


thresh_log_eq = ThresholdOptimizer(estimator=log_clf, constraints="equalized_odds", 
                       flip=True, grid_size=1000)

performance_report_thresh(5, thresh_log_eq, ["Senior", "GENDER"])


# In[28]:


thresh_log_dp = ThresholdOptimizer(estimator=log_clf, constraints="demographic_parity",
                       flip=True, grid_size=1000)

performance_report_thresh(5, thresh_log_dp, ["GENDER", "Senior"])


# In[29]:


thresh_tree_eq = ThresholdOptimizer(estimator=tree_clf, constraints="equalized_odds",
                       flip=True, grid_size=1000)

performance_report_thresh(5, thresh_tree_eq, ["Senior", "GENDER"])


# In[30]:


thresh_tree_dp = ThresholdOptimizer(estimator=tree_clf, constraints="demographic_parity",
                       flip=True, grid_size=1000)

performance_report_thresh(5, thresh_tree_dp, ["GENDER", "Senior"])


# ## Gradient Reductions

# In[31]:


def performance_report_reductions(n, model, sensitive_features):
    f1_scores = []
    accuracy_scores = []
    roc_auc_scores = []
    eqodd_ratio_scores = []
    dempar_ratio_scores = []
    eqodd_group1_scores = []
    dempar_group1_scores = []
    eqodd_group2_scores = []
    dempar_group2_scores = []
    sens1, sens2 = sensitive_features
    for _ in range(n):
        # fit model with parameters
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
        model.fit(X_train, y_train, sensitive_features=X_train[sensitive_features])
        y_probas = model._pmf_predict(X_test)
        y_preds = model.predict(X_test)
        
        # store scores
        accuracy_scores.append(accuracy_score(y_test, y_preds))
        roc_auc_scores.append(roc_auc_score(y_test, y_probas[:,1]))
        eqodd_ratio_scores.append(equalized_odds_ratio(y_test, y_preds, sensitive_features=X_test[sensitive_features], method="between_groups"))
        dempar_ratio_scores.append(demographic_parity_ratio(y_test, y_preds, sensitive_features=X_test[sensitive_features], method="between_groups"))
        eqodd_group1_scores.append(equalized_odds_difference(y_test, y_preds, sensitive_features=X_test[sens1]))
        eqodd_group2_scores.append(equalized_odds_difference(y_test, y_preds, sensitive_features=X_test[sens2]))
        dempar_group1_scores.append(demographic_parity_difference(y_test, y_preds, sensitive_features=X_test[sens1]))
        dempar_group2_scores.append(demographic_parity_difference(y_test, y_preds, sensitive_features=X_test[sens2]))  
        f1_scores.append(f1_score(y_test, y_preds))

    print(f"Performance statistics for {model} averaged over {n} times")
    print(f"F1-score: {np.mean(f1_scores)}")
    print(f"Accuracy score: {np.mean(accuracy_scores)}")
    print(f"ROC-AUC score: {np.mean(roc_auc_scores)}")
    print(f"Fairness statistics for {model} averaged over {n} times")
    print(f"Intersectional Equalized Odds ratio: {np.mean(eqodd_ratio_scores)}")
    print(f"Intersectional Demographic Parity ratio: {np.mean(dempar_ratio_scores)}")
    print(f"Equalized Odds difference for {sens1}: {np.mean(eqodd_group1_scores)}")
    print(f"Demographic Parity difference for {sens1}: {np.mean(dempar_group1_scores)}")
    print(f"Equalized Odds difference for {sens2}: {np.mean(eqodd_group2_scores)}")
    print(f"Demographic Parity difference for {sens2}: {np.mean(dempar_group2_scores)}")
    print("")
    


# In[32]:


sweep1 = ExponentiatedGradient(log_clf,
                   constraints=EqualizedOdds(),
                   max_iter=1000)

performance_report_reductions(1, sweep1, ["Senior", "GENDER"])


# In[33]:


sweep1 = ExponentiatedGradient(log_clf,
                   constraints=DemographicParity(),
                   max_iter=1000)

performance_report_reductions(1, sweep1, ["Senior", "GENDER"])


# In[34]:


sweep1 = ExponentiatedGradient(tree_clf,
                   constraints=EqualizedOdds(),
                   max_iter=50)

performance_report_reductions(1, sweep1, ["Senior", "GENDER"])


# In[35]:


sweep1 = ExponentiatedGradient(tree_clf,
                   constraints=DemographicParity(),
                   max_iter=50)

performance_report_reductions(1, sweep1, ["Senior", "GENDER"])


# ## Reject Option Classification

# In[37]:


X["LABELS"] = y
sens_ixs = [X.columns.get_loc("GENDER"), X.columns.get_loc("Senior")]
bab = StandardDataset(X, 
                               label_name="LABELS", 
                               favorable_classes=[1], 
                               protected_attribute_names=["Senior", "GENDER"], 
                               privileged_classes=[[1], [1]])

bab.features[:,sens_ixs]


# In[38]:


def performance_report_aif(n, aif_data, model, ml_model, sensitive_features):
    """Fit a model n times on the dataset and report its performance
    
    n: amount of iterations
    model: the bias mitigation algorithm
    ml_model: the classifier capable of making predictions
    with_print: whether to write the performance"""
    
    f1_scores = []
    accuracy_scores = []
    eqodd_diff_scores = []
    dempar_diff_scores = []
    roc_auc_scores = []
    eqodd_group1_scores = []
    dempar_group1_scores = []
    eqodd_group2_scores = []
    dempar_group2_scores = []
    sens1, sens2 = sensitive_features
    for _ in range(n):
        X_train_aif, X_test_aif = aif_data.split([0.7], shuffle=True)
        y_test_aif = X_test_aif.copy(deepcopy=True)

        # Prediction with the original model
        scores = np.zeros_like(X_test_aif.labels)
        scores = ml_model.predict_proba(X_test_aif.features)[:,1].reshape(-1,1)
        y_test_aif.scores = scores

        preds = np.zeros_like(X_test_aif.labels)
        preds = ml_model.predict(X_test_aif.features).reshape(-1,1)
        y_test_aif.labels = preds

        y_train_aif = X_train_aif.copy(deepcopy=True)

        # Prediction with the original model
        scores = np.zeros_like(X_train_aif.labels)
        scores = ml_model.predict_proba(X_train_aif.features)[:,1].reshape(-1,1)
        y_train_aif.scores = scores

        preds = np.zeros_like(X_train_aif.labels)
        preds = ml_model.predict(X_train_aif.features).reshape(-1,1)
        y_train_aif.labels = preds
        
        model.fit(X_train_aif, y_train_aif)
        model_predictions = model.predict(y_test_aif)
        test_preds = model_predictions.labels
        test_probas = model_predictions.scores
        
        f1_scores.append(f1_score(X_test_aif.labels, test_preds))
        accuracy_scores.append(accuracy_score(X_test_aif.labels, test_preds))
        roc_auc_scores.append(roc_auc_score(X_test_aif.scores, test_probas))
        eqodd_diff_scores.append(equalized_odds_ratio(X_test_aif.labels, test_preds, sensitive_features=X_test_aif.features[:,sensitive_features]))
        dempar_diff_scores.append(demographic_parity_ratio(X_test_aif.labels, test_preds, sensitive_features=X_test_aif.features[:,sensitive_features]))
        eqodd_group1_scores.append(equalized_odds_difference(X_test_aif.labels, test_preds, sensitive_features=X_test_aif.features[:,sens1]))
        eqodd_group2_scores.append(equalized_odds_difference(X_test_aif.labels, test_preds, sensitive_features=X_test_aif.features[:,sens2]))
        dempar_group1_scores.append(demographic_parity_difference(X_test_aif.labels, test_preds, sensitive_features=X_test_aif.features[:,sens1]))
        dempar_group2_scores.append(demographic_parity_difference(X_test_aif.labels, test_preds, sensitive_features=X_test_aif.features[:,sens2]))
    
    print(f"Performance statistics for {model} averaged over {n} times")
    print(f"F1-score: {np.mean(f1_scores)}")
    print(f"Accuracy score: {np.mean(accuracy_scores)}")
    print(f"ROC-AUC score: {np.mean(roc_auc_scores)}")
    print(f"Fairness statistics for {model} averaged over {n} times")
    print(f"Equalized Odds difference: {np.mean(eqodd_diff_scores)}")
    print(f"Demographic Parity difference: {np.mean(dempar_diff_scores)}")
    print(f"Equalized Odds difference for {sens1}: {np.mean(eqodd_group1_scores)}")
    print(f"Demographic Parity difference for {sens1}: {np.mean(dempar_group1_scores)}")
    print(f"Equalized Odds difference for {sens2}: {np.mean(eqodd_group2_scores)}")
    print(f"Demographic Parity difference for {sens2}: {np.mean(dempar_group2_scores)}")
    print("")
        
    return np.mean(f1_scores), np.mean(accuracy_scores), np.mean(roc_auc_scores), np.mean(eqodd_diff_scores), np.mean(dempar_diff_scores)
                                 


# In[39]:


roc_eq = RejectOptionClassification(unprivileged_groups=[{"Senior":0, "Senior":2, "GENDER":0}],
                                 privileged_groups=[{"Senior":1, "GENDER":1}],
                                 low_class_thresh=0.1,
                                high_class_thresh=0.9,
                                  num_class_thresh=100,
                                num_ROC_margin=100,
                                  metric_name="Average odds difference",
                                  metric_ub=0.05, metric_lb=-0.05)

f1_roc_log_eq, acc_roc_log_eq, roc_auc_roc_log_eq, eqodd_roc_log_eq, dempar_roc_log_eq = performance_report_aif(5, bab, roc_eq, log_clf, sens_ixs)


# In[40]:


roc_dp = RejectOptionClassification(unprivileged_groups=[{"Senior":0,"Senior":2, "GENDER":0}],
                                 privileged_groups=[{"Senior":1, "GENDER":1}],
                                 low_class_thresh=0.1,
                                high_class_thresh=0.9,
                                  num_class_thresh=100,
                                num_ROC_margin=100,
                                  metric_name="Statistical parity difference",
                                  metric_ub=0.05, metric_lb=-0.05)

f1_roc_log_dp, acc_roc_log_dp, roc_auc_roc_log_eq, eqodd_roc_log_dp, dempar_roc_log_dp = performance_report_aif(5, bab, roc_dp, log_clf, sens_ixs)


# In[41]:


f1_roc_tree_eq, acc_roc_tree_eq, roc_auc_roc_tree_eq, eqodd_roc_tree_eq, dempar_roc_tree_eq = performance_report_aif(5, bab, roc_eq, tree_clf, sens_ixs)


# In[42]:


f1_roc_tree_dp, acc_roc_tree_dp, roc_auc_roc_tree_dp, eqodd_roc_tree_dp, dempar_roc_tree_dp = performance_report_aif(5, bab, roc_dp, tree_clf, sens_ixs)


# ## Platt-scaling

# In[43]:


X.drop("LABELS", axis=1, inplace=True)


# In[44]:


def performance_report_platt(n, ml_model, sensitive_features):
    f1_scores = []
    accuracy_scores = []
    eqodd_diff_scores = []
    dempar_diff_scores = []
    roc_auc_scores = []
    eqodd_group1_scores = []
    dempar_group1_scores = []
    eqodd_group2_scores = []
    dempar_group2_scores = []
    sens1, sens2 = sensitive_features
    for _ in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
        
        for status in X["GENDER"].unique():
            for old in X["Senior"].unique():
                # calibrate for combination
                platt = CalibratedClassifierCV(ml_model, cv='prefit')
                X_train_iter = X_train[(X_train["Senior"] == old) & (X_train["GENDER"] == status)]
                platt.fit(X_train_iter, y.iloc[X_train_iter.index])

                X_test_iter = X_test[(X_test["Senior"] == old) & (X_test["GENDER"] == status)]
                
                # not possible first iteration
                try:
                    X_test_iter.drop(["Target", "Probas"], inplace=True, axis=1)
                    
                except KeyError:
                    pass
                
                test_probas_platt = platt.predict_proba(X_test_iter)[:,1]
                test_preds_platt = platt.predict(X_test_iter)
                
                X_test.loc[X_test_iter.index, "Target"] = test_preds_platt
                X_test.loc[X_test_iter.index, "Probas"] = test_probas_platt

        test_preds = X_test["Target"]
        test_probas = X_test["Probas"]
        f1_scores.append(f1_score(y_test, test_preds))
        accuracy_scores.append(accuracy_score(y_test, test_preds))
        roc_auc_scores.append(roc_auc_score(y_test, test_probas))
        eqodd_diff_scores = equalized_odds_ratio(y_test, test_preds, sensitive_features=X_test[sensitive_features])
        dempar_diff_scores.append(demographic_parity_ratio(y_test, test_preds, sensitive_features=X_test[sensitive_features]))
        eqodd_group1_scores.append(equalized_odds_difference(y_test, test_preds, sensitive_features=X_test[sens1]))
        eqodd_group2_scores.append(equalized_odds_difference(y_test, test_preds, sensitive_features=X_test[sens2]))
        dempar_group1_scores.append(demographic_parity_difference(y_test, test_preds, sensitive_features=X_test[sens1]))
        dempar_group2_scores.append(demographic_parity_difference(y_test, test_preds, sensitive_features=X_test[sens2]))  
        
        
    print(f"Performance statistics for Platt on {ml_model} averaged over {n} times")
    print(f"F1-score: {np.mean(f1_scores)}")
    print(f"Accuracy score: {np.mean(accuracy_scores)}")
    print(f"ROC-AUC score: {np.mean(roc_auc_scores)}")
    print(f"Fairness statistics for Platt on {ml_model} averaged over {n} times")
    print(f"Equalized Odds difference: {np.mean(eqodd_diff_scores)}")
    print(f"Demographic Parity difference: {np.mean(dempar_diff_scores)}")
    print(f"Equalized Odds difference for {sens1}: {np.mean(eqodd_group1_scores)}")
    print(f"Demographic Parity difference for {sens1}: {np.mean(dempar_group1_scores)}")
    print(f"Equalized Odds difference for {sens2}: {np.mean(eqodd_group2_scores)}")
    print(f"Demographic Parity difference for {sens2}: {np.mean(dempar_group2_scores)}")
    print("")
        
    return np.mean(f1_scores), np.mean(accuracy_scores), np.mean(roc_auc_scores), np.mean(eqodd_diff_scores), np.mean(dempar_diff_scores)


# In[45]:


f1_platt_tree, acc_platt_tree, roc_auc_platt_tree, eqodd_platt_tree, dempar_platt_tree = performance_report_platt(5, tree_clf, ["GENDER", "Senior"])


# In[46]:


f1_platt_log, acc_platt_log, roc_auc_platt_log, eqodd_platt_log, dempar_platt_log = performance_report_platt(5, log_clf, ["GENDER", "Senior"])


# In[ ]:





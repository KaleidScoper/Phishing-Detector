import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
import random

data = pd.read_csv("data48.csv")
count_y0 = data[data["CLASS_LABEL"] == 0].count()["CLASS_LABEL"]
count_y1 = data[data["CLASS_LABEL"] == 1].count()["CLASS_LABEL"]

features = ["NumDots","SubdomainLevel","PathLevel","UrlLength","NumDash","NumDashInHostname"
			,"AtSymbol","TildeSymbol","NumUnderscore","NumPercent","NumQueryComponents","NumAmpersand"
			,"NumHash","NumNumericChars","NoHttps","RandomString","IpAddress","DomainInSubdomains"
			,"DomainInPaths","HttpsInHostname","HostnameLength","PathLength","QueryLength","DoubleSlashInPath"
			,"NumSensitiveWords","EmbeddedBrandName","PctExtHyperlinks","PctExtResourceUrls","ExtFavicon"
			,"InsecureForms","RelativeFormAction","ExtFormAction","AbnormalFormAction","PctNullSelfRedirectHyperlinks"
			,"FrequentDomainNameMismatch","FakeLinkInStatusBar","RightClickDisabled","PopUpWindow","SubmitInfoToEmail"
			,"IframeOrFrame","MissingTitle","ImagesOnlyInForm","SubdomainLevelRT","UrlLengthRT","PctExtResourceUrlsRT"
			,"AbnormalExtFormActionR","ExtMetaScriptLinkRT","PctExtNullSelfRedirectHyperlinksRT"]


def calc_score_median(sample_set, var):
    var_list = list(np.unique(sample_set[var]))
    var_median_list = []
    for i in range(len(var_list) - 1):
        var_median = (var_list[i] + var_list[i + 1]) / 2
        var_median_list.append(var_median)
    return var_median_list

def choose_best_split(sample_set, var, min_sample):
    score_median_list = calc_score_median(sample_set, var)
    median_len = len(score_median_list)
    sample_cnt = sample_set.shape[0]
    sample1_cnt = sum(sample_set['CLASS_LABEL'])
    sample0_cnt = sample_cnt- sample1_cnt
    Gini = 1 - np.square(sample1_cnt / sample_cnt) - np.square(sample0_cnt / sample_cnt)
    bestGini = 0.0
    bestSplit_point = 0.0
    bestSplit_position = 0.0
    for i in range(median_len):
        left = sample_set[sample_set[var] < score_median_list[i]]
        right = sample_set[sample_set[var] > score_median_list[i]]
        left_cnt = left.shape[0]
        right_cnt = right.shape[0]
        left1_cnt = sum(left['CLASS_LABEL']);
        right1_cnt = sum(right['CLASS_LABEL'])
        left0_cnt = left_cnt - left1_cnt
        right0_cnt = right_cnt - right1_cnt
        left_ratio = left_cnt / sample_cnt
        right_ratio = right_cnt / sample_cnt

        if left_cnt < min_sample or right_cnt < min_sample:
            continue

        Gini_left = 1 - np.square(left1_cnt / left_cnt) - np.square(left0_cnt / left_cnt)
        Gini_right = 1 - np.square(right1_cnt / right_cnt) - np.square(right0_cnt / right_cnt)
        Gini_temp = Gini - (left_ratio * Gini_left + right_ratio * Gini_right)
        if Gini_temp > bestGini:
            bestGini = Gini_temp; bestSplit_point = score_median_list[i]
            if median_len > 1:
                bestSplit_position = i / (median_len - 1)
            else:
                bestSplit_position = i / median_len
        else:
            continue

    Gini = Gini - bestGini
    return bestSplit_point, bestSplit_position

def bining_data_split(sample_set, var, min_sample, split_list):
    split, position = choose_best_split(sample_set, var, min_sample)
    if split != 0.0:
        split_list.append(split)
    sample_set_left = sample_set[sample_set[var] < split]
    sample_set_right = sample_set[sample_set[var] > split]
    if len(sample_set_left) >= min_sample * 2 and position not in [0.0, 1.0]:
        bining_data_split(sample_set_left, var, min_sample, split_list)

    else:
        None

    if len(sample_set_right) >= min_sample * 2 and position not in [0.0, 1.0]:
        bining_data_split(sample_set_right, var, min_sample, split_list)

    else:
        None

def get_bestsplit_list2(sample_set, var):
    min_df = sample_set.shape[0] * 0.05
    sl = [0]
    split_list = []
    bining_data_split(sample_set, var, min_df, split_list)
    if sp[var] == True:
        sl = sl + split_list
        m = sample_set[var].max()
        sl = sorted(split_list)
        sl.append(m)
        return sl
    else:
        return 1

def get_bestsplit_list(sample_set, var):
    min_df = sample_set.shape[0] * 0.05
    split_list = []
    bining_data_split(sample_set, var, min_df, split_list)
    if len(split_list) < 1:
        print(var+"不需要决策树分箱")
    else:
        split_list.insert(0,0)
        split_list.append(sample_set[var].unique().max())
        return sorted(split_list)

def get_hand_bins(sample_set):
    hand_bins = {}
    for i in features:
        j = 0
        ll = len(data[i].unique())
        if ll > 3:

            hand_bins[i] = get_bestsplit_list(sample_set,i)
    return hand_bins

def get_iv1(data,col,bins):
    data = data[[col,"CLASS_LABEL"]].copy()
    data["cut"] = pd.cut(data[col],bins,include_lowest=True)
    bins_df = data.groupby("cut")["CLASS_LABEL"].value_counts().unstack()
    bins_df["good%"] = bins_df[0] / bins_df[0].sum()
    bins_df["bad%"] = bins_df[1] / bins_df[1].sum()
    woe = bins_df["woe"] = np.log((bins_df[1]/bins_df[1].sum())/(bins_df[0]/bins_df[0].sum()))
    iv = np.sum((bins_df["bad%"] - bins_df["good%"]) * bins_df.woe)
    return iv

def get_bins1(ft):
    count_y0 = data[data["CLASS_LABEL"] == 0].groupby(by=ft).count()["CLASS_LABEL"]
    count_y1 = data[data["CLASS_LABEL"] == 1].groupby(by=ft).count()["CLASS_LABEL"]
    num_bins = [*zip(sorted(data[ft].unique()), count_y1, count_y0)]
    return num_bins

def get_woe1(num_bins):
    columns = ["特征值", "count_1", "count_0"]
    df = pd.DataFrame(num_bins, columns=columns)

    df["total"] = df.count_0 + df.count_1 
    df["percentage"] = df.total / df.total.sum()  
    df["bad_rate"] = df.count_1 / df.total  
    df["good%"] = df.count_0 / df.count_0.sum()
    df["bad%"] = df.count_1 / df.count_1.sum()
    df["woe"] = np.log(df["bad%"] / df["good%"])
    return df

def get_iv2(df):
    rate = df["bad%"] - df["good%"]
    iv = np.sum(rate * df.woe)
    return iv
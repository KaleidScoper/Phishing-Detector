import graphviz
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
import pandas as pd
from sklearn.model_selection import train_test_split
import pydotplus
import os
import re
pd.set_option('display.max_columns',20)
pd.set_option('display.width', None)


data0_path="./data/legtimate-58w.txt"
data1_path="./data/phish-58w.txt"
data0 = pd.read_table(data0_path,header=None,names=['url'])
data1 = pd.read_table(data1_path,header=None,names=['url'])

# url各个部分分割为：协议(删除) 域名 路径 文件名 参数
def split_url(line, part):
    if line.startswith("http://"):
        line=line[7:]
    if line.startswith("https://"):
        line=line[8:]
    if line.startswith("ftp://"):
        line=line[6:]
    if line.startswith("www."):
        line = line[4:]
    slash_pos = line.find('/')
    if slash_pos > 0 and slash_pos < len(line)-1: # line = "fsdfsdf/sdfsdfsd"
        primarydomain = line[:slash_pos]
        path_argument = line[slash_pos+1:]
        path_argument_tokens = path_argument.split('/')
        pathtoken = "/".join(path_argument_tokens[:-1])
        last_pathtoken = path_argument_tokens[-1]
        if len(path_argument_tokens) > 2 and last_pathtoken == '':
            pathtoken = "/".join(path_argument_tokens[:-2])
            last_pathtoken = path_argument_tokens[-2]
        question_pos = last_pathtoken.find('?')
        if question_pos != -1:
            argument = last_pathtoken[question_pos+1:]
            pathtoken = pathtoken + "/" + last_pathtoken[:question_pos]
        else:
            argument = ""
            pathtoken = pathtoken + "/" + last_pathtoken
        last_slash_pos = pathtoken.rfind('/')
        sub_dir = pathtoken[:last_slash_pos]
        filename = pathtoken[last_slash_pos+1:]
    elif slash_pos == 0:    # line = "/fsdfsdfsdfsdfsd"
        primarydomain = line[1:]
        pathtoken = ""
        argument = ""
        sub_dir = ""
        filename = ""
    elif slash_pos == len(line)-1:   # line = "fsdfsdfsdfsdfsd/"
        primarydomain = line[:-1]
        pathtoken = ""
        argument = ""
        sub_dir = ""
        filename = ""
    else:      # line = "fsdfsdfsdfsdfsd"
        primarydomain = line
        pathtoken = ""
        argument = ""
        sub_dir = ""
        filename = ""
    if part == 'pd':
        return primarydomain
    elif part == 'path':
        return pathtoken
    elif part == 'argument':
        return argument
    elif part == 'sub_dir':
        return sub_dir
    elif part == 'filename':
        return filename
    else:
        return primarydomain, sub_dir,  filename, argument


# 1.UrlLength: url长度
def urlLength(url):
    return len(url)

# 2.NumDots: url中’.‘的数量
def numDots(url):
    cnt=0
    for str in url:
        if str == '.':
            cnt=cnt+1
    return cnt

# 3.NumDash: url中’-‘的数量
def numDash(url):
    cnt=0
    for str in url:
        if str == '-':
            cnt=cnt+1
    return cnt

# 4.AtSymbol: url中是否存在’@‘符号
def atSymbol(url):
    if '@' in url:
        return 1
    else:
        return 0

# 5.NumPercent: url中'%'的数量
def numPercent(url):
    cnt=0
    for str in url:
        if str == '%':
            cnt=cnt+1
    return cnt

# 6.IpAddress: url域名部分是否使用IP地址
def ipAddress(url):
    compile_rule = re.compile(r'\d+[\.]\d+[\.]\d+[\.]\d+')
    match_list = re.findall(compile_rule, url)
    if match_list:
        return 1
    else:
        return 0

# 7.NumSensitiveWords: url中敏感词的个数
def numSensitiveWords(url):
    cnt = 0
    url = url.lower()
    sensitive = ['secure', 'account', 'webscr', 'login']
    for i in range(len(sensitive)):
        if sensitive[i] in url:
             cnt=cnt+1
    return cnt

# 8.HostnameLength: url主机名部分的字符数量
def hostnameLength(url):
    pd = split_url(url,'pd')
    return len(pd)

# 9.pathLength
def pathLength(url):
    if len(url) < 25:
        return 0
    else:
        return 1

# 10.biaodian
def blength(url):
    biaodian = [',', '.', '?', '!', ':', '[', ']', '{', '}', ':']
    cnt=0
    for i in range(len(biaodian)):
        if biaodian[i] in url:
            cnt=cnt+1
    return cnt

# 11.numNumeric
def numNumeric(url):
    shuzi = ['0','1','2','3','4','5','6','7','8','9']
    cnt=0
    for ch in url:
        if ch in shuzi:
            cnt=cnt+1
    return cnt

# 12.https
def https(url):
    pr = url[:5]
    if pr == 'https':
        return 0
    else:
        return 1

# 13.SubDots: url主机名部分中’.‘的数量
def subDots(url):
    pd = split_url(url, 'pd')
    cnt=0
    for str in pd:
        if str == '.':
            cnt=cnt+1
    return cnt


def urls_artificial(path):
    data = pd.read_table(path, header=None, names=['url'])

    data['UrlLengeh'] = data.url.apply(lambda x: urlLength(x))
    data['NumDots'] = data.url.apply(lambda x: numDots(x))
    data['NumDash'] = data.url.apply(lambda x: numDash(x))
    data['AtSymbol'] = data.url.apply(lambda x: atSymbol(x))
    data['NumPercent'] = data.url.apply(lambda x: numPercent(x))
    data['IpAddress'] = data.url.apply(lambda x: ipAddress(x))
    data['NumSensitiveWords'] = data.url.apply(lambda x: numSensitiveWords(x))
    data['HostnameLength'] = data.url.apply(lambda x: hostnameLength(x))
    data['pathLength'] = data.url.apply(lambda x: pathLength(x))
    data['blength'] = data.url.apply(lambda x: blength(x))
    data['https'] = data.url.apply(lambda x: https(x))
    data['SubDots'] = data.url.apply(lambda x: subDots(x))
    data['numNumeric'] = data.url.apply(lambda x: numNumeric(x))

    return data



urls0 = urls_artificial(data0_path)
urls0['Target'] = 0
urls1 = urls_artificial(data1_path)
urls1['Target'] = 1
data = pd.concat([urls0,urls1],ignore_index=True)

X = data.iloc[:,1:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

dt = RFC()
dt.fit(X_train, y_train)

y_pre = dt.predict(X_test)
print(classification_report(y_test, y_pre, target_names=['legtimate', 'phish'], digits=5))
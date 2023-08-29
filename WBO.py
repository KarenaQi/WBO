import pandas as pd
import numpy as np
import math
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier

features_names=['ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']

def creat_gap(data, samplestep=10):
    datatime=data[["author_date_unix_timestamp"]]
    lst_gap=[]
    gap=len(datatime) // samplestep
    start_sample=0
    stop_sample=len(datatime)-1
    for i in range(samplestep):
        if i == samplestep-1:
            lst_gap.append((stop_sample - start_sample, 0))
        else: 
            lst_gap.append((stop_sample-start_sample, stop_sample-(start_sample+gap)+1))
            start_sample+=gap  
    return lst_gap

def clean_data(path):
    data = pd.read_csv(path)
    data = data[data['contains_bug'].isin([True, False])]
    data['fix'] = data['fix'].map(lambda x: 1 if x > 0 else 0)
    data['contains_bug'] = data['contains_bug'].map(lambda x: 1 if x > 0 else 0)
    for feature in features_names:
        data = data[data[feature].astype(str).str.replace(".", "", 1).str.isnumeric()]
    data.dropna(axis=0, how='any')
    data = data[(data["la"] > 0) & (data["ld"] > 0)]
    data = data.reset_index(drop=True)
    return data

def pre_process_test_data(path):
    data=clean_data(path)
    lst_gap = creat_gap(data)
    data = data[['ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp',
                 'contains_bug']]
    lst_test_data=[]
    for gap in lst_gap:
        before_gap=gap[0]
        after_gap=gap[1]
        labels = np.array(data.iloc[:, [-1]])
        features = np.array(data.iloc[:, :-1])
        # mean = features.mean(1).reshape(features.shape[0], 1)
        # std = features.std(1).reshape(features.shape[0], 1)
        # features = (features - mean) / std
        # features[np.isnan(features)] = 0
        maxnum = np.max(features, axis=0)
        minnum = np.min(features, axis=0)
        features = (features - minnum) / (maxnum - minnum)
        features[np.isnan(features)] = 0
        old_train_labels = labels[before_gap:labels.shape[0], :]
        old_train_features = features[before_gap:labels.shape[0], :]
        new_train_labels = labels[after_gap:before_gap+1, :]
        new_train_features = features[after_gap:before_gap+1, :]
        lst_test_data.append((old_train_labels, old_train_features,new_train_labels,new_train_features))
    return lst_test_data

def liner_time_weight(path,sigma):
    data = clean_data(path)
    lst_gap = creat_gap(data)
    time_sequence=np.array(data[["author_date_unix_timestamp"]])
    lst_time_sequence=[]
    sigma=sigma
    for gap in lst_gap:
        before_gap=gap[0]
        after_gap=gap[1]
        train_time_sequence=time_sequence[before_gap:time_sequence.shape[0], :]
        T=np.max(train_time_sequence, axis=0)
        train_time_sequence=1/(1+sigma*(T-train_time_sequence))
        lst_time_sequence.append(train_time_sequence)
    return lst_time_sequence

def gauss_time_weight(path, sigma):
    data = clean_data(path)
    lst_gap = creat_gap(data)
    time_sequence = np.array(data[["author_date_unix_timestamp"]])
    time_sequence=time_sequence-np.min(time_sequence, axis=0)
    lst_time_sequence = []
    T_max=1
    T_min=0
    for gap in lst_gap:
        before_gap=gap[0]
        after_gap=gap[1]
        train_time_sequence=time_sequence[before_gap:time_sequence.shape[0], :]
        T=np.max(train_time_sequence, axis=0)
        train_time_sequence=1/(np.sqrt(2*math.pi)*sigma)*np.power(math.e,(train_time_sequence-T)**2/(-2*sigma**2))
        lst_time_sequence.append((T_max-T_min)*train_time_sequence/train_time_sequence[0]+T_min)
    return lst_time_sequence

def exp_time_weight(path,sigma):
    data = clean_data(path)
    lst_gap = creat_gap(data)
    time_sequence = np.array(data[["author_date_unix_timestamp"]])
    time_sequence=time_sequence-np.min(time_sequence, axis=0)
    lst_time_sequence = []
    sigma=1/sigma
    T_max=1
    T_min=0
    for gap in lst_gap:
        before_gap=gap[0]
        after_gap=gap[1]
        train_time_sequence=time_sequence[before_gap:time_sequence.shape[0], :]
        T=np.max(train_time_sequence, axis=0)
        train_time_sequence=T_min+(T_max-T_min)*np.power(math.e,sigma*(train_time_sequence-T))
        lst_time_sequence.append(train_time_sequence/train_time_sequence[0])
    return lst_time_sequence

def get_new_sequence(temp_predict_y,test_labels,dict_exp,test_features):
    new_hard_sequence=np.array([0.5]*len(test_labels)) 
    list_value=[]
    list_index=[]
    true_count=len(test_labels[test_labels.flatten() == temp_predict_y.flatten()])
    flase_count=len(test_labels[test_labels.flatten() != temp_predict_y.flatten()])
    base_sigma=true_count/flase_count*400  
    for i in range(0,len(test_labels)):
        if test_labels[i] != temp_predict_y[i]:
            value=0
            for j in range(0,14):
                for map in dict_exp[features_names[j]].keys():
                    if map[0]<=test_features[i][j] and test_features[i][j]<= map[1]:
                        value+=dict_exp[features_names[j]][map]
            list_value.append(value)
            list_index.append(i)
    np_value=np.array(list_value)
    np_value=np.absolute(np_value)  
    maxnum = np.max(np_value, axis=0)
    minnum = np.min(np_value, axis=0)
    np_value = (np_value - minnum) / (maxnum - minnum)*base_sigma+ base_sigma 
    np_value[np.isnan(np_value)] = 0
    list_value=np_value.tolist()
    for i in range(len(list_index)):
        new_hard_sequence[list_index[i]]=list_value[i]
    return new_hard_sequence

def mix(features,labels,time,contribution):
    if len(labels[labels==0])>len([labels==1]):
        major_features,major_labels=features[labels.flatten()==0],labels[labels==0]
        minor_features, minor_labels = features[labels.flatten() == 1], labels[labels == 1]
        time,contribution=time[labels.flatten() == 1], contribution[labels.flatten() == 1]
        minor_class=1
    else:
        major_features,major_labels=features[labels.flatten()==1],labels[labels==1]
        minor_features, minor_labels = features[labels.flatten() == 0], labels[labels == 0]
        time, contribution = time[labels.flatten() == 0], contribution[labels.flatten() == 0]
        minor_class = 0
    merge_matrix=np.concatenate([time.reshape(-1,1),contribution.reshape(-1,1)],axis=1)
    weight_matrix=merge_matrix.copy()
    n=7
    for i in range(n-1):
        weight_matrix=np.concatenate([weight_matrix,merge_matrix],axis=1)
    merge_features=np.concatenate([minor_features,weight_matrix],axis=1)
    columns_name=[]
    for i in range(n*2+14):
        columns_name.append(str(i))
    data = pd.DataFrame(merge_features, columns=columns_name)
    mean_1=np.mean(weight_matrix,axis=1)
    data["weight"]=mean_1
    data=data.sort_values('weight', inplace=False, ascending=False)
    minor_features = np.array(data.iloc[:, 0:n*2+14])
    k = len(major_features)//(len(minor_features)*5//10+1)+1
    neigh = NearestNeighbors(n_neighbors=len(minor_features)-1)
    neigh.fit(minor_features)
    th=0
    new_features=np.empty((1,n*2+14))
    for i in range(len(minor_features)-1):
        for j in range(len(minor_features)):
            nns = neigh.kneighbors([minor_features[j]], len(minor_features), return_distance=False)
            new_features=np.concatenate([new_features,(minor_features[nns[0,i+1]].reshape(1,28)+minor_features[j])/2],axis=0)
            th+=1
            if th>(len(major_features)-len(minor_features)):
                break
    minor_features=np.concatenate([minor_features,new_features],axis=0)
    minor_features=minor_features[:,0:14]
    all_features=np.concatenate([minor_features,major_features],axis=0)
    all_labels= np.concatenate([np.array([minor_class]*len(minor_features)).reshape(-1,1),major_labels.reshape(-1,1)],axis=0)
    return all_features,all_labels

def train_and_predict(project, classifier):
    sigma=500
    path=r"./dataset/{}.csv".format(project)
    for i in range(30):
        all_data=pre_process_test_data(path)        
        all_time_sequence=gauss_time_weight(path,24*60*60*sigma/100)
        predict_y = np.array([])
        label_count = 0
        all_test_labels= np.array([])
        for count in range(0,len(all_data)):
            if count==0:
                continue
            train_labels, train_features, test_labels, test_features = all_data[count]
            if len(test_features) == 0:
                continue
            new_train_labels = train_labels
            new_train_features = train_features
            time_sequence = np.array(all_time_sequence[count]).flatten()
            if classifier == 'LR':
                clf = linear_model.LogisticRegression(solver='liblinear')
            if classifier == 'KNN':
                clf = KNeighborsClassifier(n_neighbors=10)
            if classifier == 'SVM':
                clf = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf',probability=True)
            if classifier == 'RF':
                clf = RandomForestClassifier()
            if classifier == 'MLP':
                clf = MLPClassifier()
            new_train_features, new_train_labels = mix(new_train_features, new_train_labels, time_sequence, time_sequence)
            new_train_features=new_train_features[:,0:14]
            maxnum = np.max(new_train_features, axis=0)
            minnum = np.min(new_train_features, axis=0)
            new_train_features = (new_train_features - minnum) / (maxnum - minnum)
            new_train_features[np.isnan( new_train_features)] = 0      
            clf.fit(new_train_features,new_train_labels.ravel())
            temp_predict_y = clf.predict_proba(test_features)
            temp_predict_y = temp_predict_y[:, 1:]    
            temp_predict_y = temp_predict_y.flatten()
            predict_y = np.concatenate([temp_predict_y, predict_y])
            label_count+=len(test_labels) 
            test_labels = test_labels.flatten()
            all_test_labels=np.concatenate([test_labels, all_test_labels])
        test_labels=all_test_labels
     
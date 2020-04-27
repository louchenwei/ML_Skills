import pandas as pd
import numpy as np
def clean_train_data(train_df):
    train_df['Cabin'].fillna('missing',inplace=True)
    train_df['Embarked'].fillna(train_df['Embarked'].mode()[0],inplace=True)
    train_df.groupby('Pclass')['Age'].mean().reset_index()
    def fill_age(pclass,age):
        if np.isnan(age):#注意：如果当前列为float/int类型，当前列中的None会被强制转为float的nan类型
            if pclass==1:
                return 39.159930
            elif pclass==2:
                return 29.506705
            else:
                return 24.816367
        else:
        return age
    train_df['Age']=train_df.apply(lambda row:fill_age(row['Pclass'],row['Age']),axis=1)
    return train_df
    

# 盖帽
def cap_floor(low_thresh,high_thresh,train_df):
    cap_dict={}
    for column in train_df.columns:
        if train_df[column].dtype==object:
            continue
        low_value=np.percentile(train_df[column],low_thresh)
        high_value=np.percentile(train_df[column],high_thresh)
        if low_value==high_value:#这里相当于不进行盖帽
            low_value=np.min(train_df[column])
            high_value=np.max(train_df[column])
        cap_dict[column]=[low_value,high_value]
    return cap_dict
def cap_update(column,x,cap_dict):
    if column not in cap_dict:
        return x
    if x>cap_dict[column][1]:
        return cap_dict[column][1]
    elif x<cap_dict[column][0]:
        return cap_dict[column][0]
    else:
        return x
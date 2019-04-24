
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('InputData.csv',usecols=[0,2,3,4,5,7,10,12,14,16,17,18])


# In[3]:


df.head()


# In[4]:


df.info()
df = df[((df['BlockDepDateTimeGmt'] != '.') & (df['ArrStn'] != 'AMS')) | (df['ArrStn'] == 'AMS')]
df = df[((df['BlockArrDateTimeGmt'] != '.') & (df['ArrStn'] == 'AMS')) | (df['ArrStn'] != 'AMS')]
df.info()


# In[5]:


month_dict = {'jan':'01','feb':'02','mar':'03','apr':'04','may':'05','jun':'06','jul':'07','aug':'08','sep':'09','oct':'10','nov':'11','dec':'12'}

def change_date(col,dateyes=bool):
    date_new = []
    time_new = []
    for date in df[col]:
        if len(date) == 13:
            date_i = date[:2]+month_dict[date[2:5].lower()]+date[5:7]
            time_i = date[8:].replace(':','')
            if time_i[0] == '2' and time_i[1] == '3':
                time_i = str(int(time_i[:2])-23) + time_i[2:]
                date_i = str(int(date_i[:2])+1) + date_i[2:]
            else:
                time_i = str(int(time_i[:2])+1) + time_i[2:]
                
            date_new.append(int(date_i))
            time_new.append(int(time_i))
        else:
            date_new.append(np.nan)
            time_new.append(np.nan)
    if dateyes:
        return date_new,time_new
    else:
        return time_new


# In[6]:


df['SchedDepDate'],df['SchedDepTime'] = change_date('SchedDepDateTimeGmt',dateyes=True)
df.drop('SchedDepDateTimeGmt',axis=1,inplace=True)

df['SchedArrDate'],df['SchedArrTime'] = change_date('SchedArrDateTimeGmt',dateyes=True)
df.drop('SchedArrDateTimeGmt',axis=1,inplace=True)

df['BlockArrDate'],df['BlockArrTime'] = change_date('BlockArrDateTimeGmt',dateyes=True)
df.drop('BlockArrDateTimeGmt',axis=1,inplace=True)

df['MsgDate'],df['MsgTime'] = change_date('MsgDateTimeGmt',dateyes=True)
df.drop('MsgDateTimeGmt',axis=1,inplace=True)

df['NewMsgDate'],df['NewMsgTime'] = change_date('NewDateTimeGmt',dateyes=True)
df.drop('NewDateTimeGmt',axis=1,inplace=True)

df['BlockDepDate'],df['BlockDepTime'] = change_date('BlockDepDateTimeGmt',dateyes=True)
df.drop('BlockDepDateTimeGmt',axis=1,inplace=True)


# In[7]:


df.to_csv('Semi-shaved data.csv',index=False)


# In[8]:


df_init = df.drop(['NewMessage','SchedDepDate','SchedArrDate'],axis=1)


# In[9]:


df_init.head()


# In[10]:


df_init.dropna(inplace=True)


# In[11]:


airline = []
for i in df_init['FltNbr']:
    airline.append(i[:2])

df_init['FltNbr'] = airline
df_init.head()


# In[12]:


arrival = pd.get_dummies(df_init['ArrStn'])


# In[13]:


df_init['ArrStn'] = arrival['AMS']


# In[14]:


from sklearn.preprocessing import LabelEncoder


# In[15]:


number = LabelEncoder()


# In[16]:


df_init['FltNbr'] = number.fit_transform(df_init['FltNbr'])


# In[17]:


df_init.head()


# In[18]:


# # df_init.loc[df_init['NewMessage'][0]]
# newmessage = []
# for msg in df_init['NewMessage']:
#     try:
#         newmessage.append(int(msg[:-1]))
    
#     except ValueError:
#         newmessage.append(np.nan)


# In[19]:


# df_init['NewMessage'] = newmessage


# In[20]:


df_init.reset_index(inplace=True)
df_init.drop('index',axis=1,inplace=True)


# In[21]:


newmessagetime = []
newmessagedate = []
blockdeptime = []
blockdepdate = []
blockarrtime = []
blockarrdate = []
for i in range(df_init['NewMsgDate'].count()):
    newmessagetime.append(int(df_init['NewMsgTime'][i]))
    newmessagedate.append(int(df_init['NewMsgDate'][i]))
    blockdeptime.append(int(df_init['BlockDepTime'][i]))
    blockdepdate.append(int(df_init['BlockDepDate'][i]))
    blockarrtime.append(int(df_init['BlockArrTime'][i]))
    blockarrdate.append(int(df_init['BlockArrDate'][i]))

df_init['NewMsgTime'] = newmessagetime
df_init['NewMsgDate'] = newmessagedate
df_init['BlockDepTime'] = blockdeptime
df_init['BlockDepDate'] = blockdepdate
df_init['BlockArrTime'] = blockarrtime
df_init['BlockArrDate'] = blockarrdate


# In[22]:


df_init.head()


# In[23]:


arrd = []
nmsgd = []
msgd = []
depd = []
for i in range(df_init['BlockArrDate'].count()):
    arrd.append(str(df_init['BlockArrDate'][i]))
    nmsgd.append(str(df_init['NewMsgDate'][i]))
    msgd.append(str(df_init['MsgDate'][i]))
    depd.append(str(df_init['BlockDepDate'][i]))
    

df_init['BlockArrDate'] = arrd
df_init['BlockDepDate'] = depd
df_init['NewMsgDate'] = nmsgd
df_init['MsgDate'] = msgd

df_init.head()


# In[24]:


bamonthyear = []
bdmonthyear = []
nmsgmonthyear = []
msgmonthyear = []
for i in range(len(df_init['BlockArrDate'])):
    bamonthyear.append(df_init['BlockArrDate'][i][-4:])
    bdmonthyear.append(df_init['BlockDepDate'][i][-4:])
    nmsgmonthyear.append(df_init['NewMsgDate'][i][-4:])
    msgmonthyear.append(df_init['MsgDate'][i][-4:])
    
df_init['BlockArrMY'] = bamonthyear
df_init['BlockDepMY'] = bdmonthyear
df_init['NewMsgMY'] = nmsgmonthyear
df_init['MsgMY'] = msgmonthyear


# In[25]:


df_init.info()
df_init = df_init[((df_init['BlockArrMY'] == df_init['NewMsgMY']) & (df_init['ArrStn']==1)) | (df_init['ArrStn']==0)]
df_init = df_init[((df_init['BlockDepMY'] == df_init['NewMsgMY']) & (df_init['ArrStn']==0)) | (df_init['ArrStn']==1)]
df_init = df_init[((df_init['MsgMY'] == df_init['NewMsgMY']))]
df_init.info()


# In[26]:


df_init.drop(['BlockArrMY','BlockDepMY','NewMsgMY','MsgMY'],axis=1,inplace=True)
df_init.head()


# In[27]:


df_init.reset_index(inplace=True)
df_init.drop('index',axis=1,inplace=True)


# In[28]:


def timedif(BlockArrT,BlockDepT,NewMsgT,BlockArrD,BlockDepD,NewMsgD):
    dif_col = []
    for i in range(len(df_init[BlockArrT])):
        time_y = str(df_init[NewMsgT][i])
        hour_y = float(time_y[0]) if len(str(df_init[NewMsgT][i])) < 4 else float(time_y[:2])
        try:
            minutes_y = float(time_y[1:]) if len(str(df_init[NewMsgT][i])) < 4 else float(time_y[2:])
        except ValueError:
            minutes_y = float(time_y)
        
        if df_init['ArrStn'][i] == 1:
            time_x = str(df_init[BlockArrT][i])
            hour_x = float(time_x[0]) if len(str(df_init[BlockArrT][i])) < 4 else float(time_x[:2])
            try:
                minutes_x = float(time_x[1:]) if len(str(df_init[BlockArrT][i])) < 4 else float(time_x[2:])
            except ValueError:
                minutes_x = float(time_x)
            if int(str(df_init[BlockArrD][i])[:2]) > int(str(df_init[NewMsgD][i])[:2]):
                dif = (((24+hour_x)-hour_y)*60) + (minutes_x-minutes_y)
            elif int(str(df_init[BlockArrD][i])[:2]) < int(str(df_init[NewMsgD][i])[:2]):
                dif = (((24+hour_y)-hour_x)*60) + (minutes_y-minutes_x)
            else:
                dif = ((hour_x-hour_y)*60) + (minutes_x-minutes_y)
        else:
            time_x = str(df_init[BlockDepT][i])
            hour_x = float(time_x[0]) if len(str(df_init[BlockDepT][i])) < 4 else float(time_x[:2])
            try:
                minutes_x = float(time_x[1:]) if len(str(df_init[BlockDepT][i])) < 4 else float(time_x[2:])
            except ValueError:
                minutes_x = float(time_x)
            if int(str(df_init[BlockDepD][i])[:2]) > int(str(df_init[NewMsgD][i])[:2]):
                dif = (((24+hour_x)-hour_y)*60) + (minutes_x-minutes_y)
            elif int(str(df_init[BlockDepD][i])[:2]) < int(str(df_init[NewMsgD][i])[:2]):
                dif = (((24+hour_y)-hour_x)*60) + (minutes_y-minutes_x)
            else:
                dif = ((hour_x-hour_y)*60) + (minutes_x-minutes_y)

        dif_col.append(dif)
    return dif_col


# In[29]:


df_init['TimeToFly'] = timedif('NewMsgTime','NewMsgTime','MsgTime','NewMsgDate','NewMsgDate','MsgDate')
df_init.head()


# In[30]:


df_init['Error'] = timedif('BlockArrTime','BlockDepTime','NewMsgTime','BlockArrDate','BlockDepDate','NewMsgDate')
df_init.head()


# In[31]:


err = []
ttofly = []
for i in range(df_init['Error'].count()):
    err.append(int(df_init['Error'][i]))
    ttofly.append(int(df_init['TimeToFly'][i]))

df_init['Error'] = err
df_init['TimeToFly'] = ttofly


# In[32]:


df_arrival = df_init[df_init['ArrStn'] == 1].drop(['ArrStn','ARR_COUNTRYNAME'],axis=1)
df_departure = df_init[df_init['ArrStn'] == 0].drop(['ArrStn','DEP_COUNTRYNAME'],axis=1)


# In[33]:


df_arrival['DepCountry'] = number.fit_transform(df_arrival['DEP_COUNTRYNAME'])
df_departure['ArrCountry'] = number.fit_transform(df_departure['ARR_COUNTRYNAME'])


# In[34]:


# df_arrival.drop(['AircraftType','BlockDepTime','BlockArrTime','SchedDepTime','SchedArrTime','BlockArrDate','MsgDate','NewMsgDate','BlockDepDate'],axis=1,inplace=True)
# df_departure.drop(['AircraftType','BlockDepTime','BlockArrTime','SchedDepTime','SchedArrTime','BlockArrDate','MsgDate','NewMsgDate','BlockDepDate'],axis=1,inplace=True)


# In[35]:


df_arrival.drop('DEP_COUNTRYNAME',axis=1,inplace=True)
df_departure.drop('ARR_COUNTRYNAME',axis=1,inplace=True)


# In[36]:


colsA = list(df_arrival.columns.values)
colsD = list(df_departure.columns.values)
colsA.pop(colsA.index('Error'))
colsD.pop(colsD.index('Error'))
df_arrival = df_arrival[colsA+['Error']]
df_departure = df_departure[colsD+['Error']]


# In[37]:


df_arrival.to_csv('Arriving Data.csv',index=False)
df_departure.to_csv('Departing Data.csv',index=False)


# In[38]:


# sns.pairplot(df_arrival,hue='DepCountry')


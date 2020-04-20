# data preprocessing: clean and split data

# coding=utf8
import pandas as pd
import warnings

month=1
appended_target_A= []
appended_target_B = []

while month < 5:

    df = pd.read_excel('./raw data/0%d.xlsx' % month)
    
    # filter warning messages
    warnings.filterwarnings("ignore", 'This pattern has match groups')
    
    # delete the first column
    df = df.drop(df.columns[0], axis = 1) 
    
    # split one column to 2 column
    df['name'], df['item'] =df['name'].str.split(' > ', 1).str
    
    # p/r flag   
    def PRflag(x):
      if "p" in x['flag']:
        return 'p'
      elif "r" in x['flag']:
        return 'r'
      elif "others" in x['flag']:
        return 'others'
    df['newflag'] = df.apply(PRflag, axis=1)
    
    #change time's datatype from string to datetime
    df['date_stngs'] = pd.to_datetime(pd.Series(df['time']))
    
    #week number
    #df['date_stngs'].dt.week
    df['Week_Number'] = df['date_stngs'].dt.week
    
    
    #depart dataframes
    target_A = df[df[u'header'].str.contains(u'target_A', regex=True)|df[u'content'].str.contains(u'target_A', regex=True)]
    target_B = df[df[u'header'].str.contains(u'target_B', regex=True)|df[u'content'].str.contains(u'target_B', regex=True)]
    
    # where to save it, usually as a .pkl 
    target_A.to_pickle('./target_A_0%d.pkl' % month)
    target_B.to_pickle('./target_B_0%d.pkl' % month)  
    
    #save as excel files
    writer1 = pd.ExcelWriter('./target_A_xlsx/target_A_0%d.xlsx' % month)
    target_A.to_excel(writer1,'Sheet1')
    writer1.save()
    writer2 = pd.ExcelWriter('./target_B_xlsx/target_B_0%d.xlsx' % month)
    target_B.to_excel(writer2,'Sheet1')
    writer2.save()
    
    #append
    appended_target_A.append(target_A) 
    appended_target_B.append(target_B)
    
    month +=1

#concat    
appended_target_A = pd.concat(appended_target_A, axis=0, ignore_index=True)
appended_target_A.to_pickle("./target_A_01-04.pkl")
appended_target_B = pd.concat(appended_target_A, axis=0, ignore_index=True)
appended_target_B.to_pickle("./target_B_01-04.pkl")      
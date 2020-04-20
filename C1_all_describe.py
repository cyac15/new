import pandas as pd

count_all_month = []
count_target_A_month = []
count_target_A_p_month = []
count_target_B_month = []
count_target_B_p_month = []
month = 1
month_name = []

while month < 5:
    month_name.append("month_0%d" % month)
    
    df = pd.read_excel('./raw data/0%d.xlsx' % month)
    count_all_month.append(df["header"].count())
 
    target_A = df[df[u'header'].str.contains(u'target_A', regex=True)|df[u'content'].str.contains(u'target_A', regex=True)]
    count_target_A_month.append(target_A["header"].count())
    
    target_A_post = target_A[target_A[u'PR'].str.contains(u'PR', regex=True)]
    count_target_A_p_month.append(target_A_post["header"].count())    
    
    target_B = df[df[u'header'].str.contains(u'target_B', regex=True)|df[u'content'].str.contains(u'target_B', regex=True)]
    count_target_B_month.append(target_B["header"].count())
    
    target_B_post = target_B[target_B[u'PR'].str.contains(u'P', regex=True)]
    count_target_B_p_month.append(target_B_post["header"].count())    
    
    month +=1

df_count = pd.DataFrame()
df_count["month"] = month_name
df_count["total_count"] = count_all_month
df_count["target_A"] = count_target_A_month
df_count["target_A_P"] = count_target_A_p_month
df_count["target_B"] = count_target_B_month
df_count["target_B_P"] = count_target_B_pt_month

writer = pd.ExcelWriter('count.xlsx')
df_count.to_excel(writer,'Sheet1')
writer.save()
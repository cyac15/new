import pandas as pd

#target_A = pd.read_pickle("./target_A/target_A_01-04.pkl")
#target_B = pd.read_pickle("./target_B/target_B_01-04.pkl")   
i=0
j=0
var_object = {'target_B': {}, 'target_A': {}}
# var_object = { target_A:{PRdescribe, ...}、target_B：{PRdescribe, ...}}

for key_object, obj in var_object.items():
    # key_object = target_A or target_B , obj =  {PR:describe, S:describe...}
    
    if ( key_object=='target_B' or key_object=='target_A' ):
        # obj['all'] = df
        df = pd.read_pickle('./'+key_object+'/'+key_object+'_01-04.pkl')
        obj['A'] = df.describe()
        obj['B'] = df.groupby(["B"]).describe()
        obj['C'] = df.groupby(["C"]).describe()
        obj['A-month'] = df.groupby([df['date_stngs'].dt.to_period("M").astype(str)]).describe()
        obj['B-month'] = df.groupby([df['date_stngs'].dt.to_period("M").astype(str),"B"]).describe()
        obj['B1-month'] = df.groupby(["B1", df['date_stngs'].dt.to_period("M").astype(str)]).describe()
        obj['C-month'] = df.groupby([df['date_stngs'].dt.to_period("M").astype(str),"C"]).describe()
        obj['B-week'] = df.groupby([df['date_stngs'].dt.to_period("W").astype(str),"B"]).describe()
        obj['C-week'] = df.groupby([df['date_stngs'].dt.to_period("W").astype(str),"C"]).describe()
        obj['A-week'] = df.groupby([df['date_stngs'].dt.to_period("W").astype(str), "A"]).describe()
                    
    i=i+1
    print('calculate '+key_object+': %d' % i)    
    
    
for key_object, obj in var_object.items(): 
    # key_object = target_A or target_B , obj = {PR:describe, S:describe...}
    writer = pd.ExcelWriter(key_object+'.xlsx')
    for key_element, element in obj.items():
        # key_element = sheet name , element = describe的表格   
        element.to_excel(writer, key_element)
        
        j=j+1
        print('write_to_excel '+key_object+' '+key_element+' : %d' % j) 
    writer.save()    
        
#####calculate the specific observations
# df_post.groupby([df_post['date_stngs'].dt.to_period("M").astype(str)])['typeA','typeB'].mean()    


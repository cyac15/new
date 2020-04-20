#make figures

import pandas as pd
import matplotlib.pyplot as plt

var_object = {'target_B': {}, 'target_A': {}, 
            'target_B_sourceA':{}, 'target_A_sourceA':{}, 
            'target_B_sourceB':{}, 'target_A_sourceB':{}, 
            'target_B_sourceC':{}, 'target_A_sourceC':{}}
name = ["target_B", "target_A"]
for key_object, obj in var_object.items():

    if ( key_object=='target_B' or key_object=='target_A' ):
        # obj['all'] = df
        df = pd.read_pickle('./'+key_object+'/'+key_object+'_01-04.pkl')
        df_1 = df.groupby(["tag",pd.DatetimeIndex(df['date_stngs']).normalize()])["header"].count()
        
        # examine the problems of fonts
        plt.rcParams['font.sans-serif']=['HEIT']
        plt.rcParams['axes.unicode_minus']=False

        plot_df = df_1.unstack('tag')
        plot_df = plot_df.fillna(0)  #if date is Null value
        fig, ax = plt.subplots()
        labels = ['typeA', 'typeB', 'typeC']
        for key in labels:
            if key=='typeA':
                ax = plot_df.reset_index().plot(ax=ax, kind='line', x='date_stngs', y=key, color='#FFC039', lw=1.2)         
            elif key=='typeB':
                ax = plot_df.reset_index().plot(ax=ax, kind='line', x='date_stngs', y=key, linestyle='dashed', color='#d0d0d0', lw=1.2)        
            elif key=='typeC':
                ax = plot_df.reset_index().plot(ax=ax, kind='line', x='date_stngs', y=key, color='#5BA9FF', lw=1.2)       
    
        lines, _ = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc='best')

        plt.ylabel("counts")
        plt.xlabel("time：1991/01/01-1991/03/30")
        plt.subplots_adjust(bottom=0.15)
        if key_object=="target_B":
            plt.title("target_B _________")
            plt.savefig(''+key_object+'_tag', dpi=1000)
        elif key_object=="target_A":
            plt.title("target_A _________")
            plt.savefig(''+key_object+'_tag', dpi=1000)
        
              
    elif ( key_object=='target_B_sourceA' or key_object=='target_A_sourceA' ):
        key_object_revise_sourceA = key_object.replace('_sourceA', '')
        df = pd.read_pickle('./'+key_object_revise_sourceA+'/'+key_object_revise_sourceA+'_01-04.pkl')
        df_post =  df[df[u'PRflag']=='P']
        df_sourceA =  df_post[df_post[u'source']=='sourceA']
        df_sourceA_count = df_sourceA.groupby(["tag",pd.DatetimeIndex(df_sourceA['date_stngs']).normalize()])["header"].count()
        
        # examine the problems of fonts
        plt.rcParams['font.sans-serif']=['HEIT']
        plt.rcParams['axes.unicode_minus']=False

        plot_df = df_sourceA_count.unstack('tag')
        plot_df = plot_df.fillna(0)
        fig, ax = plt.subplots()
        labels = ['typeA', 'typeＢ', 'typeＣ']
        for key in labels:
            if key=='typeA':
                ax = plot_df.reset_index().plot(ax=ax, kind='line', x='date_stngs', y=key, color='#FFC039', lw=1.2)         
            elif key=='typeＣ':
                ax = plot_df.reset_index().plot(ax=ax, kind='line', x='date_stngs', y=key, linestyle='dashed', color='#d0d0d0', lw=1.2)         
            elif key=='typeＢ':
                ax = plot_df.reset_index().plot(ax=ax, kind='line', x='date_stngs', y=key, color='#5BA9FF', lw=1.2)       
    
        lines, _ = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc='best')

        plt.ylabel("counts")
        plt.xlabel("time：1991/01/01-1991/03/30")
        plt.subplots_adjust(bottom=0.15)
        
        if key_object=="target_B_sourceA":
            plt.title("target_B SourceA tag trend")
            plt.savefig(''+key_object+'_emotiontag', dpi=1000)
        elif key_object=="target_A_sourceA":
            plt.title("target_A SourceA tag trend")
            plt.savefig(''+key_object+'_emotiontag', dpi=1000)
             
        
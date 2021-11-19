import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def boxplot_var_by_target(X_all,y_all, X_neigh, y_neigh, X_cust, main_cols, figsize=(15, 4)):
    '''Boxplot of each variable of main_cols for all, neighboor and customer'''

    df_all = pd.concat([X_all[main_cols], y_all.to_frame(name='TARGET')], axis=1)
    df_neigh = pd.concat([X_neigh[main_cols], y_neigh.to_frame(name='TARGET')], axis=1)
    df_cust = X_cust[main_cols].to_frame('values').reset_index()  # pd.Series to pd.DataFrame

    fig, ax = plt.subplots(figsize=figsize)

    # random sample of customers of the train set
    df_melt_all = df_all.reset_index()
    df_melt_all.columns = ['index'] + list(df_melt_all.columns)[1:]
    df_melt_all = df_melt_all.melt(id_vars=['index', 'TARGET'],  # SK_ID_CURR
                                   value_vars=main_cols,
                                   var_name="variables",
                                   value_name="values")
    sns.boxplot(data=df_melt_all, x='variables', y='values', hue='TARGET', linewidth=1,
                width=0.4, palette=['tab:green', 'tab:red'], showfliers=False, saturation=0.5,
                ax=ax)

    # 20 nearest neighbors
    df_melt_neigh = df_neigh.reset_index()
    df_melt_neigh.columns = ['index'] + list(df_melt_neigh.columns)[1:]
    df_melt_neigh = df_melt_neigh.melt(id_vars=['index', 'TARGET'],  # SK_ID_CURR
                                       value_vars=main_cols,
                                       var_name="variables",
                                       value_name="values")
    sns.swarmplot(data=df_melt_neigh, x='variables', y='values', hue='TARGET', linewidth=1,
                  palette=['darkgreen', 'darkred'], marker='o', edgecolor='k', ax=ax)

    # applicant customer
    df_melt_cust = df_cust.rename(columns={'index': "variables"})
    sns.swarmplot(data=df_melt_cust, x='variables', y='values', linewidth=1, color='y',
                  marker='o', size=10, edgecolor='k', label='applicant customer', ax=ax)

    # legend
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h[:5])

    plt.xticks(rotation=20, ha='right')
    plt.show()

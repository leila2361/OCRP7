import streamlit as st
import matplotlib
#matplotlib.use('TkAgg')
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt

import shap

from dashboard_plot import boxplot_var_by_target, radar_neigh



def main():
    # local API (à remplacer par l'adresse de l'application déployée)
    API_URL = "http://127.0.0.1:5000/api/"
    #API_URL = "https://oc-api-flask-mm.herokuapp.com/api/"
    # Get list of SK_IDS (cached)
    @st.cache
    def get_sk_id_list():
        # URL of the sk_id API
        SK_IDS_API_URL = API_URL + "sk_ids/"
        # Requesting the API and saving the response
        response = requests.get(SK_IDS_API_URL)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of SK_IDS from the content
        SK_IDS = pd.Series(content['data']).values
        return SK_IDS

    # Get Personal data (cached)
    @st.cache
    def get_data_cust(select_sk_id):
        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        PERSONAL_DATA_API_URL = API_URL + "data_cust/?SK_ID_CURR=" + str(select_sk_id)
        # save the response to API request
        response = requests.get(PERSONAL_DATA_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.Series
        data_cust = pd.Series(content['data']).rename(select_sk_id)
        data_cust_std = pd.Series(content['data_std']).rename(select_sk_id)
        return data_cust, data_cust_std

    # Get data from 10 nearest neighbors in train set (cached)
    @st.cache
    def get_data_neigh(select_sk_id):
        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        NEIGH_DATA_API_URL = API_URL + "neigh_cust/?SK_ID_CURR=" + str(select_sk_id)
        # save the response of API request
        response = requests.get(NEIGH_DATA_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame and pd.Series
        X_neigh = pd.DataFrame(content['X_neigh'])
        y_neigh = pd.Series(content['y_neigh']).rename('TARGET')
        X_neigh0 = pd.DataFrame(content['X_neigh0'])
        X_neigh1 = pd.DataFrame(content['X_neigh1'])
        return X_neigh, y_neigh, X_neigh0, X_neigh1

    # Get all data in train_std set (cached)
    @st.cache
    def get_all_proc_data_tr():
        # URL of the scoring API
        ALL_PROC_DATA_API_URL = API_URL + "all_proc_data_tr/"
        # save the response of API request
        response = requests.get(ALL_PROC_DATA_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.Series
        X_tr_std = pd.DataFrame(content['X_tr_std'])
        y_tr = pd.Series(content['y_train']).rename('TARGET')
        return X_tr_std, y_tr

    # Get scoring of one applicant customer (cached)
    @st.cache
    def get_cust_scoring(select_sk_id):
        # URL of the scoring API
        SCORING_API_URL = API_URL + "scoring_cust/?SK_ID_CURR=" + str(select_sk_id)
        # Requesting the API and save the response
        response = requests.get(SCORING_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # getting the values from the content
        score = content['proba']

        return score



    # Get the list of feature importances (according to lgbm classification model)
    @st.cache
    def get_features_importances():
        # URL of the aggregations API
        FEAT_IMP_API_URL = API_URL + "feat_imp"
        # Requesting the API and save the response
        response = requests.get(FEAT_IMP_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert back to pd.Series
        feat_imp = pd.Series(content['data']).sort_values(ascending=False)
        return feat_imp

    # Get the shap values of the customer and 20 nearest neighbors (cached)
    @st.cache
    def get_shap_values(select_sk_id):
        # URL of the scoring API
        GET_SHAP_VAL_API_URL = API_URL + "shap_values/?SK_ID_CURR=" + str(select_sk_id)
        # save the response of API request
        response = requests.get(GET_SHAP_VAL_API_URL)

        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame or pd.Series
        shap_val_df = pd.DataFrame(content['shap_val'])
        shap_val_trans = pd.Series(content['shap_val_cust_trans'])
        exp_value = content['exp_val']
        exp_value_trans = content['exp_val_trans']
        X_neigh_ = pd.DataFrame(content['X_neigh_'])
        return shap_val_df, shap_val_trans, exp_value, exp_value_trans, X_neigh_


    # Configuration of the streamlit page
    st.set_page_config(page_title='Loan application scoring dashboard',
                       page_icon='random',
                       layout='centered',
                       initial_sidebar_state='auto')

    # Display the title
    st.title('Loan application scoring dashboard')
    st.markdown("Leila Reille - Data Science project 7")




    # Select the customer's ID
    SK_IDS = get_sk_id_list()
    select_sk_id = st.sidebar.selectbox('Select SK_ID from list:', SK_IDS, key=18)
    st.write('You selected: ', select_sk_id)
    # get shap's values for customer and 20 nearest neighbors
    shap_val, shap_val_trans, exp_val, exp_val_trans, X_neigh_ = \
        get_shap_values(select_sk_id)

    # Get All Data relative to customer
    # Get personal data (unprocessed and preprocessed)
    X_cust, X_cust_std = get_data_cust(select_sk_id)  # pd.Series

    # Get 10 neighbors personal data (preprocessed)
    X_neigh, y_neigh, X_neigh0, X_neigh1 = get_data_neigh(select_sk_id)
    y_neigh = y_neigh.replace({0: 'repaid (neighbors)',
                               1: 'not repaid (neighbors)'})

    # Get all preprocessed training data
    X_tr_std_all, y_tr_all = get_all_proc_data_tr()  #
    y_tr_all = y_tr_all.replace({0: 'repaid (global)',
                                 1: 'not repaid (global)'})

    feat_imp = get_features_importances()
    main_cols = list(feat_imp.iloc[:12].index)


    def get_list_features(shap_val_trans, nb, key):

        n = st.slider("Nb of features to display",
                      min_value=2, max_value=12,
                      value=nb, step=None, format=None, key=key)

        if st.checkbox('Choose main features according to SHAP local interpretation for the applicant customer',
                       key=key):
            disp_cols = list(shap_val_trans.abs()
                             .sort_values(ascending=False)
                             .iloc[:n].index)
        else:
            disp_cols = list(get_features_importances().sort_values(ascending=False).iloc[:n].index)


        return disp_cols


    # SCORING

    if st.sidebar.checkbox("Scoring and model's decision", key=38):

        st.header("Scoring and model's decision")

        #  Get score
        score = get_cust_scoring(select_sk_id)

        # Display score (default probability)
        st.write('Default probability: {:.0f}%'.format(score * 100))


        # Compute decision according to the best threshold (True: loan refused)
        bool_cust = (score >= 0.5)

        if bool_cust is False:
            decision = "Loan granted"

        else:
            decision = "LOAN REJECTED"

        st.write('Decision:', decision)

        expander = st.expander("Concerning the classification model...")

        expander.write("The prediction was made using a LGBM (Light Gradient Boosting Model) \
classification model.")

        expander.write("The default model threshold is tuned to maximize a gain function that penalizes \
'false negative'/type II errors (i.e. granted loans that would not actually not be repaid) \
10 times more than 'false positive'/type I errors (i.e. rejected loans that would actually be repaid).")

        if st.checkbox('Show local interpretation', key=37):

            with st.spinner('SHAP waterfall plot creation in progress...'):

                nb_features = st.slider("Nb of features to display",
                                        min_value=2, max_value=12,
                                        value=10, step=None, format=None, key=14)

                # # get shap's values for customer and 10 nearest neighbors
                # shap_val, shap_val_trans, exp_val, exp_val_trans, X_neigh_ = \
                #     get_shap_values(select_sk_id)

                # draw the graph (only for the customer with scaling)
                shap.plots._waterfall.waterfall_legacy(exp_val_trans,
                                                       shap_val_trans,
                                                       feature_names=list(shap_val_trans.index),
                                                       max_display=nb_features,
                                                       show=True)
                fig = plt.gcf()
                fig.set_size_inches((10, nb_features/2))
                # Plot the graph on the dashboard
                st.pyplot(fig)

                st.markdown('_SHAP waterfall plot for the applicant customer._')

                expander = st.expander("Concerning the SHAP waterfall plot...")

                expander.write("The above waterfall plot displays \
explanations for the individual prediction of the applicant customer.\
The bottom of a waterfall plot starts as the expected value of the model output \
(i.e. the value obtained if no information (features) were provided), and then \
each row shows how the positive (red) or negative (blue) contribution of \
each feature moves the value from the expected model output over the \
background dataset to the model output for this prediction.")
                expander.write("NB: for LGBM classification model, the sum of the SHAP values is NOT \
usually the final probability prediction but log odds values. \
On this graph, a simple scaling has been performed so that the base value \
represents the probability obtained if no particular information is given, and the sum of \
the values on the arrows is the predicted probability of default on the loan (non repayment).")



    # CUSTOMER'S DATA

    if st.sidebar.checkbox("Customer's data"):

        st.header("Customer's data")

        format_dict = {'cust prepro': '{:.2f}',
                       '10 neigh (mean)': '{:.2f}',
                       '10k samp (mean)': '{:.2f}'}


        if st.checkbox('Show comparison with 20 neighbors (10 granted, 10 rejected)', key=31):
            # Concatenation of the information to display
            mean_n0 = X_neigh0[main_cols].mean(axis=0).to_frame('mean_n0')
            mean_n1 = X_neigh1[main_cols].mean(axis=0).to_frame('mean_n1')
            df_cust_radar = pd.concat([X_cust_std[main_cols],
                                       mean_n0.T, mean_n1.T], axis=0)
            for col in main_cols:
                df_cust_radar[col] = df_cust_radar[col].fillna(0)
            df_cust_radar['Mean_Ext_Source'] = df_cust_radar[['EXT_SOURCE_3',
                                                              'EXT_SOURCE_2',
                                                              'EXT_SOURCE_1']].mean(axis=1)
            df_cust_radar.drop(['EXT_SOURCE_3','EXT_SOURCE_2','EXT_SOURCE_1'],
                               axis=1, inplace=True)
            df_cust_radar['DAYS_EMPLOYED'] = df_cust_radar['DAYS_EMPLOYED'].apply(lambda x: -x)
            df_cust_radar['DAYS_BIRTH'] = df_cust_radar['DAYS_BIRTH'].apply(lambda x: -x)
            radar_cols = ["Mean_Ext_Source", "PAYMENT_RATE", "DAYS_EMPLOYED",
                         "AMT_ANNUITY", "DAYS_BIRTH", "INSTAL_DPD_MEAN"]
            fig = radar_neigh(df_cust_radar[radar_cols].copy())
            st.write(fig)

        # Display only personal_data
        df_display = pd.concat([X_cust.loc[main_cols].rename('cust'),
                                    X_cust_std.loc[main_cols].rename('cust_std')],
                                   axis=1)  # all pd.Series

        # Display at last
        st.dataframe(df_display.style.format(format_dict)
                     .background_gradient(cmap='seismic',
                                          axis=0, subset=None,
                                          text_color_threshold=0.2,
                                          vmin=-1, vmax=1)
                     .highlight_null('lightgrey'))

        st.markdown('_Data used by the model, for the applicant customer,\
            for the 10 nearest neighbors and for a random sample_')

        expander = st.expander("Concerning the data table...")
        # format de la première colonne objet ?

        expander.write("The above table shows the value of each feature:\
  \n- _cust_: values of the feature for the applicant customer,\
unprocessed \n- _10 neigh (mean)_: mean of the values of each feature \
  for the 10 nearest neighbors of the applicant customer in the training \
set  \n- _10k samp (mean)_: mean of the  values of each feature \
for a random sample of customers from the training set.")


    # BOXPLOT FOR MAIN 10 VARIABLES


    if st.sidebar.checkbox('Boxplots of the main features', key=23):
        st.header('Boxplot of the main features')

        disp_box_cols = get_list_features(shap_val_trans, 10, key=42)

        fig = boxplot_var_by_target(X_tr_std_all, y_tr_all, X_neigh, y_neigh,
                                             X_cust_std, disp_box_cols, figsize=(10, 4))

        st.write(fig)
        st.markdown('_Dispersion of the main features for random sample,\
 10 nearest neighbors and applicant customer_')

        expander = st.expander("Concerning the dispersion graph...")

        expander.write("These boxplots show the dispersion of the preprocessed features values\
 used by the model to make a prediction. The green boxplot are for the customers that repaid \
their loan, and red boxplots are for the customers that didn't repay it.Over the boxplots are\
 superimposed (markers) the values\
 of the features for the 10 nearest neighbors of the applicant customer in the training set. The \
 color of the markers indicate whether or not these neighbors repaid their loan. \
 Values for the applicant customer are superimposed in yellow.")



    # FEATURES' IMPORTANCE (SHAP VALUES) for 10 nearest neighbors


    if st.sidebar.checkbox("Importance of the features", key=29):

        st.header("Comparison of local and global feature importance")

        plot_choice = []
        if st.checkbox('Add global feature importance', value=True, key=25):
            plot_choice.append(0)
        if st.checkbox('Add local feature importance for the nearest neighbors', key=28):
            plot_choice.append(1)
        if st.checkbox('Add local feature importance for the applicant customer', key=26):
            plot_choice.append(2)

        # # get shap's values for customer and 10 nearest neighbors
        # shap_val_df, _, _, _, X_neigh_ = get_shap_values(select_sk_id)

        disp_box_cols = get_list_features(shap_val_trans, 10, key=42)
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 3))

        global_imp = get_features_importances().loc[disp_box_cols]
        mean_shap_neigh = shap_val.mean().loc[disp_box_cols]
        shap_val_cust = shap_val.iloc[-1].loc[disp_box_cols]

        from sklearn.preprocessing import MinMaxScaler
        minmax = MinMaxScaler()

        df_disp = pd.concat([global_imp.to_frame('global'),
                             mean_shap_neigh.to_frame('neigh'),
                             shap_val_cust.to_frame('cust')],
                            axis=1)
        df_disp_sc = pd.DataFrame(minmax.fit_transform(df_disp),
                                  index=df_disp.index,
                                  columns=df_disp.columns)
        plot_choice = [0] if plot_choice == [] else plot_choice
        disp_df_choice = df_disp_sc.iloc[:, plot_choice]
        disp_df_choice.sort_values(disp_df_choice.columns[0]).plot.barh(width=0.8, ec='k',
                                                                        color=['navy', 'red', 'orange'],
                                                                        ax=ax2)

        plt.legend()
        fig2.set_size_inches((8, len(disp_box_cols) / 2))
        # # Plot the graph on the dashboard
        st.pyplot(fig2)

        st.markdown('_Relative global and local feature importance_')

        expander = st.expander("Concerning the comparison of local and global feature importance...")

        expander.write("The global feature importance in blue (computed globally for the lgbm model when trained on the training set)\
 are compared in the above bar chart to the local importance (SHAP values) of each features for the 10 nearest neighbors (red) \
 or for the applicant customer (orange).")
        expander.write("NB: For easier visualization the values in each bar plot has be scaled with min-max \
scaling (0 stands for min and 1 for max value).")

        if st.checkbox('Detail of SHAP feature importances for the applicant customer neighbors', key=24):
            st.header("Local feature importance of the features for the nearest neighbors")

            plt.clf()
            nb_features = st.slider("Nb of features to display",
                                    min_value=2, max_value=12,
                                    value=10, step=None, format=None, key=13)

            # draw the graph
            shap.summary_plot(shap_val.values,  # shap values
                              X_neigh_.values,  # data (np.array)
                              feature_names=list(X_neigh_.columns),  # features name of data (order matters)
                              max_display=nb_features,  # nb of displayed features
                              show=True)
            fig = plt.gcf()
            fig.set_size_inches((10, nb_features / 2))
            # Plot the graph on the dashboard
            st.pyplot(fig)

            st.markdown('_Beeswarm plot showing the SHAP values for each feature and for \
the nearest neighbors of the applicant customer_')

            expander = st.expander("Concerning the SHAP waterfall plot...")

            expander.write("The above beeswarm plot displays \
the SHAP values for the individual prediction of the applicant customer and his \
20 nearest neighbors for each feature, as well a the corresponding value of this feature (colormap).")






if __name__ == '__main__':
    main()




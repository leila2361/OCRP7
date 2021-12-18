import os


from pickle import load
import pandas as pd

from flask import Flask, jsonify, request
import json
from sklearn.neighbors import NearestNeighbors



path = os.path.join('data', 'df_api.csv')
df_api = pd.read_csv(path, index_col='SK_ID_CURR')
train_df = df_api[df_api.TARGET.notnull()]
X_train = train_df[[col for col in train_df.columns if col != 'TARGET']]
y_train = train_df ['TARGET']
test_df = df_api[df_api.TARGET.isnull()]
X_test = test_df.drop('TARGET', axis=1)
path = os.path.join('data', 'X_tr_std.csv')
X_tr_std = pd.read_csv(path, index_col='SK_ID_CURR')
path = os.path.join('data', 'X_te_std.csv')
X_te_std = pd.read_csv(path, index_col='SK_ID_CURR')

path = os.path.join('data', 'feat_sel_df.csv')
feat_sel = pd.read_csv(path)
important_features = list(feat_sel['feature'].values[:12])

path = os.path.join('data', 'results.csv')
results = pd.read_csv(path, index_col='SK_ID_CURR')
# model
path = os.path.join('model', 'LightGBMModel2.pkl')
with open(path, 'rb') as file:
    lgbm = load(file)

path = os.path.join('data', 'shap_val_all.csv')
shap_val_all = pd.read_csv(path, index_col=0)
# expected value
path = os.path.join('data', 'expected_val.pkl')
with open(path, 'rb') as file:
    expected_val = load(file)


#instantiate Flask object
app = Flask(__name__)

# view when API is launched
# Test local : http://127.0.0.1:5000

@app.route("/")
def index():
    return "API loaded, models and data loaded, data computed…"

@app.route('/api/sk_ids/')
def sk_ids():
    # Extract list of all the 'SK_ID_CURR' ids in the X_test dataframe
    sk_ids = pd.Series(list(X_test.index.sort_values()))
    # Convert pd.Series to JSON
    sk_ids_json = json.loads(sk_ids.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
    		        'data': sk_ids_json})

@app.route('/api/feat_imp/')
def send_feat_imp():
    feat_imp = pd.Series(feat_sel['importance'].values, index=feat_sel['feature'].values)
    feat_imp_json = json.loads(feat_imp.to_json())
    # Return the processed data as a json object
    return jsonify({'status': 'ok',
    		        'data': feat_imp_json})

@app.route('/api/data_cust/')
def data_cust():
    # Parse the http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # Get the personal data for the customer (pd.Series)
    X_cust_ser = X_test.loc[sk_id_cust, :]
    X_cust_std = X_te_std.loc[sk_id_cust,:]
    # Convert the pd.Series (df row) of customer's data to JSON
    X_cust_json = json.loads(X_cust_ser.to_json())
    X_cust_std_json = json.loads(X_cust_std.to_json())
    # Return the cleaned data
    return jsonify({'status': 'ok',
    				'data': X_cust_json,
                    'data_std': X_cust_std_json})

# find 10 nearest neighbors among the training set
def get_df_neigh(sk_id_cust):
    # get data of 10 nearest neigh in the X_train dataframe (pd.DataFrame)
    df_neigh = X_tr_std[important_features].copy()
    neigh = NearestNeighbors(n_neighbors=10)
    neigh.fit(df_neigh)
    X_cust = X_te_std.loc[sk_id_cust: sk_id_cust]
    cust_list = X_cust[important_features]
    idx = neigh.kneighbors(X=cust_list,
                           n_neighbors=10,
                           return_distance=False).ravel()
    nearest_cust_idx = list(X_tr_std.iloc[idx].index)
    X_neigh_df = X_tr_std.loc[nearest_cust_idx, :]
    y_neigh = y_train.loc[nearest_cust_idx]
    return X_neigh_df, y_neigh

def get_df_neigh_target(sk_id_cust):
    # get data of 20 nearest neigh in the X_train dataframe (pd.DataFrame)
    # 10 with loan granted and 10 with loan rejected
    df_neigh = X_tr_std[important_features].copy()
    df_neigh['TARGET'] = y_train
    df_neigh0 = df_neigh[df_neigh['TARGET']==0.].copy()
    df_neigh0.drop('TARGET',axis=1, inplace=True)
    df_neigh1 = df_neigh[df_neigh['TARGET']==1.].copy()
    df_neigh1.drop('TARGET',axis=1, inplace=True)
    neigh = NearestNeighbors(n_neighbors=10)
    neigh.fit(df_neigh0)
    X_cust = X_te_std.loc[sk_id_cust: sk_id_cust]
    cust_list = X_cust[important_features]
    idx0 = neigh.kneighbors(X=cust_list,
                           n_neighbors=10,
                           return_distance=False).ravel()
    nearest_cust_idx0 = list(X_tr_std.iloc[idx0].index)
    X_neigh_0 = X_tr_std.loc[nearest_cust_idx0, :]
    neigh.fit(df_neigh1)
    idx1 = neigh.kneighbors(X=cust_list,
                            n_neighbors=10,
                           return_distance=False).ravel()
    nearest_cust_idx1 = list(X_tr_std.iloc[idx1].index)
    X_neigh_1 = X_tr_std.loc[nearest_cust_idx1, :]
    return X_neigh_0, X_neigh_1

@app.route('/api/neigh_cust/')
def neigh_cust():
    # Parse the http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # return the nearest neighbors
    X_neigh_df, y_neigh = get_df_neigh(sk_id_cust)
    X_neigh_0, X_neigh_1 = get_df_neigh_target(sk_id_cust)
    # Convert the customer's data to JSON
    X_neigh_json = json.loads(X_neigh_df.to_json())
    y_neigh_json = json.loads(y_neigh.to_json())
    X_neigh0_json = json.loads(X_neigh_0.to_json())
    X_neigh1_json = json.loads(X_neigh_1.to_json())
    # Return the cleaned data jsonified
    return jsonify({'status': 'ok',
    				'X_neigh': X_neigh_json,
    				'y_neigh': y_neigh_json,
                    'X_neigh0': X_neigh0_json,
                    'X_neigh1': X_neigh1_json})

@app.route('/api/all_proc_data_tr/')
def all_proc_data_tr():
    # get all data from X_train, X_test and y_train data
    # and convert the data to JSON
    X_tr_std_json = json.loads(X_tr_std.to_json())
    y_train_json = json.loads(y_train.to_json())
    X_te_std_json = json.loads(X_te_std.to_json())
    # Return the cleaned data jsonified
    return jsonify({'status': 'ok',
    				'X_tr_std': X_tr_std_json,
    				'y_train': y_train_json,
                   'X_te_std': X_te_std_json})



@app.route('/api/scoring_cust/')
def scoring_cust():
    # Parse http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # read the score of the customer
    proba_cust = results.loc[sk_id_cust, 'Proba_impaye']
    # Return score
    return jsonify({'status': 'ok',
    		        'SK_ID_CURR': sk_id_cust,
    		        'proba': proba_cust})

#Importing the logit function for the base value transformation
from scipy.special import expit
# Conversion of shap values from log odds to probabilities
def shap_transform_scale(shap_values, expected_value, model_prediction):
    #Compute the transformed base value, which consists in applying the logit function to the base value
    expected_value_transformed = expit(expected_value)
    #Computing the original_explanation_distance to construct the distance_coefficient later on
    original_explanation_distance = sum(shap_values)
    #Computing the distance between the model_prediction and the transformed base_value
    distance_to_explain = model_prediction - expected_value_transformed
    #The distance_coefficient is the ratio between both distances which will be used later on
    distance_coefficient = original_explanation_distance / distance_to_explain
    #Transforming the original shapley values to the new scale
    shap_values_transformed = shap_values / distance_coefficient
    return shap_values_transformed, expected_value_transformed

@app.route('/api/shap_values/')
# get shap values of the customer and 20 nearest neighbors
# Test local : http://127.0.0.1:5000/api/shap_values/?SK_ID_CURR=100128
# Test Heroku : https://oc-api-flask-mm.herokuapp.com/api/shap_values/?SK_ID_CURR=100128
def shap_values():
    # Parse http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # return the nearest neighbors
    X_neigh, y_neigh = get_df_neigh(sk_id_cust)
    X_cust = X_te_std.loc[sk_id_cust].to_frame(sk_id_cust).T
    X_neigh_ = pd.concat([X_neigh, X_cust], axis=0)
    # prepare the shap values of nearest neighbors + customer
    shap_val_neigh_ = shap_val_all.loc[X_neigh_.index]
    # Conversion of shap values from log odds to probabilities of the customer's shap values
    shap_t, exp_t = shap_transform_scale(shap_val_all.loc[sk_id_cust],
                                         expected_val,
                                         results.loc[sk_id_cust, 'Proba_impaye'])
    shap_val_cust_trans = pd.Series(shap_t,
                                    index=X_neigh_.columns)
    # Converting the pd.Series to JSON
    X_neigh__json = json.loads(X_neigh_.to_json())
    shap_val_neigh_json = json.loads(shap_val_neigh_.to_json())
    shap_val_cust_trans_json = json.loads(shap_val_cust_trans.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'shap_val': shap_val_neigh_json, # pd.DataFrame
                    'shap_val_cust_trans': shap_val_cust_trans_json, # pd.Series
                    'exp_val': expected_val,
                    'exp_val_trans': exp_t,
                    'X_neigh_': X_neigh__json})

####################################
# if the api is run and not imported as a module
if __name__ == "__main__":
    app.run(debug=False)


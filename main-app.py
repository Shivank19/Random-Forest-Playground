import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston, load_iris, load_wine
#****************************************************

#Page Layout.......expanded to full width
st.set_page_config(
    page_title='Random Forest PlayGround',
    layout = "wide",
    )

#****************************************************

#Model
def build_model(df):

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    #Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - split_size)/100)
    
    st.markdown('**1.2 Data Splits**')
    st.write('Training Set')
    st.info(X_train.shape)
    st.write('Test Set')
    st.info(X_test.shape)

    st.markdown('**1.3 Variable Details**')
    st.write('X Variable')
    st.info(list(X.columns))
    st.write('y Variable')
    st.info(y.name)

    rf = RandomForestRegressor(
        n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs
        )
    rf.fit(X_train, y_train)

    st.subheader('2. Model Performance')

    st.markdown('**2.1 Training Set**')
    y_pred_train = rf.predict(X_train)
    st.write('Coefficient of Determination ($R^2$): ')
    st.info( r2_score(y_train, y_pred_train) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(y_train, y_pred_train) )

    st.markdown('**2.2 Test Set**')
    y_pred_test = rf.predict(X_test)
    st.write('Coefficient of Determination ($R^2$): ')
    st.info( r2_score(y_test, y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(y_test, y_pred_test) )

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())

#****************************************************

st.write("""
# The Machine Learning App
In this implementation, the **Random Forest** algorithm is used to build a model.
Play around with the hyperparameters to explore how the algorithm works.
""")

#****************************************************

#Sidebar
## Upload Data
with st.sidebar.header('1. Upload your Data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV data file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://people.sc.fsu.edu/~jburkardt/data/csv/biostats.csv)
""")

## Parameter Settings
with st.sidebar.header('2. Parameter Settings'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1 Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 50)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2.2 General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed Number(random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

#****************************************************
#Dataset Selector
def dataSel(d):
    boston = load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    Y = pd.Series(boston.target, name='response')
    df = pd.concat( [X,Y], axis=1 )

    st.markdown('The Boston housing dataset is used as the example.')
    st.write(df.head(5))

    build_model(df)

#****************************************************
#Main Panel
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1. Dataset Overview**')
    st.write(df)
    build_model(df)
else:
    st.info('Please upload a CSV dataset or choose from the following sample datasets.')
    if st.button('Press to use Boston Dataset'):
        dataSel(load_boston)
    if st.button('Press to use Iris Dataset'):
        dataSel(load_iris)
    if st.button('Press to use Diabetes Dataset'):
        dataSel(load_diabetes)
    if st.button('Press to use Wine Dataset'):
        dataSel(load_wine)


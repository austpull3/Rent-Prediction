
import streamlit as st
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold
#import statsmodels.formula.api as smf
#import statsmodels.stats.api as sms
#import statsmodels.api as sm
#from statsmodels.formula.api import ols
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from scipy import stats, linalg

import streamlit as st 
import os
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
df = pd.read_csv("USAHousing2.csv")
import random
random.seed(42)
df = df.sample(frac = 0.80)

df = df[(df.type != 'condo') & (df.type!= 'duplex') & (df.type != 'manufactured') 
        & (df.type!= 'cottage/cabin') & (df.type != 'loft') & (df.type!= 'flat') & (df.type!= 'in-law') &
        (df.type!= 'land') &(df.type!= 'assisted living')]

#Missing values
df.isna().sum()
#Fill missing values with the mode
df["parking_options"] = df["parking_options"].fillna(df["parking_options"].mode()[0])
df["laundry_options"] = df["laundry_options"].fillna(df["laundry_options"].mode()[0])
df.fillna(0, inplace=True)

#Remove irrelevant features
#df.drop(columns = ["id", "url", "region_url", "image_url", "description"],axis = 1,inplace = True)


#Fix dataset so that it does not include Zero for price and sqfeet
df=df[df["price"] > 200 ]
df=df[df["sqfeet"]>= 200]

df=df[df["price"]<2000]

df= df[(df["sqfeet"]<= 1600) & (df['sqfeet'] > 300)]

df=df[df["beds"]<= 3]

df=df[df["baths"]<= 3.5]

df= df[(df["lat"]< 55) & (df['lat'] > 20)]

df= df[(df["long"]< -20) & (df['long'] > -110)] 
 
def main_page():
    st.snow()
    
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local('/Users/austinpullar/Desktop/Christmas2.webp')    
    
    st.markdown("# Welcome to the Rent Predictor 🏘🎄")
    st.markdown("### In the sidebar to the left there are several pages that can take you through the machine learning side of the predictor.")
    st.markdown("### If you wish to go straight to the predictor select that page.")
    from PIL import Image 
    image1 = Image.open('/Users/austinpullar/Desktop/house3.jpeg')
    st.image(image1)

    st.markdown("# ENJOY!")
    st.sidebar.markdown("# Welcome!🎈❄️")


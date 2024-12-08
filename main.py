import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import linear_model, metrics
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
import requests

st.title("Cardiovascular Disease Analysis")
"**Created by Arseniy Arsentyev**\n\n"

df = pd.read_csv('cardio.csv', sep=';')

"""
! IMPORTANT NOTE !
You can find a nice interface for my FastAPI api at the bottom of the page"""

st.header("Introduction")
"""In this project I worked with cardiovascular disease dataset. Later, I added two custom columns to the dataset.
Also, after exploratory analysis, I formed and tested a hypothesis that logistic even simple logistic regression on
some features can provide satisfactory accuracy"""

"Below you can see original dataset's field description:"
st.markdown('''
| Feature                                       | IsObjective         | Column Name | Type                                             |
|:----------------------------------------------|:--------------------|:------------|:-------------------------------------------------|
| Age                                           | Objective Feature   | age         | int (days)                                       |
| Height                                        | Objective Feature   | height      | int (cm)                                         |
| Weight                                        | Objective Feature   | weight      | float (kg)                                       |
| Gender                                        | Objective Feature   | gender      | categorical code                                 |
| Systolic blood pressure                       | Examination Feature | ap_hi       | int                                              |
| Diastolic blood pressure                      | Examination Feature | ap_lo       | int                                              |
| Cholesterol                                   | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
| Glucose                                       | Examination Feature | gluc        | 1: normal, 2: above normal, 3: well above normal |
| Smoking                                       | Subjective Feature  | smoke       | binary                                           |
| Alcohol intake                                | Subjective Feature  | alco        | binary                                           |
| Physical activity                             | Subjective Feature  | active      | binary                                           |
| Presence or absence of cardiovascular disease | Target Variable     | cardio      | binary                                           |
''')

import streamlit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv('cardio.csv', sep=';')

st.header("Part 1: Descriptive Statistics")

"First five rows of the dataset:"
st.write(df.head())

"And corresponding columns' datatypes:"
df.dtypes

"Correlation Matrix:"
corr = df.corr()
heat_map = sns.heatmap(corr, annot=True, fmt='.1f', cmap="coolwarm")
st.pyplot(heat_map.get_figure())

"""Notice that the strongest correlations exist between age, weight, and cholesterol relative to cardio disease probability.
In addition, there is some correlation in the disease probability with distolic and systolic pressure"""

st.header("Part 2: Data Cleaning and Transformation")

"Null-value Counts:"
st.write(df.isnull().sum())
'''There are no null-values. Moreover, all values are in the form suitable for further processing. So, let's proceed to transformation.\n'''

'''I am adding two new columns to the original dataset:\n
    - BMI: persons bmi (weight / height squared)\n
    - Pressure index: composite value consisting of normalized ap_lo, ap_hi, and age. In the proportions relative to correlation with target column'''

df.age //= 365
'''First of all, I converted the age from days to years\n
Here are the stats for the columns we will use:'''
st.write(df[['age', 'height', 'weight', 'ap_hi', 'ap_lo']].describe())



df['BMI'] = df['weight'] / ((df['height'] / 100.0)**2)
"The following is a graph showing useful statistics for the new BMI column:"
plt.clf()
box_plot = sns.boxplot(data=df, x="cardio", y="BMI", fill=False, showfliers=False, hue="cardio")
plt.ylim(13, 45)
st.pyplot(box_plot.get_figure())
'''
As one can see, higher BMI (i.e. obesity) is the usual feature of cardiovascular disease'''

"Next, interact with the plot of distolic and systolic pressure with the third axis of age"
condition = (40 < df['ap_lo']) & (df['ap_lo'] < 120) & (50 < df['ap_hi']) & (df['ap_hi'] < 180)
df_sample = df[condition].sample(n=900, random_state=11)
fig = px.scatter_3d(
    df_sample,
    x='age',
    y='ap_lo',
    z='ap_hi',
    color='cardio',
    opacity=0.7,
    color_continuous_scale=['white', 'red']
)

fig.update_layout(
    width=1000,
    height=500,
)

fig.update_coloraxes(showscale=False)

st.plotly_chart(fig)

"After carefully looking for patterns. It is clear that there are direct relation between ap_lo and ap_hi."
"Nevertheless, there also exist correlation of growing medical pressure with whether person has heart disease. So, lets combine all three features into one coefficient."
"\nHere are relative percentages of affecting probability of disease:"


corr_to_cardio = df.corr()['cardio'][['age', 'ap_lo', 'ap_hi']]
total = corr_to_cardio.sum()
coef = [corr_to_cardio['age'] / total,
        corr_to_cardio['ap_lo'] / total,
        corr_to_cardio['ap_hi'] / total]
st.write('age:', round(coef[0] * 100, 2), '%')
st.write('distolic pressure:', round(coef[1] * 100, 2), '%')
st.write('systolic pressure:', round(coef[2] * 100, 2), '%')

def normalize_column(data, column):
    return ((data[column] - data[column].min()) /
            (data[column].max() - data[column].min()))

df['pressure_index'] = ((coef[0] * normalize_column(df, 'age')) +
                        (coef[1] * normalize_column(df, 'ap_lo')) +
                        (coef[2] * normalize_column(df, 'ap_hi')))
"Using these coefficients I added new column 'pressure_index' combining three features together"
plt.clf()
hist = sns.histplot(data=df, x='pressure_index', hue='cardio', bins=20)
plt.xlim(0.2, 0.7)
st.pyplot(hist.get_figure())

"Given the correlations to the 'cardio' column"
st.write(df.corr()['cardio'][['age', 'ap_lo', 'ap_hi', 'pressure_index']])
"Compound column shows an increasing correlation."

st.header("Part 3: Further Analysis")
"""
This section presents visual analyses of cardiovascular disease related to key health metrics. 
The first two plots show cholesterol and glucose levels, categorized by normal and abnormal ranges. 
The third plot examines age distribution by disease status, while the fourth explores BMI distribution.
These plots helped me to propose my hypothesis.
"""
plt.clf()
chol_gluc_plot = sns.countplot(
    data=df,
    x='cholesterol',
    hue='cardio',
    palette='coolwarm',
    alpha=0.9
)
chol_gluc_plot.set_xticklabels(["Normal", "Above Normal", "Well Above Normal"])
chol_gluc_plot.set_title("Cholesterol Levels vs Cardiovascular Disease")
st.pyplot(chol_gluc_plot.get_figure())

plt.clf()
gluc_plot = sns.countplot(
    data=df,
    x='gluc',
    hue='cardio',
    palette='coolwarm',
    alpha=0.9
)
gluc_plot.set_xticklabels(["Normal", "Above Normal", "Well Above Normal"])
gluc_plot.set_title("Glucose Levels vs Cardiovascular Disease")
st.pyplot(gluc_plot.get_figure())

plt.clf()
age_dist = sns.kdeplot(
    data=df, x='age', hue='cardio', fill=True, alpha=0.5, palette='coolwarm'
)
age_dist.set_title("Age Distribution by Cardiovascular Disease")
st.pyplot(age_dist.get_figure())

plt.clf()
bmi_dist = sns.kdeplot(
    data=df, x='BMI', hue='cardio', fill=True, alpha=0.5, palette='coolwarm'
)
bmi_dist.set_title("BMI Distribution by Cardiovascular Disease")
plt.xlim(15, 50)
st.pyplot(bmi_dist.get_figure())

st.header("Part 4: Hypothesis Testing")
'''Hypothesis: Only with four features: bmi, pressure_index, cholesterol, and glucose, it is possible to rather precise predict
the cardiovascular disease using logistic regression.'''

X = df[['BMI', 'pressure_index', 'cholesterol', 'gluc']]
y = df.cardio

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=11)

logic_regr = linear_model.LogisticRegression()
logic_regr.fit(X_train, y_train)
y_pred = logic_regr.predict(X_test)

"Confusion Matrix:"
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
plt.clf()
cnf = sns.heatmap(cnf_matrix, annot=True)
st.pyplot(cnf.get_figure())

"ROC Curve:"
plt.clf()
y_pred_proba = logic_regr.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.legend(loc=4)
st.pyplot(plt)

st.write("Accuracy: ", accuracy_score(y_test, y_pred))
"""
So, my hypothesis was right. Even a simple logistic regression on my custom-calculated parameters showed an acceptable accuracy (~64%)"""

############ FASTAPI PART ############
url = "http://66.151.32.244//api/"

st.header("Appendix: FastAPI")
st.markdown("#### Get Sample from the Dataset")
start = st.number_input("Start Index", min_value=0, value=0)
limit = st.number_input("Number of Rows to Fetch", min_value=1, value=10)
filter_cardio = st.selectbox("Filter by Cardio", options=[None, 0, 1],
                             format_func=lambda x: "All" if x is None else x)
if st.button("Fetch Data"):
    params = {"start": start, "limit": limit}
    if filter_cardio is not None:
        params["filter_cardio"] = filter_cardio
    response = requests.get(f"{url}/data/", params=params)
    if response.status_code == 200:
        data = response.json()
        st.write(pd.DataFrame(data))
    else:
        st.error("!!! ERROR !!!")


st.markdown("#### Append New Row")
new_entry = {
    "age": st.number_input("Age (in years)", min_value=0, value=40),
    "height": st.number_input("Height (in cm)", min_value=50, value=170),
    "weight": st.number_input("Weight (in kg)", min_value=10.0, value=70.0),
    "gender": st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female"),
    "ap_hi": st.number_input("Systolic Pressure", min_value=50, value=120),
    "ap_lo": st.number_input("Diastolic Pressure", min_value=30, value=80),
    "cholesterol": st.selectbox("Cholesterol Level", options=[1, 2, 3],
                                format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x - 1]),
    "gluc": st.selectbox("Glucose Level", options=[1, 2, 3],
                         format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x - 1]),
    "smoke": st.selectbox("Smoking", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
    "alco": st.selectbox("Alcohol Intake", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
    "active": st.selectbox("Physical Activity", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
    "cardio": st.selectbox("Cardiovascular Disease", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
}

if st.button("Submit New Entry"):
    response = requests.post(f"{url}/data/", json=new_entry)
    if response.status_code == 200:
        st.success("Successfully Added!")
    else:
        st.error("!!! ERROR !!!")
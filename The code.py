import pandas as pd
import sklearn.metrics as met
import sklearn.linear_model as lm
data=pd.read_csv("gender_height_weight (1).csv")
data.replace(["Male"],0,inplace=True)
data.replace(["Female"],1,inplace=True)
x,y=data[["Height","Weight"]],data["Gender"]
print(x)
print(y)
model=lm.LogisticRegression()
model.fit(x,y)

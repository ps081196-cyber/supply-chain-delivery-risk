import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

FEATURES = ["distance_km","promised_days","carrier_rating","weather_severity","traffic_index","order_value","dispatch_delay_hours"]

def generate_shipments(n=1200, seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "distance_km":rng.integers(20,2500,n),
        "promised_days":rng.integers(1,8,n),
        "carrier_rating":rng.uniform(2.5,5,n).round(2),
        "weather_severity":rng.integers(0,6,n),
        "traffic_index":rng.uniform(0,100,n).round(1),
        "order_value":rng.uniform(200,30000,n).round(2),
        "dispatch_delay_hours":rng.uniform(0,36,n).round(1)})
    pressure=df.distance_km/700-df.promised_days*.65-df.carrier_rating*.55+df.weather_severity*.5+df.traffic_index/40+df.dispatch_delay_hours/12
    probability=1/(1+np.exp(-(pressure-1.2)))
    df["late_delivery"]=(rng.random(n)<probability).astype(int)
    df.insert(0,"shipment_id",[f"SHP-{i+1:05d}" for i in range(n)])
    return df

def train_model(df):
    x_train,x_test,y_train,y_test=train_test_split(df[FEATURES],df["late_delivery"],test_size=.25,random_state=42,stratify=df["late_delivery"])
    model=RandomForestClassifier(n_estimators=250,max_depth=10,class_weight="balanced",random_state=42)
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
    return model, accuracy_score(y_test,pred), classification_report(y_test,pred,output_dict=True)

def score_shipments(model, df):
    out=df.copy()
    out["late_risk_probability"]=model.predict_proba(out[FEATURES])[:,1]
    out["risk_level"]=pd.cut(out["late_risk_probability"],[-1,.35,.65,1],labels=["Low","Medium","High"])
    return out.sort_values("late_risk_probability",ascending=False)

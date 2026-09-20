import plotly.express as px
import streamlit as st
from core import FEATURES, generate_shipments, score_shipments, train_model

st.set_page_config(page_title="Delivery Risk Predictor",page_icon="🚚",layout="wide")
st.title("🚚 Supply Chain Delivery Risk Predictor")
st.caption("Prioritize shipments before they become SLA breaches")
training=generate_shipments()
model,accuracy,report=train_model(training)
uploaded=st.file_uploader("Upload shipment CSV",type="csv")
if uploaded:
    import pandas as pd
    current=pd.read_csv(uploaded)
else:
    current=generate_shipments(250,7).drop(columns=["late_delivery"])
scored=score_shipments(model,current)
a,b,c=st.columns(3)
a.metric("Model accuracy",f"{accuracy:.1%}")
b.metric("High-risk shipments",int((scored.risk_level=="High").sum()))
c.metric("Average risk",f"{scored.late_risk_probability.mean():.1%}")
importance=sorted(zip(FEATURES,model.feature_importances_),key=lambda x:x[1],reverse=True)
st.plotly_chart(px.bar(x=[x[1] for x in importance],y=[x[0] for x in importance],orientation="h",title="Risk drivers"),use_container_width=True)
st.subheader("Priority exception queue")
st.dataframe(scored[["shipment_id","risk_level","late_risk_probability"]+FEATURES],use_container_width=True)
st.download_button("Download scored shipments",scored.to_csv(index=False),"delivery_risk_scores.csv","text/csv")

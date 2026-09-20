from core import generate_shipments, train_model, score_shipments

def test_pipeline_scores_every_shipment():
    df=generate_shipments(200,1)
    model,accuracy,_=train_model(df)
    scored=score_shipments(model,df.drop(columns=["late_delivery"]))
    assert len(scored)==200
    assert scored.late_risk_probability.between(0,1).all()
    assert accuracy>=0.5

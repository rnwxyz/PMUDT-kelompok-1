from keras.models import load_model
import numpy as np

def predict_sentiment(row):
    # Define classification pipelines for each model
    model_ilc_feateng = load_model('./model/machine/ilc_feateng_0.89_.h5')
    model_kickandy_feateng = load_model('model/machine/kick_andy_feateng_0.85_.h5')
    model_hitamputih_feateng = load_model('model/machine/hitam_putih_feateng_0.88_.h5')
    model_matanajwa_feateng = load_model('model/machine/mata_najwa_feateng_0.85_.h5')
    
    text_vectorized = row.drop(['Text Tweet', 'Predicted Cluster', 'Cluster Name']).values.astype(np.float32)
    text_vectorized = text_vectorized.reshape(1, -1)

    if row['Predicted Cluster'] == 0:
        print("Cluster = 0")
        raw_prediction = model_ilc_feateng.predict(text_vectorized)[0]
    elif row['Predicted Cluster'] == 1:
        print("Cluster = 1")
        raw_prediction = model_hitamputih_feateng.predict(text_vectorized)[0]
    elif row['Predicted Cluster'] == 2:
        print("Cluster = 2")
        raw_prediction = model_kickandy_feateng.predict(text_vectorized)[0]
    elif row['Predicted Cluster'] == 3:
        print("Cluster = 3")
        raw_prediction = model_matanajwa_feateng.predict(text_vectorized)[0]
    else:
        return -1  # Handle the case where the cluster is not in the expected range
    print("Raw Prediction:", raw_prediction)


    # Find the class with the highest probability
    threshold = 0.5
    predicted_class = 1 if raw_prediction[0] >= threshold else 0
    print("Predicted Prediction:", predicted_class)

    return predicted_class
from flask import Flask, render_template, request
import gradio as gr
import pickle
import numpy as np

# Load the pre-trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Define a function to make predictions using the loaded model


def predict_price(income, house_age, num_rooms, num_bedrooms, population):
    inputs = np.array([income, house_age, num_rooms,
                      num_bedrooms, population]).reshape(1, -1)
    prediction = model.predict(inputs)
    return prediction[0]


# Define Gradio interface
iface = gr.Interface(fn=predict_price,
                     inputs=["number", "number", "number", "number", "number"],
                     outputs="number",
                     title="Housing Price Predictor",
                     description="Enter the features to predict the housing price.")

# Define Flask route


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        income = float(request.form["income"])
        house_age = float(request.form["house_age"])
        num_rooms = float(request.form["num_rooms"])
        num_bedrooms = float(request.form["num_bedrooms"])
        population = float(request.form["population"])
        prediction = predict_price(
            income, house_age, num_rooms, num_bedrooms, population)
        return render_template('index.html', prediction_text='Predicted Price: ${:,.2f}'.format(prediction))
    return render_template('index.html')


# Start Gradio interface on a separate thread
iface.launch(share=True)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)

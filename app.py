from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model from the model file
model = pickle.load(open('model.pkl', 'rb'))

# Load the dataset
df = pd.read_csv(r"C:\Users\omer0\OneDrive\Desktop\House_Rent_Dataset.csv")
df.pop("Posted On")
df.pop("Floor")
df.pop("Area Locality")
df.pop("Point of Contact")
df.pop("Area Type")
df.pop("Tenant Preferred")
df['City'] = df['City'].replace(["Mumbai", "Bangalore", "Hyderabad", "Delhi", "Chennai", "Kolkata"], [5, 4, 3, 2, 1, 0])
df['Furnishing Status'] = df['Furnishing Status'].replace(["Furnished", "Semi-Furnished", "Unfurnished"], [2, 1, 0])

# Select the necessary columns for input data
input_columns = ['BHK', 'Size','City', 'Furnishing Status', 'Bathroom']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input values from the form
    features = []
    for column in input_columns:
        value = request.form.get(column)
        if value is None or value == '':
            return render_template('index.html', error_message='Please fill all the fields.')
        features.append(float(value))

    input_data = pd.DataFrame([features], columns=input_columns)

    # Make a prediction using the loaded model
    prediction = model.predict(input_data)[0]

    # Display the prediction result in the template
    return render_template('index.html', prediction_text=f"Predicted Rent: {prediction:.2f}")


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

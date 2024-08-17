from flask import Flask, request, jsonify
import utils

app = Flask(__name__)

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': utils.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    print("Received request:", request.get_data(as_text=True))  # Print the request data
    if request.is_json:
        data = request.json
    else:
        data = request.form

    # Check if required keys are present
    if 'total_sqft' not in data or 'location' not in data or 'bhk' not in data or 'bath' not in data:
        return jsonify({'error': 'Missing required fields'}), 400

    total_sqft = float(data['total_sqft'])
    location = data['location']
    bhk = int(data['bhk'])
    bath = int(data['bath'])

    response = jsonify({
        'estimated_price': utils.get_estimated_price(location, total_sqft, bhk, bath)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    utils.load_saved_artifacts()
    app.run()

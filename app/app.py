# Importing essential libraries and modules
from flask import Flask, redirect, render_template, request, jsonify, url_for
from markupsafe import Markup
import markdown

import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from utils.weed import weed_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
import base64
import json
import os
import google.generativeai as genai

# ==============================================================================================
# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model
disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = r'E:\DiskD\CSP\New folder\Harvestify\plant-disease-model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# Loading crop recommendation model
crop_recommendation_model_path = r'E:\DiskD\CSP\New folder\Harvestify\models\RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

# Configure Google Gemini API
if hasattr(config, 'gemini_api_key'):
    genai.configure(api_key=config.gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# =========================================================================================

# Custom functions for calculations
def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]
        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

def identify_weed(image_data):
    """
    Uses Plant.ID API to identify weeds from images
    :params: image data (bytes)
    :return: weed name, scientific name, confidence, description, management
    """
    if not hasattr(config, 'plant_id_api_key'):
        return "Unknown", "Not available", 0, "API key not configured", "Please configure Plant.ID API key"
    
    # Prepare image data for Plant.ID API
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    # Plant.ID API request
    api_endpoint = "https://api.plant.id/v2/identify"
    headers = {
        "Content-Type": "application/json",
        "Api-Key": config.plant_id_api_key
    }
    data = {
        "images": [image_base64],
        "modifiers": ["weed", "crops"],
        "plant_details": ["common_names", "url", "wiki_description", "taxonomy"]
    }
    
    try:
        response = requests.post(api_endpoint, headers=headers, json=data)
        result = response.json()
        
        # Check if identification was successful
        if 'suggestions' in result and len(result['suggestions']) > 0:
            best_match = result['suggestions'][0]
            confidence = best_match['probability'] * 100
            
            # Get the common name, or use the scientific name if common name is not available
            if 'plant_details' in best_match and 'common_names' in best_match['plant_details'] and len(best_match['plant_details']['common_names']) > 0:
                weed_name = best_match['plant_details']['common_names'][0]
            else:
                weed_name = best_match['plant_name']
            
            scientific_name = best_match['plant_name']
            
            # Check if it's in our weed dictionary
            if weed_name in weed_dic:
                description = weed_dic[weed_name]['description']
                management = weed_dic[weed_name]['management']
            elif any(common.lower() in map(str.lower, weed_dic.keys()) for common in best_match.get('plant_details', {}).get('common_names', [])):
                # Try to find a match with any common name
                for common in best_match.get('plant_details', {}).get('common_names', []):
                    for key in weed_dic.keys():
                        if common.lower() == key.lower():
                            weed_name = key
                            description = weed_dic[key]['description']
                            management = weed_dic[key]['management']
                            break
            else:
                # Use wikipedia description if available
                if 'plant_details' in best_match and 'wiki_description' in best_match['plant_details'] and 'value' in best_match['plant_details']['wiki_description']:
                    description = best_match['plant_details']['wiki_description']['value']
                else:
                    description = "No detailed information available for this weed."
                
                # Use Gemini to generate management advice if available
                if hasattr(config, 'gemini_api_key'):
                    management = generate_management_advice(weed_name, scientific_name)
                else:
                    management = "No specific management information available for this weed."
            
            return weed_name, scientific_name, confidence, description, management
        else:
            return "Unknown", "Not available", 0, weed_dic["Unknown"]["description"], weed_dic["Unknown"]["management"]
    
    except Exception as e:
        print(f"Error in weed identification: {e}")
        return "Unknown", "Not available", 0, "Error in weed identification process", "Please try again with a clearer image"

def generate_management_advice(weed_name, scientific_name):
    """
    Uses Google's Gemini model to generate management advice for weeds
    :params: weed_name, scientific_name
    :return: management advice text
    """
    if not hasattr(config, 'gemini_api_key'):
        return "Gemini API key not configured. Cannot generate management advice."
    
    try:
        prompt = f"""
        Provide practical management advice for controlling the weed {weed_name} (scientific name: {scientific_name}) in agricultural settings.
        Include information on:
        1. Prevention methods
        2. Cultural, mechanical, and chemical control options where appropriate
        3. Environmentally friendly approaches
        Keep your response concise and focused on practical advice a farmer could implement.
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating management advice: {e}")
        return "Unable to generate management advice at this time. Please consult with a local agricultural extension for specific control methods."

def get_pesticide_recommendations(weed_name):
    """
    Uses Google's Gemini model to generate pesticide recommendations for a specific weed
    :params: weed_name
    :return: tuple of (pesticide name, recommended dosage)
    """
    if not hasattr(config, 'gemini_api_key'):
        return None, None
    
    try:
        prompt = f"""
        For controlling the weed {weed_name} in agricultural settings, provide:
        1. The most effective and commonly used pesticide (active ingredient name)
        2. The recommended dosage in milliliters per acre (ml/acre)
        
        Format your response exactly like this example:
        Pesticide: glyphosate
        Dosage: 500
        
        Choose a commonly available and effective pesticide that would be available in India.
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Parse the response
        pesticide = None
        dosage = None
        
        for line in response_text.split('\n'):
            if line.startswith('Pesticide:'):
                pesticide = line.replace('Pesticide:', '').strip()
            elif line.startswith('Dosage:'):
                try:
                    dosage = float(line.replace('Dosage:', '').strip())
                except ValueError:
                    dosage = 200  # Default value if parsing fails
        
        print(f"Recommended pesticide for {weed_name}: {pesticide} at {dosage} ml/acre")
        return pesticide, dosage
    except Exception as e:
        print(f"Error generating pesticide recommendation: {e}")
        return None, None

def fetch_pesticide_info(pesticide_name):
    """
    Fetches pesticide information from the agribegri API
    :params: pesticide_name
    :return: list of dictionaries with pesticide information for the 3 cheapest products
    """
    if not pesticide_name:
        return None
        
    try:
        # Clean up the pesticide name - remove any punctuation and extra spaces
        cleaned_name = pesticide_name.lower().strip()
        print(f"Searching for pesticide: {cleaned_name}")
        
        url = "https://agribegri.com/typehead_search.php"
        payload = {'query': cleaned_name}
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Origin': 'https://agribegri.com',
            'Referer': 'https://agribegri.com/'
        }
        
        print(f"Sending request to {url} with payload: {payload}")
        response = requests.post(url, data=payload, headers=headers)
        
        print(f"API Response status: {response.status_code}")
        products = []
        
        if response.status_code == 200:
            result = response.json()
            print(f"API Response: {result}")
            
            if result and len(result) > 0:
                # Process results and sort by price
                valid_results = []
                for product in result:
                    # Check if product has a price
                    if 'abpd_price' in product and product['abpd_price']:
                        try:
                            # Convert price to float for sorting
                            product['price_float'] = float(product['abpd_price'])
                            valid_results.append(product)
                        except (ValueError, TypeError):
                            # Skip products with invalid prices
                            continue
                
                # Sort by price (lowest first) and take up to 3
                if valid_results:
                    products = sorted(valid_results, key=lambda x: x['price_float'])[:3]
                    return products
            
            # If no results with original name, try some common alternatives
            if not products:
                common_pesticides = {
                    "glyphosate": ["roundup", "glycel"],
                    "2,4-d": ["2 4 d", "24d", "weedmar"],
                    "atrazine": ["atrataf"],
                    "pendimethalin": ["stomp", "pendistar"],
                    "paraquat": ["gramoxone"],
                    "metsulfuron": ["ally"],
                    "metribuzin": ["sencor"],
                    "dicamba": ["banvel"]
                }
                
                # Check if our pesticide is in the common list
                for key, alternatives in common_pesticides.items():
                    if cleaned_name in key or any(alt in cleaned_name for alt in alternatives):
                        all_alt_results = []
                        for alt_name in [key] + alternatives:
                            print(f"Trying alternative name: {alt_name}")
                            alt_payload = {'query': alt_name}
                            alt_response = requests.post(url, data=alt_payload, headers=headers)
                            if alt_response.status_code == 200:
                                alt_result = alt_response.json()
                                if alt_result and len(alt_result) > 0:
                                    # Add valid results to our list
                                    for product in alt_result:
                                        if 'abpd_price' in product and product['abpd_price']:
                                            try:
                                                product['price_float'] = float(product['abpd_price'])
                                                all_alt_results.append(product)
                                            except (ValueError, TypeError):
                                                continue
                        
                        # Sort by price and take top 3
                        if all_alt_results:
                            products = sorted(all_alt_results, key=lambda x: x['price_float'])[:3]
                            return products
        
        # If we get here, no results were found
        print("No pesticide products found")
        return None
    except Exception as e:
        print(f"Error fetching pesticide info: {e}")
        return None

def process_chat_message(message):
    """
    Process user chat messages using Google's Gemini model
    :params: user message
    :return: bot response
    """
    if not hasattr(config, 'gemini_api_key'):
        return "Chatbot is not available as the Gemini API key is not configured."
    
    try:
        prompt = f"""
        You are an agricultural expert specializing in weed management and crop protection. The user is asking:
        
        "{message}"
        
        Provide a helpful, accurate, and concise response. Focus on practical advice for farmers and gardeners.
        If the question is not about agriculture, weeds, or crop management, politely steer the conversation back to these topics.
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error in chatbot: {e}")
        return "I'm sorry, I encountered an issue processing your question. Please try again or ask a different question about weed management or crop protection."

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------
app = Flask(__name__)

# Add markdown filter to Jinja2
@app.template_filter('markdown')
def markdown_filter(text):
    return Markup(markdown.markdown(text, extensions=['extra', 'nl2br']))

# render home page
@app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page
@app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page
@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)

# render disease prediction form page
@app.route('/disease')
def disease_prediction():
    title = 'Harvestify - Disease Detection'
    return render_template('disease.html', title=title)

# render weed detection form page
@app.route('/weed')
def weed_detection():
    title = 'Harvestify - Weed Detection'
    return render_template('weed.html', title=title)

# ===============================================================================================
# RENDER PREDICTION PAGES

# render crop recommendation result page
@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)
        else:
            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page
@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    df = pd.read_csv(r'E:\DiskD\CSP\New folder\Harvestify\Data-processed\fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction_result():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()
            prediction = predict_image(img)
            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except Exception as e:
            print(f"Error processing image: {e}")
            return render_template('disease.html', title=title)
    return render_template('disease.html', title=title)

# render weed detection result page
@app.route('/weed-predict', methods=['GET', 'POST'])
def weed_prediction():
    title = 'Harvestify - Weed Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('weed.html', title=title)
        
        try:
            # Read and process the image
            img_data = file.read()
            
            # Create the static/images directory if it doesn't exist
            static_dir = os.path.join('static', 'images')
            os.makedirs(static_dir, exist_ok=True)
            
            # Save a copy of the image for display
            img_path = os.path.join(static_dir, 'uploaded_weed.jpg')
            with open(img_path, 'wb') as f:
                f.write(img_data)
            
            # Identify the weed using Plant.ID API
            weed_name, scientific_name, confidence, description, management = identify_weed(img_data)
            
            # Get pesticide recommendation
            pesticide_name, recommended_dosage = get_pesticide_recommendations(weed_name)
            pesticide_info = None
            if pesticide_name:
                pesticide_info = fetch_pesticide_info(pesticide_name)
                print(f"Final pesticide info: {pesticide_info}")
            
            return render_template('weed-result.html', 
                                 title=title,
                                 weed_name=weed_name,
                                 scientific_name=scientific_name,
                                 confidence=confidence,
                                 description=description,
                                 management=management,
                                 weed_image=img_path,
                                 pesticide_name=pesticide_name,
                                 pesticide_info=pesticide_info,
                                 recommended_dosage=recommended_dosage)
        except Exception as e:
            print(f"Error processing weed image: {e}")
            return render_template('weed.html', title=title)
    
    return render_template('weed.html', title=title)

# handle chat messages through API
@app.route('/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({'error': 'Invalid request format'}), 400
    
    data = request.get_json()
    if 'message' not in data or not data['message']:
        return jsonify({'error': 'No message provided'}), 400
    
    user_message = data['message']
    
    # Process the message using Gemini
    response = process_chat_message(user_message)
    
    return jsonify({'response': response})

# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)


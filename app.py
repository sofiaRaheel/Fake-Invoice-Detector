# import os
# import cv2
# import numpy as np
# import hashlib
# import pytesseract
# from pytesseract import Output
# from flask import Flask, render_template, request, redirect, url_for, flash
# from werkzeug.utils import secure_filename

# # Configure paths
# UPLOAD_FOLDER = 'static/uploads/'
# TRAIN_IMAGES = "static/inv_img/train/"
# TEST_IMAGES = "static/inv_img/test/"
# TEMPLATES = "static/templates/"
# OUTPUT_DIR = "static/output/"

# # Allowed file extensions
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

# # Create directories if they don't exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(TEMPLATES, exist_ok=True)
# os.makedirs(TRAIN_IMAGES, exist_ok=True)
# os.makedirs(TEST_IMAGES, exist_ok=True)

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['SECRET_KEY'] = 'supersecretkey'

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def preprocess_image(image_path):
#     """Preprocess the image for better OCR and template matching"""
#     img = cv2.imread(image_path)
#     if img is None:
#         return None
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Apply adaptive thresholding
#     processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                     cv2.THRESH_BINARY, 11, 2)
    
#     # Remove noise
#     kernel = np.ones((1, 1), np.uint8)
#     processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    
#     return processed

# def generate_image_hash(image_path):
#     """Generate SHA-256 hash of an image file"""
#     with open(image_path, "rb") as f:
#         bytes = f.read()
#         return hashlib.sha256(bytes).hexdigest()

# def create_template(image_path, output_name):
#     """Create and save a template from a training image"""
#     img = cv2.imread(image_path)
#     if img is None:
#         return False
    
#     template_path = os.path.join(TEMPLATES, f"template_{output_name}.jpg")
#     cv2.imwrite(template_path, img)
#     print(f"Template created: {template_path}")
#     return True

# def match_template(uploaded_img, template_img):
#     """Compare uploaded image with template using template matching"""
#     img_gray = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2GRAY)
#     template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    
#     # Perform template matching
#     res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
#     # Draw rectangle around matched area
#     h, w = template_gray.shape
#     top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     cv2.rectangle(uploaded_img, top_left, bottom_right, (0, 255, 0), 2)
    
#     return max_val, uploaded_img

# def extract_text(image):
#     """Extract text from image using Tesseract OCR"""
#     custom_config = r'--oem 3 --psm 6'
#     d = pytesseract.image_to_data(image, output_type=Output.DICT, config=custom_config)
    
#     extracted_text = {}
#     n_boxes = len(d['level'])
#     for i in range(n_boxes):
#         text = d['text'][i]
#         if text.strip():
#             (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#             extracted_text[text] = (x, y, w, h)
    
#     return extracted_text

# def highlight_text_fields(image, extracted_text):
#     """Draw rectangles around detected text fields"""
#     highlighted = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     for text, (x, y, w, h) in extracted_text.items():
#         cv2.rectangle(highlighted, (x, y), (x + w, y + h), (0, 255, 0), 1)
#         cv2.putText(highlighted, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
#                    0.3, (0, 0, 255), 1)
#     return highlighted

# def verify_invoice(image_path):
#     """Main function to verify an invoice"""
#     # Preprocess the image
#     processed_img = preprocess_image(image_path)
#     if processed_img is None:
#         return {"error": "Invalid image file"}, None
    
#     # Save processed image
#     processed_path = os.path.join(OUTPUT_DIR, "processed.jpg")
#     cv2.imwrite(processed_path, processed_img)
    
#     # Generate hash and compare with training images
#     uploaded_hash = generate_image_hash(image_path)
#     train_hashes = {img: generate_image_hash(os.path.join(TRAIN_IMAGES, img)) 
#                    for img in os.listdir(TRAIN_IMAGES) if img.endswith(('.png', '.jpg', '.jpeg'))}
    
#     # Check for exact hash match
#     if uploaded_hash in train_hashes.values():
#         matched_img = [img for img, h in train_hashes.items() if h == uploaded_hash][0]
#         return {"result": "Original", "reason": f"Exact hash match with {matched_img}"}
    
#     # If no hash match, proceed with template matching and OCR
#     uploaded_img = cv2.imread(image_path)
#     best_match_score = 0
#     best_template = None
#     best_matched_img = None
    
#     # Compare with all templates
#     for template_file in os.listdir(TEMPLATES):
#         template_path = os.path.join(TEMPLATES, template_file)
#         template_img = cv2.imread(template_path)
        
#         if template_img is None:
#             continue
            
#         score, matched_img = match_template(uploaded_img.copy(), template_img)
#         if score > best_match_score:
#             best_match_score = score
#             best_template = template_file
#             best_matched_img = matched_img
    
#     # Save template matched image
#     if best_matched_img is not None:
#         matched_path = os.path.join(OUTPUT_DIR, "template_matched.jpg")
#         cv2.imwrite(matched_path, best_matched_img)
    
#     # Extract text from uploaded image
#     extracted_text = extract_text(processed_img)
#     highlighted_img = highlight_text_fields(processed_img, extracted_text)
#     highlighted_path = os.path.join(OUTPUT_DIR, "highlighted.jpg")
#     cv2.imwrite(highlighted_path, highlighted_img)
    
#     # Threshold for template matching
#     if best_match_score > 0.8:
#         # Compare with template text
#         template_path = os.path.join(TEMPLATES, best_template)
#         template_processed = preprocess_image(template_path)
#         template_text = extract_text(template_processed)
        
#         # Calculate text field similarity
#         common_fields = set(extracted_text.keys()) & set(template_text.keys())
#         match_percentage = len(common_fields) / max(len(extracted_text), len(template_text)) * 100
        
#         if match_percentage > 70:
#             return {
#                 "result": "Original",
#                 "reason": f"Template match ({best_match_score:.2f}) with {best_template} and {match_percentage:.1f}% text match",
#                 "extracted_text": extracted_text,
#                 "images": {
#                     "original": image_path.replace('static/', ''),
#                     "processed": processed_path.replace('static/', ''),
#                     "highlighted": highlighted_path.replace('static/', ''),
#                     "template_matched": matched_path.replace('static/', '') if best_matched_img is not None else None
#                 }
#             }
#         else:
#             return {
#                 "result": "Fake",
#                 "reason": f"Template matched but text fields don't align ({match_percentage:.1f}% match)",
#                 "extracted_text": extracted_text,
#                 "images": {
#                     "original": image_path.replace('static/', ''),
#                     "processed": processed_path.replace('static/', ''),
#                     "highlighted": highlighted_path.replace('static/', ''),
#                     "template_matched": matched_path.replace('static/', '') if best_matched_img is not None else None
#                 }
#             }
#     else:
#         return {
#             "result": "Fake",
#             "reason": f"No matching template found (best score: {best_match_score:.2f})",
#             "extracted_text": extracted_text,
#             "images": {
#                 "original": image_path.replace('static/', ''),
#                 "processed": processed_path.replace('static/', ''),
#                 "highlighted": highlighted_path.replace('static/', '')
#             }
#         }

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         # Check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
        
#         file = request.files['file']
        
#         # If user does not select file, browser also submits an empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
        
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)
            
#             # Verify the invoice
#             result = verify_invoice(filepath)
            
#             return render_template('result.html', result=result)
    
#     return render_template('index.html')

# if __name__ == '__main__':
#     # First create templates from training images if they exist
#     if os.path.exists(TRAIN_IMAGES) and os.listdir(TRAIN_IMAGES):
#         print("Creating templates from training images...")
#         for i, img in enumerate(os.listdir(TRAIN_IMAGES)):
#             if img.endswith(('.png', '.jpg', '.jpeg')):
#                 create_template(os.path.join(TRAIN_IMAGES, img), f"train_{i+1}")
    
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import hashlib
import pytesseract
from pytesseract import Output
import os

app = Flask(__name__)

# Configure paths
TRAIN_IMAGES = "inv_img/train/"
TEMPLATES = "templates/"

# Ensure directories exist
os.makedirs(TEMPLATES, exist_ok=True)

# Initialize templates from training images
def init_templates():
    for i, img in enumerate(os.listdir(TRAIN_IMAGES)):
        img_path = os.path.join(TRAIN_IMAGES, img)
        template = cv2.imread(img_path)
        if template is not None:
            cv2.imwrite(f"{TEMPLATES}/template_{i}.jpg", template)

init_templates()  # Create templates on startup

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    return processed

def generate_image_hash(img):
    return hashlib.sha256(cv2.imencode('.jpg', img)[1]).hexdigest()

def match_template(uploaded_img, template_img):
    img_gray = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val

def extract_text(img):
    custom_config = r'--oem 3 --psm 6'
    d = pytesseract.image_to_data(img, output_type=Output.DICT, config=custom_config)
    extracted = {}
    for i, text in enumerate(d['text']):
        if text.strip():
            extracted[text] = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    return extracted

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/verify", methods=["POST"])
def verify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Read image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # Generate hash and check for exact match
    img_hash = generate_image_hash(img)
    train_hashes = {}
    for train_img in os.listdir(TRAIN_IMAGES):
        train_img_path = os.path.join(TRAIN_IMAGES, train_img)
        train_img_data = cv2.imread(train_img_path)
        if train_img_data is not None:
            train_hashes[train_img] = generate_image_hash(train_img_data)

    if img_hash in train_hashes.values():
        matched_img = [k for k, v in train_hashes.items() if v == img_hash][0]
        return jsonify({
            "result": "Original",
            "reason": f"Exact hash match with {matched_img}",
            "extracted_text": {}
        })

    # If no hash match, check templates
    best_score = 0
    best_template = None
    for template_file in os.listdir(TEMPLATES):
        template = cv2.imread(os.path.join(TEMPLATES, template_file))
        if template is not None:
            score = match_template(img, template)
            if score > best_score:
                best_score = score
                best_template = template_file

    # Extract text
    processed_img = preprocess_image(img)
    extracted_text = extract_text(processed_img)

    # Determine result
    if best_score > 0.8:
        template_img = cv2.imread(os.path.join(TEMPLATES, best_template))
        template_processed = preprocess_image(template_img)
        template_text = extract_text(template_processed)
        common_fields = set(extracted_text.keys()) & set(template_text.keys())
        match_percentage = len(common_fields) / max(len(extracted_text), len(template_text)) * 100

        if match_percentage > 70:
            return jsonify({
                "result": "Original",
                "reason": f"Template match ({best_score:.2f}) with {best_template} & {match_percentage:.1f}% text match",
                "extracted_text": extracted_text
            })
        else:
            return jsonify({
                "result": "Fake",
                "reason": f"Template matched but text mismatch ({match_percentage:.1f}%)",
                "extracted_text": extracted_text
            })
    else:
        return jsonify({
            "result": "Fake",
            "reason": f"No template match (best score: {best_score:.2f})",
            "extracted_text": extracted_text
        })

if __name__ == "__main__":
    app.run(debug=True)
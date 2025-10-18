##########------with cheque number ----------#########
import cv2
import torch
import os
import re
import numpy as np
import mysql.connector as mq
from flask import Flask, render_template, request, redirect, url_for, session,flash,jsonify
from PIL import Image
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from markupsafe import Markup
import mailing
import difflib

app = Flask(__name__)
app.secret_key = "your_secret_key"

# MySQL Connection
def dbconnection():
    con = mq.connect(host='localhost', database='chequeverification',user='root',password='root')
    return con


# Load YOLO and TrOCR Models
model = YOLO("yolomodel.pt")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

UPLOAD_FOLDER = "static/uploads/cheques"
SIGNATURE_FOLDER = "static/uploads/signatures"
CROPPED_FOLDER = "cropped_regions"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/logout')
def logout():
    # Clear the session
    session.clear()
    flash('You have been logged out.', 'success')
    # Redirect to the login page or homepage
    return redirect(url_for('login'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/addaccountpage')
def addaccountpage():
    return render_template('addaccount.html')

@app.route('/uploadcheque')
def uploadcheque():
    return render_template('uploadcheque.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    con = dbconnection()
    db = con.cursor()
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        db.execute("SELECT * FROM authority WHERE email = '{}' AND password ='{}'".format(email, password))
        user =db.fetchall()
        if user:
            return redirect(url_for('addaccountpage'))
        else:
            message = Markup("<h3> Invalid credentials </h3>")
            flash(message)
            return render_template('login.html')
    return render_template('login.html')

@app.route('/addaccount', methods=['GET', 'POST'])
def addaccount():
    con = dbconnection()
    db = con.cursor()
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        accno = request.form['accno']
        amount = request.form['amount']
        uploaded_file = request.files['simage']
        db.execute("select * from accounts where email='{}' or phone='{}' or accno='{}'".format(email,phone,accno))
        user=db.fetchall()
        if user:
            message = Markup("<h3> Customer email or phone or account no already exist.</h3>")
            flash(message)
            return redirect(url_for('addaccountpage'))
        else:
            uploaded_file = request.files['simage']
        if uploaded_file.filename != '':
            # Save the uploaded file to a specific folder
            ext = os.path.splitext(uploaded_file.filename)[1]  # Get file extension
            new_filename = f"{accno}{ext}"  # Rename with accno
            save_path = os.path.join('static/uploads/signatures', new_filename)
            
            uploaded_file.save(save_path)  # Save the file
            db.execute("INSERT INTO accounts (name, email, phone, address, accno, amount, signature) VALUES ('{}','{}','{}','{}','{}',{},'{}')".format(
                name, email, phone, address, accno, amount, new_filename))

            con.commit()
            con.close()
            message = Markup("<h3> Customer Added.</h3>")
            flash(message)
            return redirect(url_for('addaccountpage'))
    return redirect(url_for('addaccountpage'))


@app.route('/viewaccountspage')
def viewaccountspage():
    con = dbconnection()
    db = con.cursor()
    db.execute("SELECT * FROM accounts")
    acc =db.fetchall()
    return render_template('viewaccounts.html',res=acc)

@app.route('/deleteacc')
def deleteacc():
    id = request.args.get("id")
    con = dbconnection()
    db = con.cursor()
    db.execute("delete FROM accounts where id={}".format(int(id)))
    con.commit()
    con.close()
    return redirect(url_for('viewaccountspage'))

@app.route('/searchaccno', methods=['POST'])
def searchaccno():
    con = dbconnection()
    cursor = con.cursor()
    search_term = request.json.get('searchTerm')
    sql = "SELECT name,email,phone,address,accno,amount,signature FROM accounts where accno LIKE %s"
    cursor.execute(sql, ('%' + search_term + '%',))
    res = cursor.fetchall()
    print(res)
    return jsonify(res)

def show_signature_comparison(stored_signature, cropped_signature):
    """
    Compares the stored and extracted signature images using SSIM.
    """

    # Convert to grayscale
    stored_gray = cv2.cvtColor(stored_signature, cv2.COLOR_BGR2GRAY)
    cropped_gray = cv2.cvtColor(cropped_signature, cv2.COLOR_BGR2GRAY)

    # Ensure both images have the same size
    common_width = min(stored_gray.shape[1], cropped_gray.shape[1])  # Choose the smaller width
    common_height = min(stored_gray.shape[0], cropped_gray.shape[0])  # Choose the smaller height

    stored_gray = cv2.resize(stored_gray, (common_width, common_height))
    cropped_gray = cv2.resize(cropped_gray, (common_width, common_height))

    # Compute Structural Similarity Index (SSIM)
    similarity, diff = ssim(stored_gray, cropped_gray, full=True)

    # Normalize the difference image to 0-255 scale
    diff = (diff * 255).astype(np.uint8)

    # Apply colormap to visualize differences
    diff_colored = cv2.applyColorMap(255 - diff, cv2.COLORMAP_JET)

    # Convert original images to match the new dimensions
    stored_signature_resized = cv2.resize(stored_signature, (common_width, common_height))
    cropped_signature_resized = cv2.resize(cropped_signature, (common_width, common_height))

    # Stack images horizontally for visualization
    comparison_image = np.hstack([stored_signature_resized, cropped_signature_resized, diff_colored])

    # Display similarity score
    cv2.putText(comparison_image, f"Similarity: {similarity:.2f}", (50, common_height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the comparison
    cv2.imshow("Signature Comparison", comparison_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return similarity
# Function to preprocess image for OCR
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.dilate(thresh, kernel, iterations=1)
    return processed

# Function to extract text using TrOCR
def extract_text_trocr(image):
    pil_image = Image.fromarray(image).convert("RGB")
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    generated_ids = ocr_model.generate(pixel_values)
    extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return extracted_text.strip()

# Function to clean extracted text
def clean_text(label, text):
    text = text.strip()
    if label == "accno":
        text = re.sub(r"\D", "", text)  # Only digits
    elif label == "numbers":
        text = re.search(r"\d+", text)
        text = text.group() if text else ""
    elif label == "words":
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabet characters
    elif label == "checkno":
        text = re.sub(r"\D", "", text)  # Only digits
        
    return text

# Function to compare signatures using SSIM
def compare_signatures(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_resized = cv2.resize(img1_gray, (img2_gray.shape[1], img2_gray.shape[0]))
    similarity = ssim(img1_resized, img2_gray)
    return similarity

# Flask route to upload and process cheque
@app.route("/uploadcheque", methods=["POST"])
def upload_cheque():
    con = dbconnection()
    cursor = con.cursor()
    if "cimage" not in request.files:
        flash("No file uploaded")
        return redirect("/uploadcheque")

    file = request.files["cimage"]
    if file.filename == "":
        flash("No file selected")
        return redirect("/uploadcheque")

    # Save uploaded cheque image
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    print("cheque saved")

    # Load image for processing
    image = cv2.imread(file_path)
    print("image loaded")
    results = model(image)
    print("got results from model")
    extracted_data = {}

    # Process detections
    for result in results:
        for i, (box, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]

            cropped_img = image[y1:y2, x1:x2]
            cropped_path = os.path.join(CROPPED_FOLDER, f"{label}.jpg")
            
            cv2.imwrite(cropped_path, cropped_img)
            print("saving",label)
            
            processed_image = preprocess_image(cropped_img)
            print("preprocessed image")
            text = extract_text_trocr(processed_image)
            print("text extracted : ",text)
            cleaned_text = clean_text(label, text)
            print("cleaned text : ",cleaned_text)

            extracted_data[label] = cleaned_text

    # Extracted account number & amount
    extracted_accno = extracted_data.get("accno", "")
    extracted_amount = int(extracted_data.get("numbers", "0"))
    cropped_signature_path = os.path.join(CROPPED_FOLDER, "signature.jpg")
    extracted_checkno = extracted_data.get("checkno", "0")
    print("extracted_accno",extracted_accno)
    print("extracted_amount",extracted_amount)
    print("cropped_signature_path",cropped_signature_path)
    print("extracted_Cheque No",extracted_checkno)
    
    '''this is closest matching in case of failure of proper extraction'''
    cursor.execute("SELECT accno FROM accounts")  # Modify table name if needed
    accounts = [row[0] for row in cursor.fetchall()]
    print("accounts",accounts)
    matches = difflib.get_close_matches(extracted_accno, accounts, n=1, cutoff=0.7)
    print("matched accounts :",matches)
    account=""
    if matches:
        
        # Fetch account from database
        #cursor.execute("SELECT * FROM accounts WHERE accno = %s", (extracted_accno,))
        cursor.execute("SELECT * FROM accounts WHERE accno = %s", (matches[0],))
        account = cursor.fetchone()
        print("fetching account from db",account)

    if not account or account=="":
        print("account not found")
        flash("Account not found")
        return redirect(url_for('uploadcheque'))

    stored_signature_path = os.path.join(SIGNATURE_FOLDER, account[7])  # Assuming signature is the 7th column in your database
    print("stored_signature_path",stored_signature_path)
    # Load stored and cropped signatures
    stored_signature = cv2.imread(stored_signature_path)
    cropped_signature = cv2.imread(cropped_signature_path)

    if stored_signature is None or cropped_signature is None:
        flash("Error loading signatures")
        print("Error loading signatures")
        return redirect(url_for('uploadcheque'))

    # Compare signatures
    similarity_score = compare_signatures(stored_signature, cropped_signature)
    print("similarity_score",similarity_score)
    show_signature_comparison(stored_signature, cropped_signature)

    if similarity_score >= 0.3:  # Threshold for signature match
        new_balance = int(account[6]) - extracted_amount
        cursor.execute("UPDATE accounts SET amount = %s WHERE accno = %s", (new_balance, extracted_accno))
        con.commit()
        flash(f"Transaction successful! New Balance: {new_balance}")
        helpline="111000222"
        bankname="MY INDIAN BANK"
        subject="Urgent: Your cheque is About to be Processed"
        body = (
    "Dear " + str(account[1]) +
    "\n\nA cheque cheque Number: " + str(extracted_checkno) +
    " with the amount of: ₹" + str(extracted_amount) +
    ", is about to be debited from your account " + str(account[5]) +
    ". If you recognize this transaction, no action is required.\n" +
    "However, if you did NOT authorize this cheque, please call our fraud prevention team immediately at " + helpline +
    "\nFor any concerns, please reach out to us.\n\nBest regards,\n" + bankname
)

        mailing.mailsend(account[2],subject,body)
        print("mail sent")
        return redirect(url_for('uploadcheque'))
        
    else:
        helpline="111000222"
        bankname="MY INDIAN BANK"
        subject="Urgent: We found fraud signature for your check"
        body="Dear"+str(account[1])+"\n\nA cheque number "+str(extracted_checkno)+" has been detected with fraud signature.\nHowever, if you did NOT authorize this cheque, please call our fraud prevention team immediately at"+helpline+"\nFor any concerns, please reach out to us.\n\nBest regards\n"+bankname
        mailing.mailsend(account[2],subject,body)
        print("fraud detected, mail sent")
        flash("Signature mismatch!.")
        return redirect(url_for('uploadcheque'))


# Flask route to show the upload page
@app.route("/uploadcheque", methods=["GET"])
def upload_page():
    return render_template("upload.html")


@app.route('/deposit')
def deposit():
    id = request.args.get("id")
    con = dbconnection()
    db = con.cursor()
    db.execute("select * FROM accounts where id={}".format(int(id)))
    res = db.fetchall()
    return render_template("deposit.html",res=res)

@app.route('/depositamount', methods=['GET', 'POST'])
def depositamount():
    con = dbconnection()
    db = con.cursor()
    if request.method == 'POST':
        id = request.form['id']
        email = request.form['email']
        amount = request.form['amount']
        dep = request.form['dep']
        finalamount = int(amount)+int(dep)
        db.execute("update accounts set amount='{}' where id={}".format(finalamount,int(id)))

        con.commit()
        con.close()
        message = Markup("<h3> Amount Deposited</h3>")
        flash(message)
        return redirect(url_for('viewaccountspage'))

if __name__ == "__main__":
    app.run(debug=True)

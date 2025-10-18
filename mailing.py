import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Email configuration
sender_email = "siddiq76543@gmail.com"
password = "txkijyibpfosttmq"
def mailsend(receiver_email,subject,body):
    # Create message object instance
    message = MIMEMultipart()

    # Email subject and body
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Add body to email
    body = body
    message.attach(MIMEText(body, "plain"))

    # Connect to the SMTP server
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()  # Secure the connection
        server.login(sender_email, password)
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        print("Email sent successfully!")

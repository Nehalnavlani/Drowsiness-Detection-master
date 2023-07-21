import os
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time


def sendMail(sender_address,sender_pass,receiver_address,ImgFileName,mail_subject,mail_content,):
    current_time = time.strftime("%H:%M:%S", time.localtime())

    with open(ImgFileName, 'rb') as f:
        img_data = f.read()

    
    #Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = mail_subject + current_time
    #The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))

    image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
    message.attach(image)

    #Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
    session.starttls() #enable security
    session.login(sender_address, sender_pass) #login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print('Mail Sent......')

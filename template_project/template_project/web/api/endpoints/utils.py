from typing import Dict
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email import encoders


class BackgroundTaskManager:
    def __init__(self):
        self.tasks: Dict[str, str] = {}


    def add_task(self, task_id: str):
        if task_id in self.tasks and self.tasks[task_id] == "Running":
            raise Exception(f"task_id {task_id} is already running, choose another")
        self.tasks[task_id] = "Running"

    def mark_task_completed(self, task_id: str):
        self.tasks[task_id] = "Completed"

    def get_task_status(self, task_id: str):
        return self.tasks.get(task_id, "Not Found")


class EmailSender:
    def __init__(self,
                 sender_email: str,
                 sender_password: str,
                 smtp_server: str = 'smtp.gmail.com',
                 smtp_port: int = 587
                 ):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    #TODO SEND MULTIPLE IMAGES
    def send_img_via_email(self,
                           img_path: str,
                           recipient_email: str):
        # Create the MIME object
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = recipient_email
        msg['Subject'] = 'Generated image'

        # Attaching the image
        with open(img_path, 'rb') as image_file:
            img = MIMEImage(image_file.read(), name='image.jpg')
            msg.attach(img)

        # Connecting to the SMTP server
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            # Sending the email
            server.sendmail(self.sender_email, recipient_email, msg.as_string())
        return

    def send_mult_imgs_via_email(self,
                                 img_path_list: list,
                                 recipient_email: str):
        # Create the MIME object
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = recipient_email
        msg['Subject'] = 'Generated image'

        for i, img_path in enumerate(img_path_list):
            # Attaching the image
            with open(img_path, 'rb') as image_file:
                img = MIMEImage(image_file.read(), name=f'image_{i}.jpg')
                msg.attach(img)

        # Connecting to the SMTP server
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            # Sending the email
            server.sendmail(self.sender_email, recipient_email, msg.as_string())
        return

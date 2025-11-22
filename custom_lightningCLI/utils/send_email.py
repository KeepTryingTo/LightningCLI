"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/20-15:35
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

# utils/notifications.py
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import datetime


def send_email(address: str, message: str, subject: str = "Training Notification"):
    """
    发送邮件通知的简化实现
    在实际使用中需要配置SMTP服务器
    """
    print(f"[Email Notification] To: {address}")
    print(f"[Email Notification] Subject: {subject}")
    print(f"[Email Notification] Message: {message}")
    print(f"[Email Notification] Time: {datetime.datetime.now()}")
    print("-" * 50)

    # 需要配置SMTP,关于怎么配置这个SMTP，大家可以到网上搜索，上面有很多教程
    msg = MIMEText(message, 'plain', 'utf-8')
    msg['From'] = 'your_email@example.com'
    msg['To'] = address
    msg['Subject'] = Header(subject, 'utf-8')

    server = smtplib.SMTP('smtp.example.com', 587)
    server.starttls()
    server.login('your_email@example.com', 'your_password')
    server.send_message(msg)
    server.quit()

def send_training_start_notification(email, model_name, dataset_name):
    message = f"""
    Training Started!
    - Model: {model_name}
    - Dataset: {dataset_name}
    - Start Time: {datetime.datetime.now()}
    """
    send_email(email, message, "Training Started")


def send_training_end_notification(email, model_name, training_time, final_metrics=None):
    metrics_str = ""
    if final_metrics:
        metrics_str = "\nFinal Metrics:\n" + "\n".join([f"- {k}: {v:.4f}" for k, v in final_metrics.items()])

    message = f"""
    Training Completed!
    - Model: {model_name}
    - Training Time: {training_time}
    - End Time: {datetime.datetime.now()}
    {metrics_str}
    """
    send_email(email, message, "Training Completed")
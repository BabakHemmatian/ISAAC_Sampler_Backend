#!/usr/bin/env python3
"""Send an alert email using the SMTP credentials already in /home/ubuntu/.env.
Usage: keepalive_notify.py "<subject>" "<body>"
"""
import sys, smtplib, ssl
from email.message import EmailMessage

ENV = "/home/ubuntu/.env"
cfg = {}
with open(ENV) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip().strip('"').strip("'")

host = cfg.get("SMTP_HOST")
port = int(cfg.get("SMTP_PORT", "587"))
user = cfg.get("SMTP_USER")
pw   = cfg.get("SMTP_PASSWORD")
to   = cfg.get("ISSUE_RECEIVER_EMAIL", user)

subject = sys.argv[1] if len(sys.argv) > 1 else "ISAAC keep-alive alert"
body    = sys.argv[2] if len(sys.argv) > 2 else ""

msg = EmailMessage()
msg["From"] = user
msg["To"] = to
msg["Subject"] = subject
msg.set_content(body)

ctx = ssl.create_default_context()
with smtplib.SMTP(host, port, timeout=30) as s:
    s.starttls(context=ctx)
    s.login(user, pw)
    s.send_message(msg)
print("sent to", to)

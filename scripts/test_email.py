#!/usr/bin/env python3
"""
NeuralTrader Email Heartbeat Test
===============================

Simple test to verify email notification system is working.
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.utils.notifier import EmailNotifier
    
    print("🚀 NeuralTrader Email Heartbeat Test")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize email notifier
    print("📧 Initializing Email Notifier...")
    notifier = EmailNotifier()
    
    # Send test email
    print("📨 Sending heartbeat test email...")
    subject = "Heartbeat Test"
    body = f"""NeuralTrader System Heartbeat Test

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: System is Online
Mode: Email Connection Test

This is a test email to verify the notification system is working properly.

If you receive this email, the email notification system is operational.

🤖 NeuralTrader Automation System
"""
    
    success = notifier.send_alert(subject, body)
    
    if success:
        print("✅ SUCCESS: Email sent successfully!")
        print("📬 Check your inbox for the heartbeat test email")
    else:
        print("❌ FAILED: Email sending failed")
        print("🔍 Check the logs above for SMTP or authentication errors")
    
    print()
    print("=" * 50)
    print("🎯 Email Heartbeat Test Complete")
    
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("🔧 Make sure the virtual environment is activated")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("🔍 Check email configuration and network connection")
    sys.exit(1)

#!/usr/bin/env python3
"""
NeuralTrader Email Self-Test Script
==================================

Standalone email test that loads .env and sends a 'Hello World' email
to lugassy.ai@gmail.com. This allows testing email logic without running
the full trading orchestrator.

Usage:
    python tests/test_email.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env")
except ImportError:
    print("❌ python-dotenv not installed. Install with: pip install python-dotenv")
    sys.exit(1)

from src.utils.notifier import EmailNotifier

def test_hello_world_email():
    """Send a simple 'Hello World' test email"""
    print("🧪 NeuralTrader Email Self-Test")
    print("=" * 50)
    
    # Check environment variables
    notifier_email = os.getenv('NOTIFIER_EMAIL')
    notifier_password = os.getenv('NOTIFIER_PASSWORD')
    
    print(f"📧 NOTIFIER_EMAIL: {'✅ Set' if notifier_email else '❌ Not set'}")
    print(f"🔑 NOTIFIER_PASSWORD: {'✅ Set' if notifier_password else '❌ Not set'}")
    
    if not notifier_email or not notifier_password:
        print("\n❌ Email credentials not configured!")
        print("Please edit .env file with your Gmail credentials.")
        return False
    
    # Initialize email notifier
    try:
        notifier = EmailNotifier()
        print(f"✅ Email Notifier initialized")
        print(f"📤 Sender: {notifier.sender_email}")
        print(f"📨 Recipient: {notifier.recipient_email}")
    except Exception as e:
        print(f"❌ Error initializing Email Notifier: {e}")
        return False
    
    # Send Hello World email
    try:
        print("\n📧 Sending 'Hello World' test email...")
        
        subject = "🧪 NeuralTrader Email Self-Test - Hello World"
        body = f"""Hello World! 🌍

This is a self-test email from NeuralTrader to verify email functionality.

Test Details:
- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- From: {notifier.sender_email}
- To: {notifier.recipient_email}
- Purpose: Email system verification

If you receive this email, the NeuralTrader email system is working correctly! 🎉

Best regards,
NeuralTrader Automation System
"""
        
        success = notifier.send_email(
            to_email=notifier.recipient_email,
            subject=subject,
            body=body
        )
        
        if success:
            print("✅ 'Hello World' email sent successfully!")
            print(f"📨 Check your inbox: {notifier.recipient_email}")
            print("📊 Also check logs/automation.log for detailed SMTP conversation")
            return True
        else:
            print("❌ Failed to send 'Hello World' email")
            print("📊 Check logs/automation.log for detailed error information")
            return False
            
    except Exception as e:
        print(f"❌ Error sending 'Hello World' email: {e}")
        return False

def test_smtp_connection():
    """Test SMTP connection without sending email"""
    print("\n🔗 Testing SMTP Connection...")
    
    try:
        import smtplib
        from src.utils.notifier import logger
        
        notifier_email = os.getenv('NOTIFIER_EMAIL')
        notifier_password = os.getenv('NOTIFIER_PASSWORD')
        
        if not notifier_email or not notifier_password:
            print("❌ Cannot test SMTP connection - credentials not set")
            return False
        
        # Test connection
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.set_debuglevel(1)
            print("🔗 Connecting to smtp.gmail.com:587...")
            
            server.starttls()
            print("🔒 TLS connection established")
            
            server.login(notifier_email, notifier_password)
            print("✅ SMTP login successful!")
            
        print("✅ SMTP connection test passed")
        return True
        
    except Exception as e:
        print(f"❌ SMTP connection test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 NeuralTrader Email Self-Test Suite")
    print("=" * 50)
    
    # Test SMTP connection
    smtp_test = test_smtp_connection()
    
    # Test Hello World email
    email_test = test_hello_world_email()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SELF-TEST RESULTS:")
    print(f"SMTP Connection Test: {'✅ PASS' if smtp_test else '❌ FAIL'}")
    print(f"Hello World Email Test: {'✅ PASS' if email_test else '❌ FAIL'}")
    
    if smtp_test and email_test:
        print("\n🎉 ALL TESTS PASSED! Email system is working correctly.")
        print("📨 You should receive a 'Hello World' email shortly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check logs/automation.log for details.")
        sys.exit(1)

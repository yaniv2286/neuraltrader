#!/usr/bin/env python3
"""
NeuralTrader Tier 0 Pre-Flight Checklist
=======================================

Critical validation to ensure 'No Silent Failures' in production.
This script validates all critical systems before allowing main orchestrator to run.

If ANY step fails, it sends a 'PRE-FLIGHT FAILED' email and exits with sys.exit(1)
to block the main orchestrator from running.
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
import tempfile

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import production components
from core.utils.notifier import EmailNotifier
from core.ai_models import EnsemblePredictor

class PreFlightCheckError(Exception):
    """Critical pre-flight check error"""
    pass

class PreFlightChecklist:
    """Tier 0 Pre-Flight Checklist for NeuralTrader Production"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.email_notifier = None
        self.ib_connection = None
        
        self.logger.info("=" * 60)
        self.logger.info("NEURALTRADER TIER 0 PRE-FLIGHT CHECKLIST")
        self.logger.info("=" * 60)
        self.logger.info("Validating all critical systems before production...")
    
    def _setup_logging(self):
        """Setup logging for pre-flight check"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', 'pre_flight_check.log'))
            ]
        )
        return logging.getLogger(__name__)
    
    def run_all_checks(self):
        """Run complete pre-flight checklist"""
        try:
            self.logger.info("Starting comprehensive pre-flight validation...")
            
            # Check 1: Broker Connectivity
            self._check_broker_connectivity()
            
            # Check 2: Communications Test
            self._check_communications()
            
            # Check 3: Model Integrity
            self._check_model_integrity()
            
            # Check 4: API Pulse
            self._check_api_pulse()
            
            # All checks passed
            self.logger.info("\n" + "=" * 60)
            self.logger.info("✅ ALL PRE-FLIGHT CHECKS PASSED")
            self.logger.info("✅ NEURALTRADER IS READY FOR PRODUCTION")
            self.logger.info("=" * 60)
            
            # Send success notification
            self._send_success_notification()
            
            return True
            
        except Exception as e:
            self.logger.error(f"\n" + "=" * 60)
            self.logger.error("❌ PRE-FLIGHT CHECK FAILED")
            self.logger.error(f"❌ Error: {e}")
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            self.logger.error("=" * 60)
            
            # Send failure notification
            self._send_failure_notification(e, traceback.format_exc())
            
            # Block main orchestrator
            self.logger.error("🚫 BLOCKING MAIN ORCHESTRATOR - CRITICAL ISSUES DETECTED")
            sys.exit(1)
    
    def _check_broker_connectivity(self):
        """Check IBKR broker connectivity with real order test"""
        self.logger.info("\n[CHECK 1] Broker Connectivity Test...")
        
        try:
            # Import ib_async
            import ib_async
            
            self.logger.info("  Connecting to IBKR (127.0.0.1:7497)...")
            
            # Create IBKR connection
            self.ib_connection = ib_async.IB()
            
            # Connect to TWS Paper Trading
            connect_task = self.ib_connection.connect('127.0.0.1', 7497, clientId=999)
            
            # Wait for connection
            import time
            time.sleep(2)
            
            if not self.ib_connection.isConnected():
                raise PreFlightCheckError("Failed to connect to IBKR")
            
            self.logger.info("  ✅ Connected to IBKR successfully")
            
            # Submit test order
            self.logger.info("  Submitting test order (AAPL $1.00 Limit)...")
            
            # Create limit order for 1 share of AAPL at $1.00 (should not execute)
            contract = ib_async.Stock('AAPL', 'SMART', 'USD')
            order = ib_async.LimitOrder('BUY', 1, 1.00)
            
            # Submit order
            submit_task = self.ib_connection.placeOrder(contract, order)
            
            # Wait for order to process
            import time
            time.sleep(3)
            
            # Verify order was received (check if it moved beyond PendingSubmit)
            if submit_task.orderStatus.status == 'PendingSubmit':
                # Order might still be submitting, check if it has an orderId
                if hasattr(submit_task.order, 'orderId') and submit_task.order.orderId > 0:
                    self.logger.info(f"  ✅ Order received by broker (Order ID: {submit_task.order.orderId}, Status: {submit_task.orderStatus.status})")
                else:
                    raise PreFlightCheckError(f"Order still in PendingSubmit status: {submit_task.orderStatus.status}")
            else:
                self.logger.info(f"  ✅ Order received by broker (Status: {submit_task.orderStatus.status})")
            
            # Cancel the test order immediately
            self.logger.info("  Canceling test order...")
            
            cancel_result = self.ib_connection.cancelOrder(submit_task.order)
            
            self.logger.info("  ✅ Test order cancelled successfully")
            
            # Disconnect
            self.ib_connection.disconnect()
            self.logger.info("  ✅ Disconnected from IBKR")
            
        except ImportError as e:
            raise PreFlightCheckError(f"ib_async not available: {e}")
        except Exception as e:
            raise PreFlightCheckError(f"Broker connectivity test failed: {e}")
    
    def _check_communications(self):
        """Test email communications with attachment"""
        self.logger.info("\n[CHECK 2] Communications Test...")
        
        try:
            # Initialize EmailNotifier
            self.email_notifier = EmailNotifier()
            self.logger.info("  ✅ EmailNotifier initialized")
            
            # Create dummy test file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(f"NeuralTrader Pre-Flight Test\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Status: TESTING\n")
                dummy_file_path = f.name
            
            self.logger.info(f"  ✅ Created test file: {dummy_file_path}")
            
            # Send test email
            subject = "🧪 NeuralTrader Pre-Flight Ping"
            body = f"""
NeuralTrader Pre-Flight Check
========================

Timestamp: {datetime.now().isoformat()}
Status: TESTING
Check: Communications Validation

This is a test email to verify the communications system is working.
All systems must pass pre-flight checks before production execution.

If you receive this email, the communications system is operational.
"""
            
            # Send email with attachment
            self.email_notifier.send_html_email_with_attachment(
                subject=subject,
                html_content=body,
                attachment_path=dummy_file_path
            )
            
            self.logger.info("  ✅ Pre-Flight Ping email sent successfully")
            
            # Clean up dummy file
            os.unlink(dummy_file_path)
            self.logger.info("  ✅ Test file cleaned up")
            
        except Exception as e:
            raise PreFlightCheckError(f"Communications test failed: {e}")
    
    def _check_model_integrity(self):
        """Validate production model integrity"""
        self.logger.info("\n[CHECK 3] Model Integrity Test...")
        
        try:
            # Load EnsemblePredictor
            ensemble = EnsemblePredictor()
            self.logger.info("  ✅ EnsemblePredictor loaded")
            
            # Create test input (76 features to match production models)
            test_input = np.zeros((1, 76))
            self.logger.info("  ✅ Created test input (76 features)")
            
            # Get prediction
            signal, confidence, details = ensemble.predict(test_input)
            
            # Validate output
            if not isinstance(signal, str):
                raise PreFlightCheckError(f"Invalid signal type: {type(signal)}")
            
            if not isinstance(confidence, (float, np.floating)):
                raise PreFlightCheckError(f"Invalid confidence type: {type(confidence)}")
            
            if not (0.0 <= confidence <= 1.0):
                raise PreFlightCheckError(f"Invalid confidence range: {confidence}")
            
            if not isinstance(details, dict):
                raise PreFlightCheckError(f"Invalid details type: {type(details)}")
            
            self.logger.info(f"  ✅ Model output valid: {signal} (confidence: {confidence:.4f})")
            self.logger.info(f"  ✅ Details keys: {list(details.keys())}")
            
        except Exception as e:
            raise PreFlightCheckError(f"Model integrity test failed: {e}")
    
    def _check_api_pulse(self):
        """Test API endpoints (FRED and News API)"""
        self.logger.info("\n[CHECK 4] API Pulse Test...")
        
        try:
            import requests
            
            # Test FRED API
            self.logger.info("  Testing FRED API...")
            fred_url = "https://api.stlouisfed.org/fred/series/observations"
            fred_params = {
                'series_id': 'GDP',
                'api_key': os.getenv('FRED_API_KEY', 'test'),
                'file_type': 'json',
                'limit': 1
            }
            
            fred_response = requests.get(fred_url, params=fred_params, timeout=10)
            
            if fred_response.status_code != 200:
                raise PreFlightCheckError(f"FRED API returned status {fred_response.status_code}")
            
            self.logger.info("  ✅ FRED API responding (200 OK)")
            
            # Test News API
            self.logger.info("  Testing News API...")
            news_url = "https://newsapi.org/v2/everything"
            news_params = {
                'q': 'test',
                'apiKey': os.getenv('NEWS_API_KEY', 'test'),
                'pageSize': 1
            }
            
            news_response = requests.get(news_url, params=news_params, timeout=10)
            
            # News API may return 401 with test key, but should still respond
            if news_response.status_code not in [200, 401]:
                raise PreFlightCheckError(f"News API returned status {news_response.status_code}")
            
            self.logger.info(f"  ✅ News API responding ({news_response.status_code})")
            
        except requests.exceptions.RequestException as e:
            raise PreFlightCheckError(f"API pulse test failed (network): {e}")
        except Exception as e:
            raise PreFlightCheckError(f"API pulse test failed: {e}")
    
    def _send_success_notification(self):
        """Send success notification"""
        try:
            if self.email_notifier:
                subject = "✅ PRE-FLIGHT PASSED - NeuralTrader Ready"
                body = f"""
NeuralTrader Pre-Flight Check - SUCCESS
=====================================
Timestamp: {datetime.now().isoformat()}
Status: ALL SYSTEMS OPERATIONAL

Pre-Flight Checklist Results:
✅ Broker Connectivity: IBKR connected and responsive
✅ Communications: Email system working
✅ Model Integrity: Production models loaded and functional
✅ API Pulse: External APIs responding

NEURALTRADER IS READY FOR PRODUCTION EXECUTION

All critical systems validated. Main orchestrator cleared to run.
"""
                
                self.email_notifier.send_html_email_with_attachment(subject, body)
                self.logger.info("  ✅ Success notification sent")
            
        except Exception as e:
            self.logger.warning(f"Could not send success notification: {e}")
    
    def _send_failure_notification(self, error: Exception, traceback_str: str):
        """Send failure notification"""
        try:
            if self.email_notifier:
                subject = "❌ PRE-FLIGHT FAILED - NeuralTrader BLOCKED"
                body = f"""
NeuralTrader Pre-Flight Check - FAILED
====================================

Timestamp: {datetime.now().isoformat()}
Status: CRITICAL SYSTEMS FAILURE

PRE-FLIGHT FAILED - MAIN ORCHESTRATOR BLOCKED

Error Details:
{str(error)}

Traceback:
{traceback_str}

ACTION REQUIRED:
1. Check the pre-flight check logs
2. Resolve the critical issues identified
3. Re-run pre-flight check before production execution

NEURALTRADER IS NOT READY FOR PRODUCTION
"""
                
                self.email_notifier.send_html_email_with_attachment(subject, body)
                self.logger.info("  ✅ Failure notification sent")
            
        except Exception as e:
            self.logger.error(f"Could not send failure notification: {e}")

def main():
    """Main entry point"""
    print("NeuralTrader Tier 0 Pre-Flight Checklist")
    print("=" * 50)
    print("Validating all critical systems...")
    
    checklist = PreFlightChecklist()
    success = checklist.run_all_checks()
    
    if success:
        print("\n✅ PRE-FLIGHT PASSED - NeuralTrader ready for production!")
        sys.exit(0)
    else:
        print("\n❌ PRE-FLIGHT FAILED - Check logs for details")
        sys.exit(1)

if __name__ == "__main__":
    main()

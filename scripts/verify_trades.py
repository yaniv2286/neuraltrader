#!/usr/bin/env python3
"""
NeuralTrader Trade Verification Script
====================================

Verifies if trades should have been taken but weren't executed.
Provides detailed analysis of missed opportunities and execution gaps.

Usage:
    python scripts/verify_trades.py [--date YYYY-MM-DD] [--send-email] [--detailed]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main function to verify trades"""
    parser = argparse.ArgumentParser(description='Verify NeuralTrader trades for missed opportunities')
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--send-email', action='store_true', help='Send verification report via email')
    parser.add_argument('--detailed', action='store_true', help='Show detailed analysis')
    parser.add_argument('--yesterday', action='store_true', help='Verify yesterday\'s trades')
    
    args = parser.parse_args()
    
    # Parse date
    if args.yesterday:
        date = datetime.now() - timedelta(days=1)
    elif args.date:
        try:
            date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        date = datetime.now()
    
    try:
        # Import trade verifier
        from src.utils.trade_verifier import TradeVerifier
        
        # Initialize verifier
        verifier = TradeVerifier()
        
        print(f"🔍 Verifying trades for {date.strftime('%Y-%m-%d')}...")
        print("=" * 60)
        
        # Run verification
        results = verifier.verify_trading_session(date)
        
        # Display summary
        summary = results.get("summary", {})
        missed_opps = results.get("missed_opportunities", [])
        execution_gaps = results.get("execution_gaps", [])
        
        print(f"📊 VERIFICATION SUMMARY:")
        print(f"   Total Signals: {summary.get('total_signals', 0)}")
        print(f"   Strong Signals: {summary.get('strong_signals', 0)}")
        print(f"   Approved Opportunities: {summary.get('approved_opportunities', 0)}")
        print(f"   Trades Executed: {summary.get('trades_executed', 0)}")
        print(f"   Missed Opportunities: {summary.get('missed_opportunities', 0)}")
        print(f"   Execution Rate: {summary.get('execution_rate', 0):.1%}")
        print(f"   Verification Status: {summary.get('verification_status', 'UNKNOWN')}")
        
        if summary.get('missed_pnl_estimate', 0) > 0:
            print(f"   💰 Missed P&L Estimate: ${summary.get('missed_pnl_estimate', 0):,.2f}")
        
        print()
        
        # Show missed opportunities
        if missed_opps:
            print(f"🚨 MISSED OPPORTUNITIES ({len(missed_opps)}):")
            print("-" * 40)
            for i, missed in enumerate(missed_opps[:3], 1):  # Show top 3
                print(f"   {i}. {missed.get('ticker', 'Unknown')}")
                print(f"      Signal: {missed.get('signal_strength', 0):.3f}")
                print(f"      Confidence: {missed.get('confidence', 0):.1%}")
                print(f"      Price: ${missed.get('current_price', 0):.2f}")
                print(f"      Position: {missed.get('recommended_position_size', 0)} shares")
                print(f"      Missed P&L: ${missed.get('potential_pnl', {}).get('potential_pnl', 0):.2f}")
                print(f"      Reason: {missed.get('missed_reason', 'Unknown')}")
                print()
        else:
            print("✅ No missed opportunities detected!")
        
        # Show execution gaps
        if execution_gaps:
            print(f"⚠️ EXECUTION GAPS:")
            print("-" * 40)
            for gap in execution_gaps:
                gap_type = gap.get('gap_type', 'Unknown')
                count = gap.get('count', 0)
                impact = gap.get('potential_impact', 0)
                
                print(f"   {gap_type}: {count} instances")
                if impact > 0:
                    print(f"   Impact: ${impact:,.2f}")
            print()
        
        # Detailed analysis if requested
        if args.detailed:
            print("📋 DETAILED ANALYSIS:")
            print("=" * 60)
            
            # Show all signals
            signals = results.get("signals_generated", [])
            if signals:
                print(f"📊 ALL SIGNALS GENERATED ({len(signals)}):")
                for signal in signals[:10]:  # Show top 10
                    print(f"   {signal.get('ticker', 'Unknown')}: {signal.get('signal_strength', 0):.3f}")
                print()
            
            # Show all opportunities
            opportunities = results.get("opportunities_analyzed", [])
            if opportunities:
                print(f"🎯 ALL OPPORTUNITIES ANALYZED ({len(opportunities)}):")
                for opp in opportunities[:10]:  # Show top 10
                    decision = opp.get('risk_decision', 'Unknown')
                    price = opp.get('current_price', 0)
                    print(f"   {opp.get('ticker', 'Unknown')}: {decision} @ ${price:.2f}")
                print()
        
        # Send email if requested
        if args.send_email:
            print("📧 Sending verification report via email...")
            success = verifier.send_verification_report(date)
            
            if success:
                print("✅ Verification report sent successfully!")
            else:
                print("❌ Failed to send verification report")
                sys.exit(1)
        
        # Exit with appropriate code
        if summary.get('verification_status') == 'ATTENTION_REQUIRED':
            print("⚠️ ATTENTION REQUIRED: Review missed opportunities!")
            sys.exit(1)
        else:
            print("✅ Verification completed successfully!")
            sys.exit(0)
        
    except Exception as e:
        print(f"❌ Error during trade verification: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

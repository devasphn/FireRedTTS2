#!/usr/bin/env python3
"""
Simple runner script for final system integration
Executes the complete final integration and system testing
"""

import asyncio
import sys
import logging
from pathlib import Path

# Import the final integration system
from final_system_integration import main as run_final_integration

def main():
    """Main entry point"""
    
    print("="*80)
    print("FireRedTTS2 Final System Integration")
    print("="*80)
    print()
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run the final integration
        print("Starting final system integration and testing...")
        print("This will:")
        print("1. Integrate all system components")
        print("2. Run comprehensive testing")
        print("3. Validate system with real user scenarios")
        print("4. Check all requirements are met")
        print("5. Create final deployment package")
        print()
        
        # Execute final integration
        exit_code = asyncio.run(run_final_integration())
        
        if exit_code == 0:
            print("\n" + "="*80)
            print("FINAL INTEGRATION COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("✅ All components integrated")
            print("✅ Testing completed")
            print("✅ System validation passed")
            print("✅ Requirements validated")
            print("✅ Deployment package created")
            print()
            print("The system is ready for deployment!")
        else:
            print("\n" + "="*80)
            print("FINAL INTEGRATION FAILED")
            print("="*80)
            print("❌ Integration or testing failed")
            print("Check the logs for details")
        
        return exit_code
    
    except KeyboardInterrupt:
        print("\n\nIntegration interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n\nFinal integration failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
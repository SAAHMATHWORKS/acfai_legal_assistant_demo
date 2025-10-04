#!/usr/bin/env python3
"""
Simple test to verify the infinite loop is fixed
"""

import asyncio
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


async def simple_assistance_test():
    """Test assistance workflow step by step"""
    from main import MultiCountryLegalRAGSystem
    
    system = MultiCountryLegalRAGSystem()
    
    try:
        print("\n" + "="*60)
        print("SIMPLE ASSISTANCE TEST")
        print("="*60)
        
        # Initialize
        print("\nInitializing...")
        if not await system.initialize():
            print("❌ Init failed")
            return
        print("✅ System ready\n")
        
        session_id = f"simple_test_{datetime.now().strftime('%H%M%S')}"
        
        # Test sequence fitahianarazafimahenina@gmail.com
        test_messages = [
            ("Je veux parler à un avocat", "Should ask for email"),
            ("fitahianarazafimahenina@gmail.com", "Should ask for description"),
            ("J'ai besoin d'aide pour un divorce au Bénin", "Should ask for confirmation"),
            ("oui", "Should trigger interrupt"),
            ("approve Demande légitime", "Should approve and send emails"),
        ]
        
        for i, (msg, expected) in enumerate(test_messages, 1):
            print(f"\n{'='*60}")
            print(f"STEP {i}: {expected}")
            print(f"{'='*60}")
            print(f"👤 User: {msg}")
            
            try:
                response = await system.chat(msg, session_id)
                print(f"\n🤖 Assistant:")
                print(response)
                
                # Check state
                state = await system.graph.aget_state(
                    {"configurable": {"thread_id": session_id}}
                )
                
                if state:
                    print(f"\n🔍 State Info:")
                    print(f"   - Next: {state.next}")
                    print(f"   - Email: {state.values.get('user_email')}")
                    print(f"   - Description: {state.values.get('assistance_description', 'None')[:50] if state.values.get('assistance_description') else 'None'}")
                    print(f"   - Step: {state.values.get('assistance_step')}")
                    print(f"   - Approval: {state.values.get('approval_status')}")
                
                # Small delay
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"\n❌ Error at step {i}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        
    finally:
        await system.cleanup()


if __name__ == "__main__":
    asyncio.run(simple_assistance_test())
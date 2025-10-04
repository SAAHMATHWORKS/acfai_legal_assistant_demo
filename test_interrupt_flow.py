# [test_interrupt_flow.py]
#!/usr/bin/env python3
"""
Test the interrupt flow with modular architecture
"""

import asyncio
import logging
from datetime import datetime

# Enable debug logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_interrupt_flow():
    """Test if assistance requests properly trigger interrupts"""
    from main import MultiCountryLegalRAGSystem
    
    system = MultiCountryLegalRAGSystem()
    
    try:
        print("🚀 Initializing system...")
        success = await system.initialize()
        if not success:
            print("❌ System initialization failed")
            return
        
        session_id = f"interrupt_test_{datetime.now().strftime('%H%M%S')}"
        print(f"📝 Session ID: {session_id}")
        
        print("\n" + "="*60)
        print("🎯 TESTING INTERRUPT FLOW")
        print("="*60)
        
        # Test direct assistance request with all info
        query = "Je veux parler à un avocat humain pour mon divorce au Bénin. Mon email est test@example.com et je veux une consultation téléphonique"
        print(f"👤 User: {query}")
        
        response = await system.chat(query, session_id)
        print(f"🤖 Assistant: {response}")
        
        # Check what happened
        if "APPROBATION HUMAINE REQUISE" in response:
            print("✅ SUCCESS: Interrupt properly triggered!")
            
            # Test moderator approval
            print("\n👨‍⚖️ Moderator: approve Demande légitime")
            final_response = await system.chat("approve Demande légitime", session_id)
            print(f"🤖 Assistant: {final_response}")
            
        elif "email" in response.lower():
            print("ℹ️  System is asking for more info (email/description)")
            print("This means the assistance workflow is working but not reaching interrupt yet")
            
        else:
            print("❌ Interrupt not triggered")
            print("Let me check the routing...")
            stats = system.get_global_stats()
            print(f"📊 Routing stats: {stats['routing_stats']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await system.cleanup()

if __name__ == "__main__":
    asyncio.run(test_interrupt_flow())
# [debug_assistance.py]
#!/usr/bin/env python3
"""
Debug script to trace assistance workflow
"""

import asyncio
import logging
from datetime import datetime

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def debug_assistance_step_by_step():
    """Debug the assistance workflow step by step"""
    from main import MultiCountryLegalRAGSystem
    
    system = MultiCountryLegalRAGSystem()
    
    try:
        print("🚀 Initializing system...")
        success = await system.initialize()
        if not success:
            print("❌ System initialization failed")
            return
        
        session_id = f"debug_{datetime.now().strftime('%H%M%S')}"
        print(f"📝 Session ID: {session_id}")
        
        print("\n" + "="*60)
        print("🔍 DEBUGGING ASSISTANCE WORKFLOW")
        print("="*60)
        
        # Step 1: Initial assistance request
        print("\n1️⃣  Sending assistance request...")
        query1 = "Je veux parler à un avocat humain"
        print(f"👤 User: {query1}")
        
        response1 = await system.chat(query1, session_id)
        print(f"🤖 Assistant: {response1}")
        
        # Check session state after step 1
        session_info = system.get_session_stats(session_id)
        print(f"📊 After step 1 - Query count: {session_info.get('query_count', 0)}")
        
        # Step 2: Provide email if requested
        if "email" in response1.lower():
            print("\n2️⃣  Providing email...")
            query2 = "test@example.com"
            print(f"👤 User: {query2}")
            
            response2 = await system.chat(query2, session_id)
            print(f"🤖 Assistant: {response2}")
            
        # Step 3: Provide description if requested
        if "description" in response2.lower():
            print("\n3️⃣  Providing description...")
            query3 = "Consultation téléphonique pour divorce"
            print(f"👤 User: {query3}")
            
            response3 = await system.chat(query3, session_id)
            print(f"🤖 Assistant: {response3}")
            
        # Step 4: Confirm if requested
        if "confirmer" in response3.lower():
            print("\n4️⃣  Confirming...")
            query4 = "oui"
            print(f"👤 User: {query4}")
            
            response4 = await system.chat(query4, session_id)
            print(f"🤖 Assistant: {response4}")
            
        print(f"\n📊 Final session stats: {system.get_session_stats(session_id)}")
        print(f"🌍 Global stats: {system.get_global_stats()}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await system.cleanup()

if __name__ == "__main__":
    asyncio.run(debug_assistance_step_by_step())
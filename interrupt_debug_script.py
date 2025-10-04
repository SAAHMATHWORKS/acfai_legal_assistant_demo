#!/usr/bin/env python3
"""
Interrupt Testing and Debugging Script
Run this to test the human approval interrupt workflow
"""

import asyncio
import logging
from datetime import datetime

# Setup logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


async def test_interrupt_flow():
    """Test the complete interrupt workflow"""
    from main import MultiCountryLegalRAGSystem
    
    system = MultiCountryLegalRAGSystem()
    
    try:
        print("\n" + "="*70)
        print("🧪 INTERRUPT WORKFLOW TEST")
        print("="*70)
        
        # Initialize system
        print("\n1️⃣ Initializing system...")
        success = await system.initialize()
        if not success:
            print("❌ Initialization failed")
            return
        print("✅ System initialized\n")
        
        # Create unique session
        session_id = f"interrupt_test_{datetime.now().strftime('%H%M%S')}"
        print(f"📋 Session ID: {session_id}\n")
        
        # Step 1: Request assistance
        print("="*70)
        print("STEP 1: User requests to talk to a lawyer")
        print("="*70)
        user_query = "Je veux parler à un avocat"
        print(f"👤 User: {user_query}")
        
        response1 = await system.chat(user_query, session_id)
        print(f"🤖 Assistant:\n{response1}\n")
        
        # Check state after step 1
        state1 = await system.graph.aget_state(
            {"configurable": {"thread_id": session_id}}
        )
        print(f"🔍 State after step 1:")
        print(f"   - Next nodes: {state1.next if state1 else 'None'}")
        print(f"   - Router decision: {state1.values.get('router_decision') if state1 else 'None'}")
        print()
        
        # Step 2: Provide email
        print("="*70)
        print("STEP 2: User provides email")
        print("="*70)
        email = "test@example.com"
        print(f"👤 User: {email}")
        
        response2 = await system.chat(email, session_id)
        print(f"🤖 Assistant:\n{response2}\n")
        
        # Check state after step 2
        state2 = await system.graph.aget_state(
            {"configurable": {"thread_id": session_id}}
        )
        print(f"🔍 State after step 2:")
        print(f"   - Next nodes: {state2.next if state2 else 'None'}")
        print(f"   - User email: {state2.values.get('user_email') if state2 else 'None'}")
        print()
        
        # Step 3: Provide description
        print("="*70)
        print("STEP 3: User provides description")
        print("="*70)
        description = "J'ai besoin d'aide pour un divorce au Bénin"
        print(f"👤 User: {description}")
        
        response3 = await system.chat(description, session_id)
        print(f"🤖 Assistant:\n{response3}\n")
        
        # Check state after step 3
        state3 = await system.graph.aget_state(
            {"configurable": {"thread_id": session_id}}
        )
        print(f"🔍 State after step 3:")
        print(f"   - Next nodes: {state3.next if state3 else 'None'}")
        print(f"   - Description: {state3.values.get('assistance_description') if state3 else 'None'}")
        print()
        
        # Step 4: Confirm
        print("="*70)
        print("STEP 4: User confirms")
        print("="*70)
        confirmation = "oui"
        print(f"👤 User: {confirmation}")
        
        response4 = await system.chat(confirmation, session_id)
        print(f"🤖 Assistant:\n{response4}\n")
        
        # Check state after step 4 - should be interrupted
        state4 = await system.graph.aget_state(
            {"configurable": {"thread_id": session_id}}
        )
        print(f"🔍 State after step 4 (SHOULD BE INTERRUPTED):")
        print(f"   - Next nodes: {state4.next if state4 else 'None'}")
        print(f"   - Approval status: {state4.values.get('approval_status') if state4 else 'None'}")
        
        if state4 and state4.next:
            print(f"   ✅ INTERRUPT DETECTED at: {state4.next}")
            print(f"   ⏸️  Graph is paused and waiting for moderator input\n")
            
            # Step 5: Moderator approves
            print("="*70)
            print("STEP 5: Moderator approves")
            print("="*70)
            moderator_decision = "approve Demande légitime de consultation juridique"
            print(f"👨‍⚖️ Moderator: {moderator_decision}")
            
            response5 = await system.chat(moderator_decision, session_id)
            print(f"🤖 Assistant:\n{response5}\n")
            
            # Check final state
            state5 = await system.graph.aget_state(
                {"configurable": {"thread_id": session_id}}
            )
            print(f"🔍 Final state:")
            print(f"   - Next nodes: {state5.next if state5 else 'None'}")
            print(f"   - Approval status: {state5.values.get('approval_status') if state5 else 'None'}")
            print(f"   - Email status: {state5.values.get('email_status') if state5 else 'None'}")
            
            if state5 and state5.values.get('approval_status') == 'approved':
                print("\n✅ TEST PASSED: Request was approved and processed!")
            else:
                print("\n❌ TEST FAILED: Approval status incorrect")
        else:
            print(f"   ❌ NO INTERRUPT DETECTED!")
            print(f"   Expected interrupt but graph continued to completion")
            print("\n❌ TEST FAILED: Interrupt was not triggered")
        
        print("\n" + "="*70)
        print("TEST COMPLETE")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Test error: {str(e)}", exc_info=True)
    finally:
        await system.cleanup()


async def debug_graph_structure():
    """Debug the graph structure to ensure interrupt is configured"""
    from main import MultiCountryLegalRAGSystem
    
    system = MultiCountryLegalRAGSystem()
    
    try:
        print("\n" + "="*70)
        print("🔍 GRAPH STRUCTURE DEBUG")
        print("="*70)
        
        await system.initialize()
        
        # Check interrupt configuration
        print("\n📊 Graph Configuration:")
        print(f"   - Graph compiled: {system.graph is not None}")
        print(f"   - Checkpointer: {system.postgres_checkpointer is not None}")
        
        # Try to inspect graph structure
        if hasattr(system.graph, 'nodes'):
            print(f"\n🔹 Graph nodes: {list(system.graph.nodes.keys())}")
        
        # Check if human_approval node exists
        if system.graph and hasattr(system.graph, 'nodes'):
            if 'human_approval' in system.graph.nodes:
                print(f"   ✅ human_approval node found")
            else:
                print(f"   ❌ human_approval node NOT found!")
        
        print("\n" + "="*70)
        
    except Exception as e:
        logger.error(f"Debug error: {str(e)}", exc_info=True)
    finally:
        await system.cleanup()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test interrupt workflow")
    parser.add_argument(
        "--mode",
        choices=["test", "debug"],
        default="test",
        help="test=run full workflow, debug=check graph structure"
    )
    
    args = parser.parse_args()
    
    if args.mode == "test":
        asyncio.run(test_interrupt_flow())
    else:
        asyncio.run(debug_graph_structure())
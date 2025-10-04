# generate_graph.py (example)
from graphviz import Digraph

def generate_graph():
    dot = Digraph(comment='Legal RAG System Workflow')
    dot.attr(rankdir='TB', size='8,5')
    dot.attr('node', shape='box', style='filled', fillcolor='#e6f3ff')

    # Core nodes
    dot.node("START", fillcolor="#90ee90")
    dot.node("ROUTER", label="router")
    dot.node("RESPONSE", label="response", fillcolor="#98fb98")
    
    # Country nodes
    dot.node("BENIN_RETRIEVAL", label="benin_retrieval")
    dot.node("MADAGASCAR_RETRIEVAL", label="madagascar_retrieval")
    
    # Handler nodes
    dot.node("GREETING", label="greeting_handler")
    dot.node("REPAIR", label="repair_handler")
    dot.node("SUMMARY", label="summary_handler")
    dot.node("UNCLEAR", label="unclear_handler")
    dot.node("OUT_OF_SCOPE", label="out_of_scope_handler")
    
    # Assistance nodes
    dot.node("ASSIST_COLLECT", label="assistance_collect_info")
    dot.node("ASSIST_CONFIRM", label="assistance_confirm")
    dot.node("HUMAN_APPROVAL", label="human_approval", fillcolor="#ffa07a")
    
    # End node
    dot.node("END", fillcolor="#ff9999")
    
    # Edges
    dot.edge("START", "ROUTER")
    dot.edge("ROUTER", "BENIN_RETRIEVAL", label="benin")
    dot.edge("ROUTER", "MADAGASCAR_RETRIEVAL", label="madagascar")
    dot.edge("ROUTER", "GREETING", label="greeting_small_talk")
    dot.edge("ROUTER", "REPAIR", label="conversation_repair")
    dot.edge("ROUTER", "SUMMARY", label="conversation_summarization")
    dot.edge("ROUTER", "UNCLEAR", label="unclear")
    dot.edge("ROUTER", "OUT_OF_SCOPE", label="out_of_scope")
    dot.edge("ROUTER", "ASSIST_COLLECT", label="assistance_request")
    
    dot.edge("GREETING", "RESPONSE")
    dot.edge("REPAIR", "RESPONSE")
    dot.edge("SUMMARY", "RESPONSE")
    dot.edge("UNCLEAR", "RESPONSE")
    dot.edge("OUT_OF_SCOPE", "RESPONSE")
    
    dot.edge("ASSIST_COLLECT", "RESPONSE", label="need_email/need_description")
    dot.edge("ASSIST_COLLECT", "ASSIST_CONFIRM", label="ready_to_confirm")
    dot.edge("ASSIST_COLLECT", "RESPONSE", label="cancelled")
    
    dot.edge("ASSIST_CONFIRM", "HUMAN_APPROVAL", label="confirmed")
    dot.edge("ASSIST_CONFIRM", "RESPONSE", label="needs_correction/cancelled")
    
    dot.edge("HUMAN_APPROVAL", "RESPONSE")
    
    dot.edge("RESPONSE", "ASSIST_COLLECT", label="continue_assistance")
    dot.edge("RESPONSE", "END", label="end")
    
    dot.render('legal_rag_workflow', format='png', cleanup=True)
    print("Graph visualization generated: legal_rag_workflow.png")

if __name__ == "__main__":
    generate_graph()
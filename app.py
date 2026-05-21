import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from main import get_agent

st.set_page_config(page_title="NewsGeni Chat", page_icon="📰")
st.title("NewsGeni Agent")
st.caption("Ask me anything or get the latest news.")

# Initialize the agent and conversation history
if "agent" not in st.session_state:
    st.session_state.agent = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input
if prompt := st.chat_input("What's happening in the world?"):
    # Append user message
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking and searching..."):
            # Run the agent graph
            output = st.session_state.agent.invoke({"messages": st.session_state.messages})
            
            # Get the final AI response (last message in state)
            final_response = output["messages"][-1]
            st.markdown(final_response.content)
            
            # Save the full state back to history
            st.session_state.messages = output["messages"]

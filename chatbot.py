import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found! Please add it to your .env file.")
        st.stop()
    return Groq(api_key=api_key)

client = get_groq_client()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a helpful, friendly, and knowledgeable AI assistant."
        }
    ]

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Model selection
    model = st.selectbox(
        "Select Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ],
        index=0
    )
    
    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random, lower values more focused"
    )
    
    # Max tokens
    max_tokens = st.slider(
        "Max Tokens",
        min_value=256,
        max_value=4096,
        value=1024,
        step=256,
        help="Maximum length of the response"
    )
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "system",
                "content": "You are a helpful, friendly, and knowledgeable AI assistant."
            }
        ]
        st.session_state.display_messages = []
        st.rerun()
    
    # Export chat button
    if st.session_state.display_messages:
        chat_export = "\n\n".join([
            f"**{msg['role'].title()}**: {msg['content']}" 
            for msg in st.session_state.display_messages
        ])
        st.download_button(
            label="📥 Export Chat",
            data=chat_export,
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.divider()
    
    # Info section
    with st.expander("ℹ️ About"):
        st.markdown("""
        **AI Chatbot** powered by Groq
        
        - Fast responses using Llama models
        - Conversation history tracking
        - Customizable parameters
        
        Built with Streamlit & Groq API
        """)
    
    # Stats
    with st.expander("📊 Chat Stats"):
        msg_count = len(st.session_state.display_messages)
        st.metric("Messages", msg_count)
        
        if msg_count > 0:
            user_msgs = len([m for m in st.session_state.display_messages if m["role"] == "user"])
            assistant_msgs = len([m for m in st.session_state.display_messages if m["role"] == "assistant"])
            st.metric("Your Messages", user_msgs)
            st.metric("AI Responses", assistant_msgs)

# Main chat interface
st.title("🤖 AI Chatbot")
st.caption("Powered by Groq API")

# Display chat messages
for message in st.session_state.display_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to display
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to conversation history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Stream the response
            stream = client.chat.completions.create(
                model=model,
                messages=st.session_state.messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            # Display streaming response
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            # Display final response
            message_placeholder.markdown(full_response)
            
            # Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })
            st.session_state.display_messages.append({
                "role": "assistant",
                "content": full_response
            })
            
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.display_messages.append({
                "role": "assistant",
                "content": error_msg
            })

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Made with ❤️ using Streamlit and Groq API</p>
    </div>
""", unsafe_allow_html=True)
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def chat(user_message, conversation_history):
    """
    Send a message to the chatbot and get a response
    
    Args:
        user_message: The user's input message
        conversation_history: List of previous messages
    
    Returns:
        The chatbot's response
    """
    # Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    try:
        # Get response from Groq
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Fast and capable model
            messages=conversation_history,
            temperature=0.7,
            max_tokens=1024
        )
        
        # Extract assistant's reply
        assistant_message = response.choices[0].message.content
        
        # Add assistant's response to history
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    """Main chatbot loop"""
    print("=" * 50)
    print("Welcome to the Chatbot!")
    print("=" * 50)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    print("-" * 50)
    
    # Initialize conversation history with system message
    conversation_history = [
        {
            "role": "system",
            "content": "You are a helpful, friendly, and knowledgeable assistant."
        }
    ]
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye! Thanks for chatting!")
            break
        
        # Check for clear command
        if user_input.lower() == 'clear':
            conversation_history = [
                {
                    "role": "system",
                    "content": "You are a helpful, friendly, and knowledgeable assistant."
                }
            ]
            print("\n[Conversation cleared]")
            continue
        
        # Skip empty messages
        if not user_input:
            continue
        
        # Get and display chatbot response
        print("\nChatbot: ", end="", flush=True)
        response = chat(user_input, conversation_history)
        print(response)

if __name__ == "__main__":
    main()
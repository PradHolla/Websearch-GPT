import streamlit as st
import uuid
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import logging
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Web Search Bot", page_icon="🔗", layout="wide")

st.header("Welcome to the Web Search Bot")

st.sidebar.title("History")

# Setup Tavily Search Tool
search = TavilySearchResults(max_results=5)

# Initialize Language Model
llm = ChatBedrock(model="us.meta.llama3-2-11b-instruct-v1:0")

def determine_need_for_search(messages):
    system_prompt = """
    You are a Search Decision Maker. Your task is to decide whether a web search is necessary.

    Rules:
    - Output 'SEARCH' if the query requires up-to-date or external information
    - Output 'CHAT' if the query can be answered from existing conversation context
    - Respond ONLY with 'SEARCH' or 'CHAT'

    Consider searching for:
    - Current events
    - Recent developments
    - Specific factual information
    - Technical or scientific queries
    - Questions about recent trends or news

    Avoid searching for:
    - Small talk
    - Conversational pleasantries
    - Topics already discussed in the conversation
    - Follow-up questions based on the conversation history
    """
    
    # Combine message contents for context
    conversation_history = "\n".join([msg.content for msg in messages])
    
    try:
        # Use a single LLM call with a clear, concise prompt
        decision = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Conversation History:\n{conversation_history}\n\nDecide: SEARCH or CHAT?")
        ]).content.strip().upper()
        
        # Optional: Add debug information
        # print(f"Search Decision: {decision}")
        
        return decision == "SEARCH"
    
    except Exception as e:
        print(f"Search decision error: {e}")
        return False

# Perform web search
def perform_web_search(query):
    try:
        search_results = search.invoke(query)
        formatted_results = "\n\n".join([
            f"Source {i+1} ({result['url']}):\n{result['content']}" 
            for i, result in enumerate(search_results)
        ])
        
        sources = "\n" + "\n".join([
            f"{i+1}. {result['url']}" 
            for i, result in enumerate(search_results)
        ])

        return formatted_results, sources
    except Exception as e:
        st.error(f"Search error: {e}")
        return None, None

# Generate AI response
def generate_response(question, search_results=None):
    system_prompt = """
    Using the following web search results, provide a comprehensive answer to the question.
    
    Guidelines:
    - Base your answer solely on the provided search results
    - Be clear and concise
    - Include key information from the sources
    - If there are no search results provided and the question is a follow-up, provide a relevant answer based on the conversation history
    """
    messages = [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
    ]
    
    # If search results are available, add them as a system message
    if search_results:
        messages.append(
            ("system", f"Search Results:\n{search_results}")
        )
    
    # Add the human question
    messages.append(("human", "{question}"))

    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,  # Always return the instance created earlier
    input_messages_key="question",
    history_messages_key="history",
    )

    config = {"configurable": {"session_id": "any"}}

    try:
        full_response = ""
        response_container = st.empty()

        for chunk in chain_with_history.stream({"question": question}, config=config):
            full_response += chunk.content
            response_container.markdown(full_response + "")

        return full_response
    except Exception as e:
        st.error(f"Response generation error: {e}")
        return "I'm sorry, I couldn't generate a response."

msgs = StreamlitChatMessageHistory(key="chat_messages")

# Display chat messages
for msg in msgs.messages:
    if msg.type == "human":
        st.chat_message("human").write(msg.content)
    elif msg.type == "AIMessageChunk":
        st.chat_message("ai").write(msg.content)

# User input
if prompt := st.chat_input("Ask me anything..."):
        # Display user message
        st.chat_message("human").write(prompt)

        # Determine if search is needed
        need_search = determine_need_for_search(msgs.messages)

        # Prepare to show loading state
        with st.chat_message("ai"):
            if need_search:
                with st.spinner("Searching the web for the most up-to-date information..."):
                    search_results, sources = perform_web_search(prompt)
            else:
                search_results, sources = None, None

            # Generate response
            print(f"\nSearch results: {search_results}")
            response = generate_response(prompt, search_results)

            # Append sources if available
            if sources:
                with st.expander("Sources"):
                    st.markdown(sources)

if st.sidebar.button("Clear Conversation"):
    msgs.clear()
    st.rerun()
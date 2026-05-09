import streamlit as st
from langchain_core.messages import HumanMessage
from chatbot_agent import chatbot,store_info
import uuid


def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def delete_thread(thread_id):
    """Delete thread from SQLite DB and session state."""
    from chatbot_agent import checkpointer
    try:
        conn = checkpointer.conn
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (str(thread_id),))
        cursor.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ?", (str(thread_id),))
        cursor.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (str(thread_id),))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error deleting: {e}")
        return False
def load_conversation():
    try:
        conversation = chatbot.get_state(config={"configurable":{"thread_id":thread_id}}).values["messages"]
        if conversation is None:
           st.info("✨ Your chat journey starts here! Type your first message to begin.")
           conversation = []
    except Exception as e:
        # Log or display error for debugging
        st.info("✨ Welcome! A new chat session has started. Let's begin your conversation.")
        conversation = []
    return conversation




if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = store_info()

add_thread(st.session_state["thread_id"])



st.sidebar.title("Langchain Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for tid in st.session_state["chat_threads"][::-1]:
    is_active = tid == st.session_state["thread_id"]
    btn_label = f"**· {str(tid)[:18]}...**" if is_active else f"{str(tid)[:18]}..."
    
    col_d, col_t = st.sidebar.columns([1, 4])
    
    if col_d.button("🗑️", key=f"d_{tid}"):
        if delete_thread(tid):
            st.session_state["chat_threads"] = [t for t in st.session_state["chat_threads"] if t != tid]
            if st.session_state["thread_id"] == tid:
                reset_chat()
            st.toast("Deleted")
            st.rerun()

    if col_t.button(btn_label, key=f"t_{tid}", use_container_width=True):
        st.session_state["thread_id"] = tid
        messages = load_conversation()
        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role":role, "content":msg.content})
        st.session_state["message_history"] = temp_messages
        st.rerun()


for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])




user_input = st.chat_input("Type here")

if user_input:
    st.session_state["message_history"].append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.text(user_input)

    config = {"configurable":{"thread_id":st.session_state["thread_id"]}}
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": st.session_state["thread_id"]}},
                stream_mode = "messages"
            )
        )

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

from openai import OpenAI
import pandas as pd
import numpy as np
import streamlit as st
import shelve
import json

# Character generation methods
def generate_character_from_race(race_name : str):
    df = pd.read_csv("filtered_df.csv")
    if len(df[df['race'] == race_name]) == 0:
        return f"No data was found for the race {race_name}. Please re-prompt the user."
    
    random_class = df[df['race'] == race_name]['class_starting'].sample().iloc[0]
    
    random_background = df[(df['race'] == race_name) & (df['class_starting'] == random_class)]['background'].sample().iloc[0]
    return f"Here's your generated character: \nRace: {race_name} \nClass: {random_class} \nBackground: {random_background}"

def generate_character_from_class(class_name : str):
    df = pd.read_csv("filtered_df.csv")
    if len(df[df['class_starting'] == class_name]) == 0:
        return f"No data was found for the class {class_name}. Please re-prompt the user."
    
    random_race = df[(df['class_starting'] == class_name) & (df['race'] != 'Homebrew')]['race'].sample().iloc[0]
    
    most_common_races = df['race'].value_counts().nlargest(13).index
    rarer_race = df[(df['class_starting'] == class_name) & (~df['race'].isin(most_common_races))]['race'].sample().iloc[0]
    
    random_background = df[(df['race'] == random_race) & (df['class_starting'] == class_name)]['background'].sample().iloc[0]
    return f"Here's your generated character: \nRace: {random_race} \nClass: {class_name} \nBackground: {random_background} \nAlternative Uncommon Race Choice: {rarer_race}"

# Function definitions for chatbot
tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_character_from_class",
            "description": "Given a DnD class, this function will generate a DnD character based on the data. Will provide a second, rarer option.",
            "parameters": {
                "type": "object",
                "required": ["class_name"],
                "properties": {
                    "class_name": {
                    "type": "string",
                    "description": "The name of the DnD class for the character"
                    }
                },
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_character_from_race",
            "description": "Given a DnD race, this function will generate a DnD character based on the data.",
            "parameters": {
                "type": "object",
                "required": ["race_name"],
                "properties": {
                    "race_name": {
                    "type": "string",
                    "description": "The name of the DnD race for the character"
                    }
                },
                "additionalProperties": False
            }
        }
    }
]

system_messages = [
        {"role": "system", "content": """You are a Dungeons and Dragons character generator. Provide only the character's 
         race, class, background, and name. If prompted to create a character using one of the two, use the data to generate a character randomly.
         If there is no data on the provided race or class, prompt the user again for a race or class."""}
        ]

key = ""

st.title("Dungeons & Dragons 5e Character Generator")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"
client = OpenAI(api_key = key)

# Ensure openai_model is initialized in session state
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"


# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])


# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages
        
        
# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Sidebar with a button to delete chat history
with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Main chat interface
if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        
        response = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=system_messages + st.session_state["messages"],
            tools=tools,
            tool_choice="auto",
            # stream=True,
        )
        
        # Function call handling
        tool_calls = response.choices[0].message.tool_calls

        if tool_calls:
            conversation = system_messages + st.session_state["messages"]
            conversation.append(response.choices[0].message)

            for tool_call in tool_calls:

                if tool_call.function.name == "generate_character_from_race":
                    arguments = json.loads(tool_call.function.arguments)
                    race_name = arguments['race_name'].title().strip()
                    function_result = generate_character_from_race(race_name)
                    function_call_result = {"role": "tool",
                                            "content": function_result,
                                            "tool_call_id": tool_call.id
                                            }
                    conversation.append(function_call_result)

                elif tool_call.function.name == "generate_character_from_class":
                    arguments = json.loads(tool_call.function.arguments)
                    class_name = arguments['class_name'].title().strip()
                    function_result = generate_character_from_class(class_name)
                    function_call_result = {"role": "tool",
                                            "content": function_result,
                                            "tool_call_id": tool_call.id
                                            }
                    conversation.append(function_call_result)

                response = client.chat.completions.create(
                    model = "gpt-4o-mini",
                    messages = conversation,
                    # stream=True
                )
        
        # TODO: figure out if this can be reimplemented
        # for text in response:
        #     full_response += text.choices[0].delta.content or ""
        #     message_placeholder.markdown(full_response + "|")
            
        full_response = response.choices[0].message.content
            
            
            
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Save chat history after each interaction
save_chat_history(st.session_state.messages)
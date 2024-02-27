import streamlit as st
import json

# from writers_room.agents_pipeline_v2 import run_agent_pipeline
# from writers_room.supervisor_agent_pipeline import run_agent_pipeline
from writers_room.supervisor_agent_pipeline_no_critique import run_agent_pipeline


# Set up the main page
st.title("Writer's Room Langgraph 🦜🕸️🚀")
st.subheader(
    "Input a topic to generate newsletter with accompanying Tweets & LinkedIn posts"
)
# Input text box
topic = st.text_input("Topic")

if "result" not in st.session_state:
    st.session_state["result"] = None
# Submit button
if st.button("Submit"):
    st.write(f"Topic submitted: {topic}")
    st.session_state["result"] = run_agent_pipeline(topic)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", ("Tweets", "Newsletter Draft", "LinkedIn Posts", "Visualization")
)


def load_json_content(filepath):
    with open(filepath, "r") as file:
        return json.load(file)


# Display the selected page and its content
if page == "Tweets" and st.session_state["result"]:
    tweets_content_str = st.session_state["result"].get("tweets", {}).get("tweets", "")
    if tweets_content_str:
        print(f"TWEETS PATH: {tweets_content_str}")
        st.subheader("Generated Tweets")
        try:
            tweets_content = json.loads(tweets_content_str)
            nested_tweets = tweets_content.get("tweets", [])
            for tweet in nested_tweets:
                st.write(tweet)
        except json.JSONDecodeError:
            st.write("Error decoding tweets JSON.")

elif page == "Newsletter Draft" and st.session_state["result"]:
    draft_str = st.session_state["result"].get("newsletter_draft").get("draft", "")
    print(f"DRAFT PATH: {draft_str}")
    if draft_str:
        # draft_content = json.loads(draft_str)

        st.subheader("Newsletter Draft")
        st.write(draft_str)
    else:
        st.write("No newsletter draft to display. Please submit a topic first.")

elif page == "LinkedIn Posts":
    st.sidebar.write("LinkedIn Posts content goes here. (To be implemented)")

elif page == "Visualization":
    st.sidebar.write("Visualization content goes here. (To be implemented)")

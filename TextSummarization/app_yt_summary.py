import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
import re

st.set_page_config(page_title="YouTube Video Summarizer", page_icon="🦜")
st.title("🦜 YouTube Video Summarizer with Groq & LangChain")
st.subheader("Summarize any YouTube video in seconds!")

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type="password")
    st.markdown("""
    **Instructions:**
    1. Enter your Groq API key.
    2. Paste a YouTube video URL below.
    3. Choose your summary options.
    4. Click 'Summarize Video'.
    """)

youtube_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

# --- New options ---
summary_length = st.selectbox(
    "Summary Length",
    options=["Short (100 words)", "Medium (300 words)", "Long (500 words)"],
    index=1
)
summary_style = st.selectbox(
    "Summary Style",
    options=["Paragraph", "Bullet Points", "Key Takeaways"],
    index=0
)
custom_prompt = st.text_area(
    "Custom Prompt (optional)",
    value="",
    help="Override the default prompt for advanced users. Use {text} as a placeholder for the transcript."
)

def get_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

# --- Prompt logic ---
length_map = {
    "Short (100 words)": "100 words",
    "Medium (300 words)": "300 words",
    "Long (500 words)": "500 words"
}
style_map = {
    "Paragraph": "as a single paragraph.",
    "Bullet Points": "as bullet points.",
    "Key Takeaways": "as a list of key takeaways."
}

if custom_prompt.strip():
    prompt_template = custom_prompt
else:
    prompt_template = f"""
Provide a concise summary ({length_map[summary_length]}) of the following YouTube video transcript {style_map[summary_style]}
Transcript:
{{text}}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize Video"):
    video_id = get_video_id(youtube_url)
    if not video_id:
        st.error("Invalid YouTube URL.")
    elif not groq_api_key:
        st.error("Please provide your Groq API key.")
    else:
        try:
            with st.spinner("Loading video transcript..."):
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                text = " ".join([x['text'] for x in transcript])
                st.write("Transcript Preview:", text[:500] + ("..." if len(text) > 500 else ""))
            with st.spinner("Summarizing video..."):
                llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run([type('Doc', (), {'page_content': text, 'metadata': {}})()])
                st.success("**Summary:**")
                st.write(summary)
        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

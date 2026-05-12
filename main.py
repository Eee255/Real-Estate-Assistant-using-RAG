# @Author: Bhanu Prakash

import streamlit as st
from rag import process_urls, generate_answer

st.title("Real Estate Research Tool")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

process_url_button = st.sidebar.button("Process URLs")

if process_url_button:
    urls = [url for url in (url1, url2, url3) if url != '']
    if len(urls) == 0:
        st.warning("You must provide at least one valid URL")
    else:
        with st.status("Processing URLs..."):        # ✅ shows live status
            for status in process_urls(urls):
                st.write(status)
        st.success("URLs processed successfully!")

st.divider()

query = st.text_input("Question")                   # ✅ separate from placeholder
if query:
    try:
        answer, sources = generate_answer(query)

        st.header("Answer:")
        st.write(answer)

        if sources:
            st.subheader("Sources:")
            for source in sources.split(","):
                st.write(source.strip())

    except RuntimeError:
        st.error("You must process URLs first before asking a question.")
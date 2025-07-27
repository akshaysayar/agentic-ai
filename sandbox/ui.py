# moved from project root

import streamlit as st
import streamlit.components.v1 as components
import requests

def main():
    st.title("Profile Agent UI")
    query = st.text_input("Enter your query:")
    if st.button("Ask"):        
        response = requests.post("http://localhost:8002/search", json={"query": query})
        if response.ok:
            answer = response.json().get("answer", "No answer returned.")
            st.write("**Answer:**", answer)
        else:
            st.write("Error:", response.text)

if __name__ == "__main__":
    main()

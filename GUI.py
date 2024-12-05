import streamlit as st,requests
with st.sidebar:
    st.title("ABOUT 👇")
    st.write("Fake news detection using machine learning is a project that involves training a machine learning model to distinguish between real and fake news articles. The model is trained on a dataset of news articles using supervised learning algorithms and NLP techniques. The goal of the project is to help identify and flag fake news articles to prevent their spread and reduce the impact of misinformation in society.")
    st.divider()
    st.title("Created By: ")
    st.write(":rainbow[Bruce-Arhin Shadrach  & Appiah Derick]")
    
    

st.header(":red[FAKE NEWS DETECTION USING  PASSIVE AGGRESSIVE CLASSIFIER ]")
st.divider()

if text:=st.chat_input("Enter news headlines "):
    
        try:
            url="http://127.0.0.1:8500/Model_prediction"
            with st.spinner():
                proced=requests.post(url,json={'txt':text})
                if proced.status_code==200:
                    with st.chat_message('assistant'):
                        st.markdown(proced.text)
        except Exception as ex:
            st.warning(f":red[Server Error occured, Please check your internet connection and try again]", icon="⚠️")    

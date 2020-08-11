import streamlit as st
import pandas as pd
import numpy as np
import joblib

html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Awesome AI App </h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)

def load_css(file_name):
  with open(file_name) as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


# Can try loading trained pickle models in this format
# gender_nv_model = open("models/naivebayesgendermodel.pkl","rb")
# gender_clf = joblib.load(gender_nv_model)

def main():
  load_css('style.css')

  st.set_option('deprecation.showfileUploaderEncoding', False)
  uploaded_file = st.file_uploader("Upload your Image")
  if uploaded_file is not None:
    # data = pd.read_csv(uploaded_file)
    st.image(uploaded_file, caption='Sameple image', use_column_width=True)

  if st.button("Predict"):
    '''
    Can run model predict method here
    '''
    st.text("6 cells")

main()

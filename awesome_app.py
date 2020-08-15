import streamlit as st
import pandas as pd
import numpy as np
import joblib
from models import *

html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Awesome AI App for Nuclear Segmentation</h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)

def load_css(file_name):
  with open(file_name) as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

model = ResNetUNet(5)
model.load_state_dict(torch.load("best_weight_bs_1.pt", map_location=torch.device('cpu')))

def predict(option):
  label_path = "Test/Labels/"
  img_path = "Test/Images/"
  test_set = NucleiDataset(img_path,label_path, transform = trans, idx=option)

  batch_size = 1

  test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

  inputs, masks = next(iter(test_loader))
  st.text("Input Image")
  plt.imshow(reverse_transform(inputs[0]))
  st.pyplot()
  outputs = model(inputs)

  st.text('Target')
  mask = masks.numpy()[0]
  target = np.zeros(mask[0].shape)
  for i in range(5):
      mask_i = mask[i]
      target += i * mask_i
  plt.imshow(target, cmap='jet', vmax=np.max(target), vmin=np.min(target))
  st.pyplot()

  st.text('Predicted image')
  threshold = 0.8
  pred = outputs.to('cpu').detach().numpy()[0]
  output = np.zeros(pred[0].shape)
  for i in range(5):
      pred_i = pred[i]
      pred_i[pred_i >= threshold] = 1
      pred_i[pred_i < threshold] = 0
      output += i * pred_i
  plt.imshow(output,  cmap='jet', vmax=np.max(target), vmin=np.min(target))
  st.pyplot()

def main():
  load_css('style.css')

  option = st.selectbox('Select an image',(1,2,3,4,5,6,7,8,9,10,11,12,13,14))

  label_path = "Test/Labels/"
  img_path = "Test/Images/"

  st.image(img_path + sorted(os.listdir(img_path))[option-1], caption="Selected Image", use_column_width=True)

  with st.spinner('Please wait...'):
    if st.button("Predict"):
      predict(option-1)
    
main()

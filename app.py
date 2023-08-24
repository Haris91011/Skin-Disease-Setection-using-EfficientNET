#Import Packages
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import models, transforms
import torch.nn as nn
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

icon = Image.open('skin.jpg')
st.set_page_config(page_title='EfficientSkinDis-xeven', page_icon = icon)
st.header('Skin Disease Detection ')

#Load Model
def effNetb2():
    model = models.efficientnet_b2(pretrained=False).to(device)
    in_features = 1024
    model._fc = nn.Sequential(
        nn.BatchNorm1d(num_features=in_features),    
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=128),
        nn.Dropout(0.4),
        nn.Linear(128, 31),).to(device)

    model.load_state_dict(torch.load('Weights.h5' , map_location=torch.device('cpu')) )
    model.eval()

    return model

#Calculating Prediction
def Predict(img):
    allClasses = ['Actinic Keratosis', 'Basal Cell Carcinoma',
	      'Dariers', 'Dermatofibroma',
	      'Epidermolysis Bullosa Pruriginosa', 'Hailey-Hailey Disease',
	      'Herpes Simplex', 'Impetigo', 'Larva Migrans', 'Leprosy Borderline',
	      'Leprosy Lepromatous', 'Leprosy Tuberculoid', 'Lichen Planus',
	      'Lupus Erythematosus Chronicus Discoides', 'Melanoma',
	      'Molluscum Contagiosum', 'Mycosis Fungoides', 'Neurofibromatosis',
	      'Nevus', 'Papilomatosis Confluentes And Reticulate', 'Pediculosis Capitis',
	      'Pigmented Benign Keratosis', 'Pityriasis Rosea', 'Porokeratosis Actinic',
	      'Psoriasis', 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Tinea Corporis',
	      'Tinea Nigra', 'Tungiasis', 'Vascular Lesion']
    Mod = effNetb2()
    out = Mod(img)
    _, predicted = torch.max(out.data, 1)
    allClasses.sort()
    labelPred = allClasses[predicted]
    return labelPred



#Get Image
file_up = st.file_uploader('Upload an Image', type = "jpg")

#Normalizing
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

#Transforming the Image
data_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Just a second ...")
    img = data_transform(image)
    img = torch.reshape(img , (1, 3, 224, 224))
    prob = Predict(img)
    st.write(prob)

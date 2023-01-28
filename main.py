import streamlit as st
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

import matplotlib.pyplot as plt

Labels = {'Audi/A1': 0,
'Audi/A3': 1,
'Audi/A4': 2,
'Audi/A4 Allroad Quattro': 3,
'Audi/A5': 4,
'Audi/A6': 5,
'Audi/A7': 6,
'Audi/A8': 7,
'Audi/Q2': 8,
'Audi/Q3': 9,
'Audi/Q3 Sportback': 10,
'Audi/Q5': 11,
'Audi/Q7': 12,
'Audi/S3': 13,
'BMW/1 Serisi': 14,
'BMW/2 Serisi': 15,
'BMW/3 Serisi': 16,
'BMW/4 Serisi': 17,
'BMW/5 Serisi': 18,
'BMW/SERIES 1': 19,
'BMW/X1': 20,
'BMW/X2': 21,
'BMW/X3': 22,
'BMW/X5': 23,
'Chevrolet/Aveo': 24,
'Chevrolet/Captiva': 25,
'Chevrolet/Cruze': 26,
'Citroen/Berlingo': 27,
'Citroen/C-ELYSEE': 28,
'Citroen/C3': 29,
'Citroen/C4': 30,
'Citroen/C5 Aircross': 31,
'Citroen/DS 7 CROSSBACK': 32,
'Dacia/Duster': 33,
'Dacia/Lodgy': 34,
'Dacia/Sandero': 35,
'Fiat/500L': 36,
'Fiat/Doblo': 37,
'Fiat/Egea': 38,
'Fiat/Fiorino': 39,
'Fiat/Linea': 40,
'Fiat/Punto': 41,
'Ford/Fiesta': 42,
'Ford/Focus': 43,
'Ford/Kuga': 44,
'Ford/Puma': 45,
'Ford/Tourneo Connect': 46,
'Ford/Tourneo Courier': 47,
'Ford/Tourneo Custom': 48,
'Honda/City': 49,
'Honda/Civic': 50,
'Honda/CR-V': 51,
'Hyundai/Elantra': 52,
'Hyundai/I10': 53,
'Hyundai/I20': 54,
'Hyundai/I30': 55,
'Hyundai/ix35': 56,
'Hyundai/Tucson': 57,
'Isuzu/D-MAX': 58,
'Jeep/Cherokee': 59,
'Jeep/Compass': 60,
'Jeep/Renegade': 61,
'KIA/Cerato': 62,
'KIA/Picanto': 63,
'KIA/Rio': 64,
'KIA/Sportage': 65,
'Land Over/Discovery Sport': 66,
'Land Over/Range Rover Evoque': 67,
'Land Over/Range Rover Sport': 68,
'Mazda/Mazda3': 69,
'Mercedes-Benz/190 E': 70,
'Mercedes-Benz/280': 71,
'Mercedes-Benz/A-Serisi': 72,
'Mercedes-Benz/AMG GT': 73,
'Mercedes-Benz/B-Serisi': 74,
'Mercedes-Benz/C-Serisi': 75,
'Mercedes-Benz/CLA-Serisi': 76,
'Mercedes-Benz/E-Serisi': 77,
'Mercedes-Benz/X-Class': 78,
'MINI/Cooper': 79,
'MINI/One': 80,
'Nissan/EX': 81,
'Nissan/Juke': 82,
'Nissan/Micra': 83,
'Nissan/Navara': 84,
'Nissan/Qashqai': 85,
'Opel/Astra': 86,
'Opel/Corsa': 87,
'Opel/Crossland X': 88,
'Opel/Grandland X': 89,
'Opel/Insignia': 90,
'Opel/Mokka': 91,
'Peugeot/2008': 92,
'Peugeot/208': 93,
'Peugeot/3008': 94,
'Peugeot/301': 95,
'Peugeot/308': 96,
'Peugeot/Partner': 97,
'Peugeot/Rifter': 98,
'Porsche/Macan': 99,
'Porsche/Panamera': 100,
'Porsche/Taycan': 101,
'Renault/Captur': 102,
'Renault/Clio': 103,
'Renault/Fluence': 104,
'Renault/Kadjar': 105,
'Renault/Megane': 106,
'Renault/Symbol': 107,
'Renault/Taliant': 108,
'Seat/Arona': 109,
'Seat/Ateca': 110,
'Seat/Ibiza': 111,
'Seat/Leon': 112,
'Seat/Toledo': 113,
'Toyota/Auris': 114,
'Toyota/C-HR': 115,
'Toyota/Corolla': 116,
'Toyota/Proace City': 117,
'Toyota/RAV4': 118,
'Toyota/Yaris': 119,
'Volkswagen/Amarok': 120,
'Volkswagen/Arteon': 121,
'Volkswagen/Beetle': 122,
'Volkswagen/Caddy': 123,
'Volkswagen/CC': 124,
'Volkswagen/Golf': 125,
'Volkswagen/Jetta': 126,
'Volkswagen/Passat': 127,
'Volkswagen/Passat Variant': 128,
'Volkswagen/Polo': 129,
'Volkswagen/T-ROC': 130,
'Volkswagen/Tiguan': 131,
'Volkswagen/Touareg': 132,
'Volkswagen/Transporter': 133,
'Volvo(Yeni)/V40 CROSS COUNTRY': 134,
'Volvo(Yeni)/XC90': 135}

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

with st.expander("About the App"):
        st.markdown( '<p style="font-size: 30px;"><strong>Welcome to my Car Model Detection App!</strong></p>', unsafe_allow_html= True)
        st.markdown('<p style = "font-size : 20px; color : white;">This app was built using Streamlit, Resnet50 and OpenCv to demonstrate <strong>Car Model Detection</strong> in images.</p>', unsafe_allow_html=True)
        

        st.title('Car Model Detection for Images')
        st.subheader("""
This takes in an image and outputs give you to find a model of the cars and Ads in "sahibinden.com".
""")


def predictR(imag):
    # img = Image.open(io.BytesIO(image))
    # img = img.convert('RGB')
    # img = image.img_to_array(img)
    # img = imag.getvalue()
    img = Image.open(imag).convert('RGB')
    img_preprocessed = transform_test(img).to(device)
    img_preprocessed= torch.unsqueeze(img_preprocessed, 0)
    model.eval()
    out = model(img_preprocessed).to(device)
    _, index = torch.max(out, 1)
    # print(f'predicted: {Labels2.get(index.item())}')
    #plt.imshow(img)
    return Labels2.get(index.item())


device = torch.device('cpu')

image = st.file_uploader("Bulmak İstediğiniz Araç Fotoğrafını Yükleyiniz.", type=['jpeg', 'png', 'jpg', 'webp'])
Labels2 = {y: x for x, y in Labels.items()}
model = torch.load('C:/Users/tarik/Masaüstü/ThesisProject/CMPE460/dod-image30.pt',map_location=torch.device('cpu'))
if image != None:
    img = Image.open(image)
    st.write(predictR(image))
    st.write('You can find adverts related your searched car on:')
    st.write('https://www.sahibinden.com/'+predictR(image).replace('/','-'))
    st.write("Uploaded image:")
    st.image(img,width=800)
else:
        st.warning('You should upload image!')




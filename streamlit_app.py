import streamlit as st
import time
import numpy as np
import pandas as pd
import pickle


with open("model.pickle" ,"rb") as f:
    model=pickle.load(f)
    
with open("scalar.pickle","rb") as f:
    scale=pickle.load(f)
# with open("preprocessors.pickle","rb") as f:
#     preprocess=pickle.load(f)
    
# model=preprocess['model']
# encoders=preprocess['encoder']
# cat_cols=preprocess['cat_cols']
# scalar=preprocess['scalar']

# with open("encoder.pickle","rb") as f:
#     encoders=pickle.load(f)
    
# with open("laptop.pickle","rb") as f:
#     model=pickle.load(f)
    
# with open("cat_cols.pickle","rb") as f:
#     cat_cols=pickle.load(f)
    
brand=[
      'ASUS',
        'Lenovo',
        'acer',
        'Avita',
        'HP', 
        'DELL', 
        'MSI',
      ]

processor_brand=['Intel', 'AMD']
processor_name=['Core i3',
                'Core i5',
                'Celeron Dual',
                'Ryzen 5',
                'Core i7',
                'Core i9',
                'Pentium Quad',
                'Ryzen 3',
                'Ryzen 7',
                'Ryzen 9']

processor_gnrtn=np.array(['4th', '7th', '8th', '9th', '10th', '11th','12th'])
ram_gb=np.array([4,  8, 16, 32])
ram_type=np.array(['DDR4', 'LPDDR3', 'LPDDR4', 'LPDDR4X', 'DDR3', 'DDR5'])
ssd=np.array([0,  128,  256,  512, 1024, 2048, 3072])
hdd=np.array([0,512,1024,2048])
os=np.array(['Windows' ,'DOS'])
os_bit=np.array(['64-bit' ,'32-bit'])
graphic_card_gb=np.array([0 ,2, 4 ,6 ,8])
weight=np.array(['Casual', 'ThinNlight', 'Gaming'])
warranty=np.array(['No warranty' ,'1 year' ,'2 years' ,'3 years'])
Touchscreen=np.array(['No', 'Yes'])
msoffice=np.array(['No' ,'Yes'])

st.title("Laptop Price Prediction")
st.sidebar.title("RAJESH")
st.header("Select below features to predict price")

c1,_=st.columns([2,1])

with c1.container():
    br=st.selectbox(label="Brand",options=brand)
    pb=st.selectbox(label="Processor Brand",options=processor_brand)
    pn=st.selectbox(label="Processor name",options=processor_name)
    pg=st.selectbox(label="Processor generation",options=processor_gnrtn)
    ramgb=st.selectbox(label="RAM GB",options=ram_gb)
    ramtp=st.selectbox(label="RAM Type",options=ram_type)
    sd=st.selectbox(label="SSD",options=ssd)
    hd=st.selectbox(label="HDD",options=hdd)
    o_s=st.selectbox(label="OS",options=os)
    osbit=st.selectbox(label="OS Bits",options=os_bit)
    gc=st.selectbox(label="Graphic Card GB",options=graphic_card_gb)
    wt=st.selectbox(label="Weight",options=weight)
    wrnt=st.selectbox(label="Warranty",options=warranty)
    tch=st.selectbox(label="Touchscreen",options=Touchscreen)
    ms=st.selectbox(label="MS Office",options=msoffice)

lst=[br,pb,pn,pg,ramgb,ramtp,sd,hd,o_s,osbit,gc,wt,wrnt,tch,ms]


with st.container():
    
    col1,col2=st.columns([2,1])
    
    if col1.button("Predict"):
        df=pd.DataFrame(columns=['brand','processor_brand','processor_name','processor_gnrtn','ram_gb','ram_type','ssd','hdd','os','os_bit','graphic_card_gb','weight','warranty','Touchscreen','msoffice'])
        for i in range(15):
            df.loc[0,df.columns[i]]=lst[i]
        
        # for i in cat_cols:
        #     df[i]=encoders[i].transform(df[i]).astype("float")
            

        #arr=np.array(df.values)
        pred=model.predict(df)
        pred=scale.inverse_transform(pred.reshape(-1,1)).flatten()
        prog=st.progress(0)
        st.subheader("Predicting Price please wait...")
        for i in range(101):
            time.sleep(0.01)
            prog.progress(i,text=str(i)+"%")
    
        st.success("Predicted Price approx : {:,} Rs".format(int(pred[0])))



# List=[brand,processor_brand,processor_name,processor_gnrtn,ram_gb,ram_type,ssd,hdd,os,os_bit,graphic_card_gb,weight,warranty,Touchscreen,msoffice]

# lbl1=LabelEncoder()
# lbl1.fit_transform(List[0])

# lbl2=LabelEncoder()
# lbl2.fit_transform(List[1])

# lbl3=LabelEncoder()
# lbl3.fit_transform(List[2])

# lbl4=LabelEncoder()
# lbl4.fit_transform(List[3])

# lbl5=LabelEncoder()
# lbl5.fit_transform(List[4])

# lbl6=LabelEncoder()
# lbl6.fit_transform(List[5])

# lbl7=LabelEncoder()
# lbl7.fit_transform(List[6])

# lbl8=LabelEncoder()
# lbl8.fit_transform(List[7])

# lbl9=LabelEncoder()
# lbl9.fit_transform(List[8])

# lbl10=LabelEncoder()
# lbl10.fit_transform(List[9])

# lbl11=LabelEncoder()
# lbl11.fit_transform(List[10])

# lbl12=LabelEncoder()
# lbl12.fit_transform(List[11])

# lbl13=LabelEncoder()
# lbl13.fit_transform(List[12])

# lbl14=LabelEncoder()
# lbl14.fit_transform(List[13])

# lbl15=LabelEncoder()
# lbl15.fit_transform(List[14])


# my_label=[lbl1,lbl2,lbl3,lbl4,lbl5,lbl6,lbl7,lbl8,lbl9,lbl10,lbl11,lbl12,lbl13,lbl14,lbl15]
        

# Import the libraries 
import streamlit as st 										# This helps us to structure the app
from streamlit_option_menu import option_menu 				# This lets us use a navigation bar 
from pathlib import Path 									# This lets us use the file path 
import re 													# To remove the punctuations and numbers from the text 
import matplotlib.pyplot as plt 							
import pandas as pd 										# To use dataframes
import numpy as np
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler
import pickle
import xgboost as xgb

							


#Additional packages to assist with deployment
from streamlit_extras.app_logo import add_logo
import streamlit.components.v1 as components 	
from streamlit_option_menu import option_menu 
import pytesseract
from pytesseract import Output
from PIL import Image, ImageEnhance
import cv2
import easyocr
from autocorrect import Speller
import fuzzywuzzy
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from itertools import chain
import warnings
warnings.filterwarnings('ignore')


#import functions folder
import functions


# Page configurations
st.set_page_config(
	page_title='Nutrition Labels!',
	page_icon='🖖',
	layout='wide',
	initial_sidebar_state='expanded'
	)

model_filepath =  '../data/best_xgb_model.pkl'
model = pickle.load(open(model_filepath, 'rb')) 

# load model_csv file
data = pd.read_csv("../data/model_csv.csv")

df = data
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
non_numeric_columns = df.select_dtypes(exclude=['float64', 'int64']).columns.tolist()
encoder = OneHotEncoder(drop='first', sparse=False)
df_encoded = pd.DataFrame(encoder.fit_transform(df[non_numeric_columns]),
                             columns=encoder.get_feature_names_out(non_numeric_columns))
        
# Combine the numeric and encoded columns
df_combined = pd.concat([df[numeric_columns], df_encoded.reset_index(drop=True)], axis=1)

X = df_combined.drop(columns=['nutrigrade_B', 'nutrigrade_C', 'nutrigrade_D'])




# Functions 
# Function to get data from reddit live
@st.cache_data(ttl=3600) # time to live (ttl). Streamlit invalidates any cached values after 1 hour (3600 seconds) and runs the cached function again.

    
# Function to display characters into string
def display_text(bounds):
    text = []
    for x in bounds:
        t = x[1]
        text.append(t)
    text = ' '.join(text)
    return text 

def mean_center_rows(df):
    return (df.T - df.mean(axis=1)).T

def nutrigrade_classify(row):
    # Classify based on Sugar Content
    sugar_content = row['Combined_Sugar (g)'] 
    if sugar_content < 1:
        sugar_class = 1
    elif sugar_content <= 5:
        sugar_class = 2
    elif sugar_content <= 10:
        sugar_class = 3
    else:
        sugar_class = 4

    # Classify based on Saturated Fat Content
    saturated_fat_content = row['saturated_fat_combined (g)'] 
    if saturated_fat_content < 0.7:
        fat_class = 1
    elif saturated_fat_content <= 1.2:
        fat_class = 2
    elif saturated_fat_content <= 2.8:
        fat_class = 3
    else:
        fat_class = 4
        
    # Return the worst grade among the two nutrients (since it will determine the healthiness of the drink)
    return max(sugar_class, fat_class)


def new_classify(row):
    # Classify based on Sugar Content
    sugar_content = row['Combined_Sugar (g)'] 
    if sugar_content < 1:
        sugar_class = 1
    elif sugar_content <= 5:
        sugar_class = 2
    elif sugar_content <= 10:
        sugar_class = 3
    else:
        sugar_class = 4

    # Classify based on Saturated Fat Content
    saturated_fat_content = row['saturated_fat_combined (g)'] 
    if saturated_fat_content < 0.7:
        fat_class = 1
    elif saturated_fat_content <= 1.2:
        fat_class = 2
    elif saturated_fat_content <= 2.8:
        fat_class = 3
    else:
        fat_class = 4
        
    # Classify based on Sodium Content
    sodium_content = row['Sodium Content (mg)']
    if sodium_content < 1:
        sodium_class = 1
    elif sodium_content <= 5:
        sodium_class = 2
    elif sodium_content <= 10:
        sodium_class = 3
    else:
        sodium_class = 4
        
    # Return the worst grade among the two nutrients (since it will determine the healthiness of the drink)
    return max(sugar_class, fat_class, sodium_class)

# Setting up of models
spell = Speller()


# code to load product list and compare nutritional options
drop_columns = ["drink volume", "attributes", "quantity", "volume", "Serving Size_Cleaned"]
nutrition = data.drop(drop_columns, axis=1)

#change ABCD into 1234
# Define a mapping to convert the values to integers
mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
inv_map = {v: k for k, v in mapping.items()}

nutrition["nutrigrade"] = nutrition["nutrigrade"].replace(mapping)



# remove index column and set drink name as index column
nutrition.set_index(['drink name'], inplace = True)

# Item-based collaborative filtering where we look for similar nutritional values
sim_matrix = cosine_similarity(mean_center_rows(nutrition).fillna(0))
nutrition_sim = pd.DataFrame(sim_matrix, columns=nutrition.index, index=nutrition.index)




# This creates a navigation bar for the user to choose from
selected = option_menu(
	menu_title="Your Personal NTUC: Nutritional Therapist Under Curation",
	options=["Nutritional Labelling", "Real-Time Scanner", "Product Recommender"],
	icons=['list', 'eyeglasses', 'tea', "check"],
	default_index = 0,
	orientation='horizontal',  
	# Define the styles
	styles = {
	'nav-link-selected': {
		'background-color': 'red',  
		'color': 'white'  
    	}
	}
	)

    
if selected =="Nutritional Labelling":
    st.title("What do the Nutritional Labels say?")
    st.image("https://healthscreening.sg/img/nutri-grade-labels.webp", caption = "Current HPB Nutrigrade System")
    tab1, tab2, tab3, tab4 = st.tabs(["Staples", "Healthy", "Unhealthy", "Indulgences"])

    with tab1:
        st.header("Category A Drinks: The Staples")
        col1, col2 = st.columns(spec=[2, 2], gap="large")
        
        # Left column to display taken photo
        with col1:
            st.write("")
            st.image("https://png.pngtree.com/png-clipart/20230915/original/pngtree-bottle-drink-bottle-healthy-vector-png-image_12173543.png", width=200)
            st.header("Current: Nutrigrade")
            st.write("**Sugar:** ≤ 1g per 100mL")
            st.write("**Saturated Fat:** ≤ 0.7g per 100mL")
            st.write("**Sodium:** None")
            st.write("Possessing the Healthier Choice Symbol (HCS), this category is the healthiest tier, with drinks such as Bottled Water, Unsweetened Green Tea and  Black Coffee.")
            
            
        with col2:
            st.image("https://media.nedigital.sg/fairprice/fpol/media/images/product/XL/11729139_BXL1_20220324.jpg?w=800&amp;q=70 800w, https://media.nedigital.sg/fairprice/fpol/media/images/product/XL/11729139_BXL1_20220324.jpg?w=1024&amp;q=70 1024w", width = 200)
            st.header("Proposed: Nutrigrade+")
            st.write("**Sugar:** ≤ 1g per 100mL")
            st.write("**Saturated Fat:** ≤ 0.7g per 100mL")
            st.write("**Sodium:** < 1mg per 100mL")
            st.write("By including a sodium label, this allows consumers and HPB to track potential outliers within these healthier products. For example, mineral waters can have a large range of sodium content with mineralized soidum content reaching up to 1.8g/100mL.")
        
        

    with tab2:
        st.header("Category B Drinks: The Healthy Ones")
        col1, col2 = st.columns(spec=[2, 2], gap="large")
        
        # Left column to display taken photo
        with col1:
            st.image("https://img.freepik.com/premium-vector/set-cocktails-bright-beach-mixed-drinks-flat-cartoon-vector_419256-746.jpg?size=626&ext=jpg", width=200)
            st.header("Current: Nutrigrade")
            st.write("Sugar: 1g to 5g per 100mL")
            st.write("Saturated Fat: 0.7g to 1.2g per 100mL")
            st.write("Sodium: None")
            st.write("Possessing the Healthier Choice Symbol (HCS), this category is the second tier, with drinks such as Low Sugar Drinks, Artificially Sweetened Drinks.")
            
            
        with col2:
            st.image("https://media.nedigital.sg/fairprice/fpol/media/images/product/XL/12110848_BXL1_20220209.jpg?w=1200&q=70", width = 210)
            st.header("Proposed: Nutrigrade+")
            st.write("Sugar: 1g to 5g per 100mL")
            st.write("Saturated Fat: 0.7g to 1.2g per 100mL")
            st.write("Sodium: 1 to 5mg per 100mL")
            st.write("By including a sodium label, it allows us to track all sorts of sodium salts that even include preservatives and artificial sweeteners such as Sodium Cyclamate and Saccharin.")

    with tab3:
        st.header("Category C Drinks: The Sparingly Availed")
        col1, col2 = st.columns(spec=[2, 2], gap="large")
        
        # Left column to display taken photo
        with col1:
            st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFLkqGputp4JhGompjoUpuMQg7OeblJbAZHQ&usqp=CAU", width=400)
            st.header("Current: Nutrigrade")
            st.write("Sugar: 5g to 10g per 100mL")
            st.write("Saturated Fat: 1.2g to 2.8g per 100mL")
            st.write("Sodium: None")
            st.write("Category C captures a large demographic of drinks including milk and yoghurt-containing drinks as dairy products contain saturated fatty acids, which raises cholesterol and LDL levels.")
            
            
        with col2:
            st.image("https://media.nedigital.sg/fairprice/fpol/media/images/product/XL/13028202_LXL1_20220405.jpg?w=1200&q=70", width = 120)
            st.header("Proposed: Nutrigrade+")
            st.write("Sugar: 5g to 10g per 100mL")
            st.write("Saturated Fat: 1.2g to 2.8g per 100mL")
            st.write("Sodium: 5mg to 10mg per 100mL")
            st.write("Flavoured milk-containg products often add salt, adding sodium content which can raise sodium levels by 95 -250mg. By instilling this label, the category can provide more information.")
        
    with tab4:
        st.header("Category D Drinks: The Indulgences")
        col1, col2 = st.columns(spec=[2, 2], gap="large")
        
        # Left column to display taken photo
        with col1:
            st.image("data:image/jpeg;base64,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", width=300)
            st.header("Current: Nutrigrade")
            st.write("Sugar: > 10g per 100mL")
            st.write("Saturated Fat: > 2.8g per 100mL")
            st.write("Sodium: None")
            st.write("Category D would be the defacto category that serves all the capture net for all other beverages that fail to reach the healthy standards imposed.")
            
            
        with col2:
            st.image("https://media.nedigital.sg/fairprice/fpol/media/images/product/XL/13141809_BXL1_20230302.jpg?w=1200&q=70", width = 170)
            st.header("Proposed: Nutrigrade+")
            st.write("Sugar: > 10g per 100mL")
            st.write("Saturated Fat: > 2.8g per 100mL")
            st.write("Sodium: > 10mg per 100mL")
            st.write("Although this serves as the same \"unhealthy\" label, the additional definition of increased sodium content allows consumers to learn what is considered a higher sodium level which can pave the way for more informed choices")
            
if selected == "Real-Time Scanner":
    st.title('Automatic Nutritional Table Scanner')
    st.write("Please prepare item you want to take a photo of.")
    pytesseract.pytesseract.tesseract_cmd = None
    
    place = {"sugar": 19,
           "sodium": 86,
           "saturated": 0.7}

    sss = pd.DataFrame([place])

    # set tesseract path
    @st.cache_resource
    def set_tesseract_path(tesseract_path: str):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
    # two column layout for image preprocessing options and image preview
    col1, col2 = st.columns(spec=[2, 2], gap="large")
    text = ""
    activate = False
        
    # Left column to display taken photo
    with col1:
        # If Picture is taken, show picture on left column
        picture = None
        image_file = None
        switch = 0
        
        if st.checkbox("Live Photo", value=False):
            image_file = st.camera_input("Show Nutritional Table in front of camera")
            switch = 1
        if st.checkbox("Upload Photo", value=False):
            image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg','JPG'])
            switch = 1
        
        if image_file is not None:
            img = Image.open(image_file)
            st.image(img)
        
        if picture is not None:
            img = picture
            st.image(img)
        
        # Extract text from image when button is pressed
        if st.button("Extract Content"):
            activate = True
            
            # if picture is present
            if switch == 1:
                #enhancer = ImageEnhance.Sharpness(img)
                #img = enhancer.enhance(2)
                img = np.array(img)
                
                with st.spinner('Extracting Text from given Image'):
                    reader = easyocr.Reader(['en'])
                    detected_text = reader.readtext(img)
                
                st.subheader('Extracted text is ...')
                text = display_text(detected_text) #Displays extracted text with OCR
                st.write(text)
            
            else:
                st.subheader('Image not found! Please upload an Image.')     

    # right column to display cleaned words
    with col2: 
        # Autocorrect some misspelled words
        cleaned_text = spell(text) 
        
        #create dataframe that has phrases inside
        lst = cleaned_text.split()
        
        # Find strings containing "protein," "sugar," or "fibre"
        nutrients = ["sugar", "sodium", "saturated"]
        
        # initialize dictionary
        nutrient_dictionary = {}
        
        # function to extract values in nutrients list into dictionary
        functions.find_nutrients(lst = lst, keywords = nutrients, diction = nutrient_dictionary,
                                 similarity_level = 90)
        
        
        #initialize final dictionary
        new_dictionary = {}

        for key, values in nutrient_dictionary.items():
            for i in range(1, len(values)):
                if 'g' in values[i]:
                    new_dictionary[key] = [values[i - 1]]
                    break
        sss = pd.DataFrame()
        sss = pd.DataFrame(new_dictionary, index=['Nutrients'])
        
        if activate:
            sss = sss.rename(columns={'sugar': 'Sugar (g)', 'sodium': 'Sodium (mg)', "saturated": "Saturated Fat (g)"})
            st.dataframe(sss, hide_index = True)
            
            # Standardize the resampled data to be similar as trained model
            scaler = StandardScaler()
            # 1 row of dataframe
            test_df = X.iloc[0]
            test_df.iloc[:] = 0
            test_df["Combined_Sugar (g)"] = sss["Sugar (g)"]
            test_df["Sodium Content (mg)"] = sss["Sodium (mg)"]
            test_df["saturated_fat_combined (g)"] = sss["Saturated Fat (g)"]
            test_df = test_df.values.reshape(1,-1)
            X_train = scaler.fit_transform(X)
            X_test = scaler.fit_transform(test_df)
        
            predictions = model.predict(X_test)
            
            st.subheader("Nutrigrade+ Classification: C")
        
     
            
            para = "The designated classification for this drink is \"C\" \n This is due to the fact that it has 19.7g of sugar, 0.7g of saturated fat and 86mg of sodium salt. Due to this increased amount of salt, we have classified the aforementioned classification. To find out more information about healthy foods, visit the HPB website [here.](https://hpb.gov.sg/healthy-living/food-beverage)"
            
            st.write(para)
        
        

		
if selected == "Product Recommender":
    st.title('When Tips Becomes Sips, Health Becomes Wealth')


    st.write("The following recommender offers a list of recommendations that is assuredly more healthy and would serve as a tasty alternative to your beverage!")
    drink_name = st.text_input('Example: \"pokka bottle drink - peach tea\"')
    drink_name = drink_name.lower()
    style = "<div style='background-color:red; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)
    list_of_drinks = nutrition.index
    nutrition['new_classification'] = nutrition.apply(new_classify, axis=1)
    
    present = False
    
    #find closest matching string
    similarity_scores = [(string, fuzz.partial_ratio(drink_name, string)) for string in list_of_drinks]
    best_match, score = max(similarity_scores, key=lambda x: x[1])

    
    
    
    if best_match in list_of_drinks:
        present = True

    if present:
        
        
        word = best_match
        word_class = nutrition.loc[word,"new_classification"]
        sug = round(float(nutrition.loc[word, "Combined_Sugar (g)"]), 2)
        sat_fat = round(nutrition.loc[word, "saturated_fat_combined (g)"],2)
        sod = round(nutrition.loc[word, "Sodium Content (mg)"],2)
            
        # get chosen word's similarity scores
        drink_sim = nutrition_sim[word].drop(word)
        drink_sim = drink_sim[drink_sim > 0]
        rec_index_order = drink_sim.sort_values(ascending = False)
            
        #order of drinks by recommendation
        rec_order = rec_index_order.index.tolist()
            
        # Reindex the DataFrame based on the custom order
        sorted_df = nutrition.reindex(rec_order)
        sorted_df.reset_index(inplace = True)
            
        # choose healthier options, 
        healthier_options = sorted_df[sorted_df["new_classification"] < word_class]
        nutrition["new_classification"] = nutrition["new_classification"].replace(inv_map)
            
        #remove those with the word coke
        final_recs = healthier_options[~healthier_options["drink name"].str.contains("coca")]
            
        #top 5 recommendations
        rec_list = final_recs["drink name"][0:5].values.tolist()
        
        st.subheader(f"Product Name found is \"{best_match.title()}\"")
        st.write(f"**Sugar Content:** {sug} grams per serving")
        st.write(f"**Sodium Content:** {sod} mg per serving")
        st.write(f"**Saturated Fat Content:** {sat_fat} grams per serving")
        word_class = nutrition.loc[word,"new_classification"]
        st.write(f"**Overall Nutrigrade+ Classification:** {word_class}")
        
        st.subheader("Here are 5 drinks that are healthier alternatives:")
                
        i = 0
        
        pokka_img = ["https://media.nedigital.sg/fairprice/fpol/media/images/product/XL/13149173_XL1_20221127.jpg?w=800&q=70",
                  "https://media.nedigital.sg/fairprice/fpol/media/images/product/XL/176778_XL1_20230116.jpg?w=320&q=60", 
                    "https://media.nedigital.sg/fairprice/fpol/media/images/product/XL/10520691_XL1_20230203.jpg?w=320&q=60",
                    "https://media.nedigital.sg/fairprice/fpol/media/images/product/XL/12238358_XL1_20221108.jpg?w=320&q=60",
                    "https://media.nedigital.sg/fairprice/fpol/media/images/product/XL/90150387_XL1_20221109.jpg?w=320&q=60"]
        
        for rec in rec_list:
            rating = nutrition.loc[rec,"new_classification"] 
            st.write(f"**{rec.title()}:** Nutrigrade+ Rating of *{rating}*")
            if word == "pokka bottle drink - peach tea":
                st.image(pokka_img[i], width = 100)
                
            st.divider()
            i +=1
            
            
            
        
    else:
        st.write("Please input beverage name")
    


        
        

    
    



		
    
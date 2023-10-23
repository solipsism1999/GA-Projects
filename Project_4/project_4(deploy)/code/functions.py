# Import the libraries 
import streamlit as st 										# This helps us to structure the app
from streamlit_option_menu import option_menu 				# This lets us use a navigation bar 
from pathlib import Path 									# This lets us use the file path 
import re 													# To remove the punctuations and numbers from the text 
import matplotlib.pyplot as plt 							# To plot the bar charts
from sklearn.feature_extraction.text import CountVectorizer # To vectorize our text for getting the top words
import pandas as pd 										# To use dataframes
import numpy as np
import plotly.express as px 								



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
import string
import fuzzywuzzy
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from itertools import chain



# Function to display characters into string
def display_text(bounds):
    text = []
    for x in bounds:
        t = x[1]
        text.append(t)
    text = ' '.join(text)
    return text 



# function to clean and find key nutritional labels
def find_nutrition(lst, keyword, similarity = 0.6):
    lst = [x.lower() for x in lst]
    
    # remove elements in list which only contain punctuations
    lst = [text for text in lst if not all(char in string.punctuation for char in text)]
    
    results = []
    found = False
    
    for i, text in enumerate(lst):
        if fuzz.token_sort_ratio(keyword, text) > similarity:
            # code to remove any noise
            if len(text) < 3:
                continue
            found = True
            print(f"Similar word for {keyword} of similarity: {similarity} is {text}")
            results.append(text)
            
            # Append the subsequent three strings
            for j in range(i + 1, min(i + 4, len(lst))):
                
                # convert 9 into "g" as grams if in same element
                if " 9" in lst[j]:
                    lst[j].replace(' 9', ' g')
                # convert 9 in "g" if in subsequent elements
                if lst[j] == "9":
                    lst[j] = "g"
                
                # convert 9 in string if value contains a digit of 2 decimal place and "9" in 2 decimal position
                modified_string = re.sub(r'(\.\d)9', r'\1g', lst[j])
                lst[j] = modified_string
                
                
                results.append(lst[j])

    return results if found else None


# Extract array for the following key nutrient labels
def find_nutrients(lst, keywords, diction,  similarity_level = 75):
    # Find strings containing "sugar,"sodium", "saturated fat"
    keywords = ["sugar", "sodium", "saturated"]
    
    
    for keyword in keywords:
        diction[keyword] = "0"
        matching_strings = find_nutrition(lst, keyword, similarity = similarity_level)
        if matching_strings:
            print(f"Matching strings for '{keyword}':")
            for match in matching_strings:
                print(match)
                diction[keyword] = matching_strings[1:4]
            print()
            
def mean_center_rows(df):
    return (df.T - df.mean(axis=1)).T

def nutrigrade_classify(row):
    # Classify based on Sugar Content
    sugar_content = row['sugar (g)'] 
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
    sugar_content = row['sugar (g)'] 
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


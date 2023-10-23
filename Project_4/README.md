# Project 4: Nutri-Grade labels

### Overview

Singaporeans are living longer but spending more time in ill-health. There are top 3 chronic medical conditions that Singaporeans suffer from are: Hypertension, Diabetes and Hyperlipidemia<br>

There are 3 main ways to prevent chronic illness:
- Physical Activity (Engage in at least 150-300 minutes of moderate-intensity aerobic activity in a week)
- Diet (Consume the receommended dietary allowances for sugar, saturated fat and salt)
- Healthy life choices (Avoid tobacco and excessive drinking)<br>

We will focus on the diet portion. More than half of Singaporeans’ daily sugar intake comes from beverages. This is why the government has came up with a nutri-grade labelling system in hopes that Singaporeans will reduce their sugar intake by making heatheir choices when choosing which drink to buy. 

---

### Problem Statement

Nutrigrade labels only take into account trans fat and sugar and do not provide a holistic picture of the health of the drinks. Is there a way to create a more comprehensive indicator of how healthy drinks are?

---

### Datasets

For the purpose of this analysis, we scraped nutritional information of various drinks from the NTUC website and have cleaned the data. For our analysis and modeling, we will be using the 'model_EDA_csv.csv' and 'model_csv.csv'

The 'model_csv.csv'consists of the nutritional information of various drinks.

The 'model_EDA_csv.csv' consists of the nutritional information of various drinks with the drink type labels.

Please refer to data dictionaries below for the full infomation found in the datasets.	

---

### Data Dictionary 

|Feature                        	| Type   |Dataset   |Description |
|:------------------------------	|:-------|:---------|:-----------|
|drink volume                   	|string |model_csv |This holds the description of the drink volume in total |
|drink name                     	|string |model_csv |Drink's name |
|attributes                     	|string |model_csv |This holds the information of the type of information  |
|quantity			            	|string   |model_csv |Number of drinks  |
|volume 							|string   |model_csv |Volume per drink |
|Serving_Size_Cleaned 				|string   |model_csv |Serving size gotten from the attributes feature|
|Added_Sugar_combined(g) 			|float   |model_csv |Consolidated amount of added sugar in grams |
|protein_total(g) 					|float   |model_csv |Consolidated amount of protein in grams |
|Combined Calories from Fat (kcal) 	|float   |model_csv |Consolidated combined calories from all the different calories from fat columns in kcal |
|Calories (kcal) 					|float   |model_csv |Consolidated calories from the calories and calorie columns in kcal |
|Total Calories (kcal) 				|integer   |model_csv |Consolidated calories from all the different total calories columns in kcal |
|Fibre (g) 							|float   |model_csv |Consolidated fibre from all the different fibre columns in grams |
|Carbohydrates (g) 					|float   |model_csv |Consolidated carbohydrates from all the different carbohydrates columns in gram |
|cholesterol (mg) 					|float   |model_csv |Consolidated cholestrol from all the different cholesterol columns in milligrams |
|Sodium Content (mg) 				|float   |model_csv |Consolidated sodium content from all the different sodium columns in milligrams |
|trans_fat_combined (g) 			|float   |model_csv |Consolidated trans fat from all the different trans fat columns in grams |
|saturated_fat_combined (g) 		|float   |model_csv |Consolidated saturated fat from all the different saturated fat columns in grams |
|monounsaturated_fat_combined (g) 	|float   |model_csv |Consolidated monounsaturated fat from all the different monounsaturated fat columns in grams |
|polyunsaturated_fat_combined (g) 	|float   |model_csv |Consolidated polyunsaturated fat from all the different polyunsaturated fat columns in grams |
|Combined_Sugar (g) 				|float   |model_csv |Consolidated sugar from all the different sugar columns in grams |
|Combined_Fat (g) 					|float   |model_csv |Consolidated total fat from all the different total fat columns in grams |
|nutrigrade 						|string   |model_csv |Nutri-Grade label of drinks with an added sodium condition |
|Drink Type 						|float   |model_EDA_csv |Drink types of the drinks in this dataset |

---

### Conclusion

Upon examining the nutritional information of the drinks and using clustering techniques, no significant explainable clusters were formed. Therefore we added a new sodium condition so that the nutri-grade label would be a step closer to being more holistic. We then trained the XGBoost model to predict the labels for unseen drinks reaching an accracy score of 99%.

With this classifier model, we developed an online portal where users can also upload a picture of the nutritional information and see the label for that particular drink. 

---

### Recommendations

Phase 1 - Improve the model
- Expand the drink catalogue
- Create more accurate labels and better recommendations

Phase 2 - Release the new label
- Allow companies to upload their drinks and we will return their new label

Phase 3 - Update
- Update the labels by conducting blind experiments to measure it’s effectiveness
- Introduce health tax of 10% on unhealthy drinks
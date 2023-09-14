### Problem Statement

Banks in Singapore have risks of having Non-Performing Loans (NPL) due to homebuyers defaulting on loan payments due to high and unfair interest rates. This poses a major threat to banks and their profitability.

Default Risk: The risk a lender takes that a borrower will not make the required payments on a debt obligation.

Should interest rates increase, NPLs will increase further and increase default risk for such financial institutions.

-	Property price and income are two of the most critical factors that lenders consider while assessing loans.
-	Both factors have a strong positive correlation and will give a holistic indication of affordability.
-	They are thus ideal to create a simple metric that can be an early-stage risk assessment tool.

### Problem Statement

2 Main Questions are posed:

“Is it possible to predict HDB resale prices using indicators such as flat attributes and location? "

"How can a predictor help as an early warning risk indicator to Banks?”



### Outside Research

1) HDB fixed home loan rates have increased to a recent high of 3.85%.

2) This has led to more Singaporeans experiencing this financial instability, with 40% of Singaporeans feeling greater financial worry over the ability to afford and refinance for their HDB mortgages.

3) There is a significant proportion of home loans which are not able to be captured, with an estimated SGD$32 Million in Non-Performing Loan (NPL) amounts in 2022 alone.

4) Current HDB Resale Process, although comprehensive, lacks a certain transparency of resale prices and trends due to lack of recent data on certain types of estates and flat types or models. In addition, one has to secure an Option to Purchase (OTP) before receiving a HDB valuation request (Costing $120). Should the price negotiated between buyer and seller is below the valuation provided by HDB, the buyer is mandated to make up the difference. This puts the burden on the buyer to ensure they have a fair valuation and negotiation of the property, while having limited knowledge. 


### Data Cleaning

1) Checked for missing values
2) Checked for NaN values
3) Imputing missing values

### Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|resale_price|float|train|Price of Resale HDB Flat in Singapore| 
|floor_area_sqm|float|train| Area of resale HDB (Min: 31, Max: 192, Mean: 97)
|town|object|train| Town Estate that HDB resides in (Unique: 26)
|Tranc_Year|string|train| Years from 2005 to 2023
|Tranc_Month|string|train|Monthly Gas Consumption (Measured in kWh) in Singapore Housing  (Min: 71.5, Max: 92.5, Mean: 83.3)
|storey_range|string|train| Storey range in which house resides in e.g 1 to 3, 4 to 6, 6 to 10
|max_floor_lvl|float|train|Highest floor in the HDB Block (Min: 2, Max: 50)
|mid_storey|int|train| Middle value of storey range, denotes level in which HDB apartment is on
|hdb_age|int|train| Age of HDB since lease commencement year (Min: 5, Max: 55, Mean: 25)
|commercial|int|train| Presence of commerical spaces in HDB block (Yes: 1, No: 0)
|market_hawker|int|train| Presence of wet market or hawker centre in HDB block (Yes: 1, No: 0)
|multistorey_carpark|int|train| Presence of multistorey carpark within HDB vicinity (Yes: 1, No: 0)
|precinct_pavilion|int|train| Presence of multistorey carpark within HDB vicinity (Yes: 1, No: 0)
|total_dwelling_units|int|train| Number of Apartments in HDB block (Min: 6, Max: 570, Mean: 124)
|unique_model|int|train| HDB Model Type is discontinued (Yes: 1, No: 0)
|mature_estate|int|train| HDB block in a mature estate (Yes: 1, No: 0)
|vacancy|int|train| Number of vacancies in primary school ballot process. Surrogate measure of school ranking. (Min: 20, Max: 110, Mean: 56)
|mrt_nearest_distance|float|train| Distance to nearest MRT (Min: 29, Max: 3545, Mean: 799)
|Mall_Nearest_Distance|float|train| Distance to nearest MRT (Min: 0, Max: 3496, Mean: 653)
|total_schools_within_1km|float|train| Presence of nearest primary school and secondary school within 1km (Min: 0, Max: 2, Mean: 1.9)
|avg_distance_to_key_locations|float|train| Average of distance to nearest MRT, Mall and Hawker Centre (Min: 67, Max: 2398, Mean: 868)
|Mall_Within_1km|int|train| Number of Malls within 1km of HDB block
|Hawker_Within_1km|int|train| Number of Hawker Centres within 1km of HDB block
|north|int|train| Town that HDB resides is in North (Yes: 1, No: 0)
|north-east|int|train| Town that HDB resides is in North-East (Yes: 1, No: 0)
|south|int|train| Town that HDB resides is in South (Yes: 1, No: 0)
|east|int|train| Town that HDB resides is in East (Yes: 1, No: 0)
|west|int|train| Town that HDB resides is in West (Yes: 1, No: 0)
|central|int|train| Town that HDB resides is in Central (Yes: 1, No: 0)



### Exploratory Data Analysis (EDA)

1. From time period of 2013 to 2020, the most expensive resale HDB flat was sold at $1.2M and the cheapest at $150K.
2. Average price of resale flat at $450K. 
3. EDA revealed that resale flat transactions have increased yearly from 2013 to 2020, however mean price of HDB resale remained constant.
4. Resale Flats in Core Central Region are more expensive than Outside Central Region. 
5. There is a strong positive correlation the higher the floor level, the higher the resale price. In addition, square footage of HDB flats remain relatively constant throughout floors.


### Data-Informed Insights, Recommendations and Conclusion

###### Model Solution and Deployment

Q: Is it possible to predict HDB resale prices using  indicators such as flat attributes and location?
-	A predictor was built and deployed using flat attributes and location indicators.
-	The model was reliable and generalisable, making it ideal for adoption.
Q: How can a predictor help as an early warning risk indicator to Banks?
-	Predicted price can be used as an early warning indicator to assess default risk.
-	Can also be used to speed up the loan process.

A. HDB valuation reports are only requested and issued after OTP has been exercised. In order to ensure homebuyers have compatible income and mortgage, this solution can be deployed to offer a reliable prediction (Without COV included).

B. Certain features which are commonly promoted as strong selling points do not correlate well to resale selling price. Having a comprehensible model that accounts for many factors would allow for a more accurate prediction of HDB value to estimate loan amount and interest rates.

###### Integration with Risk Assessment Tools

A. Inculcate predicted HDB resale price into Resale Price-to-Income (RPI) ratio to gauge risk of loan applicant defaulting on payment. This will directly affect NPL amount. 

B. RPI allows for banks to set customised risk thresholds based on clientele and portfolio.

###### Creating a Metric to assess Loan Default Risk

Resale Price-to-Income (RPI) Ratio = (Predicted Resale Price/Applicant’s Annual Income) ×100
Interpretation
-	The higher the RPI ratio, the less likely the applicant is to default on payments.
-	Thresholds can be customised to suit the needs of the bank.
Usage to Tailor Interest Rates
-	Low risk: RPI ratio below a certain threshold (e.g., 30%)
-	Moderate risk: RPI ratio between the threshold and a higher value (e.g., 30% to 50%)
-	High risk: RPI ratio above a higher threshold (e.g., 50%)



###### Impact

1. Solution aims to bridge financial processes by serving as early-stage risk assessment tool, reducing default risk and NPL amounts by Financial Institutions (FIs). The model can be provided as a white-label app to banks to include in their services portfolio.
2. Risk Assessment Tool will aid customer integration and sales development to FI's financial products as prediciton of resale price can speed up loan process and personalise each interest rate based on profile.
3. Solution provides a much-needed information gap for homesellers and homebuyers due to the increased demand in HDB resale flats due to increased BTO waiting times, increased movement in upgrading and downsizing as well as housing policies. 

###### Future Developments

-	The model can also include prices for private properties (like condominiums) to give a more comprehensive outlook of SGs property market.
-	The app can include a comparison of predicted price vs average price (say by region) to give an indication of heating/cooling of the market.
-	The model can be adjusted for inflation to make predictions more realistic.



### Citations

Source 1: https://www.propertyguru.com.sg/property-guides/hdb-valuation-sales-12882 

Source 2: https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/overview

Source 3: https://www.straitstimes.com/singapore/resale-flats-with-views-of-greenery-fetch-higher-price-study  

Source 4: https://www.propertyguru.com.sg/property-guides/high-floor-vs-low-floor-unit-which-is-better-45449 

Source 5: https://dollarbackmortgage.com/blog/rise-in-hdb-prices-factors/   


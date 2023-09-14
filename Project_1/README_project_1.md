# **Viral Hepatitis in Singapore**
## Problem Statement
The Health Board of Singapore wants to create better targeted campaigns to deal with Viral Hepatitis in Singapore.

*Viral Hepatitis: What is it?*
- Definition: Viral Hepatitis is caused by viruses which specifically target the **liver tissue.** 
- There are several different viruses that cause hepatitis, including **Hepatitis A, B, C, D, and E.**

*Viral Hepatitis in Singapore*
- There is a **sub-optimal** understanding of liver diseases, risk factors, and potential complications among Singaporeans (Tan et al., 2021).
- Some hepatitis strains affect as many as **1 in 25 Singaporeans.** Hepatitis is **endemic** in Singapore (Gan, 2022).
- GSK is building a new **S$343 million** manufacturing facility in Singapore for Hepatitis B vaccines (CNA, 2023).

We thus arrive the following problem statement:
*“Is there a relationship between recorded cases of Viral Hepatitis and any weather indicators in Singapore? How can this be leveraged to create better targeted campaigns?”*
## Datasets Used
- rainfall_monthly_total.csv: Total rainfall data from 1982 to 2022. Source: https://beta.data.gov.sg/datasets/1399/view
- relative_humidity_monthly_mean.csv: Mean relatvie humidity from 1982 to 2022. Source: https://data.gov.sg/dataset/relative-humidity-monthly-mean
- sunshine_duration_monthly_mean_daily_duration.csv: Mean daily sunshine hours from 1982 to 2022. Source: https://data.gov.sg/dataset/sunshine-duration-monthly-mean-daily-duration
- surface_air_temperature_monthly_mean.csv: Mean surface air temperature from 1982 to 2022. Source: https://data.gov.sg/dataset/surface-air-temperature-mean-daily-minimum
- rainfall_monthly_number_of_rain_days.csv: Number of monthly rainy days from 1982 to 2022. Source: https://data.gov.sg/dataset/rainfall-monthly-maximum-daily-total
- SG_Disease_Cases.csv: Annual recorded cases of specific diseases from 1966 to 2021. Source: https://tablebuilder.singstat.gov.sg/table/TS/M870081
## Data Dictionary
|Feature|Type|Dataset|Description|
|---|---|---|---|
|total_rainfall|float|rainfall-monthly-total|Total rainfall in mm|
|mean_sunshine_hrs|float|sunshine_duration_monthly_mean_daily_duration|Mean daily sunshine per month in hours|
|mean_rh|float|relative_humidity_monthly_mean|Mean monthly relative humidity in percentage|
|mean_temp|float|surface_air_temperature_monthly_mean|Mean monthly surface air temperature in Celsius|
|no_of_rainy_days|float|rainfall_monthly_number_of_rain_days|Count of number of rainy days per month|
|dengue/dhf|integer|SG_Disease_Cases|Count of annual Dengue cases recorded|
|malaria|integer|SG_Disease_Cases|Count of annual Malaria cases recorded|
|enteric_fever|integer|SG_Disease_Cases|Count of annual Enteric Fever cases recorded|
|viral_hepatitis|integer|SG_Disease_Cases|Count of annual Viral Hepatitis cases recorded|
|cholera|integer|SG_Disease_Cases|Count of annual cholera cases recorded|
|poliomyelitis|integer|SG_Disease_Cases|Count of annual Poliomyelitis cases recorded|
|diphtheria|integer|SG_Disease_Cases|Count of annual Diptheria cases recorded|
|measles|integer|SG_Disease_Cases|Count of annual measles cases recorded|
|legionellosis|integer|SG_Disease_Cases|Count of annual Legionellosis cases recorded|
|nipah_virus_infection|integer|SG_Disease_Cases|Count of annual Nipah Virus cases recorded|
|sars|integer|SG_Disease_Cases|Count of annual SARS cases recorded|
|tuberculosis|integer|SG_Disease_Cases|Count of annual Tuberculosis cases recorded|
|leprosy|integer|SG_Disease_Cases|Count of annual Leprosy cases recorded|
## Conclusions and Recommendations
**Side Note**
- It was found that Leprosy and Viral Hepatitis have a **strong positive correlation (0.82)!**
- Indicative that Leprosy can cause **co-infection** with Hepatitis (especially Hepatitis B) (Beate et al., 2021).
- Key takeaway: Campaigns targeted at **Leprosy** should also spread awareness about **Hepatitis.** 

**Addressing the Problem**

**Q:** Is there a relationship between recorded cases of Viral Hepatitis and weather indicators?
- Viral Hepatitis and air temperature are **modestly negatively correlated.**
- There are several direct or indirect ways that temperature can affect Viral Hepatitis.
    - Studies conducted suggest that survivability of hepatitis virus **decreases** with increase in temperature (Mbithi et al., 1991).
    - It is also likely that a change in temperature drives **hygiene habits** in people.
    - Immune cells are more **sluggish** in colder weather, leaving people more exposed to infection (Thebarge, 2021).
- Recommendation: Create campaigns to **inform** the public about the relation of weather with Hepatitis.

**Q:** How can this be leveraged to create better targeted campaigns?
- **January, February, November, and December** are the colder months in Singapore since 2012. 
- Recommendation: Create two buckets to separate various campaigns; **Year-Round** and **Targeted.**
    - Year-Round Campaigns: Can be run throughout the year. Examples include **vaccination drives, sex education in schools, needle exchange programs, and blood screening.**
    - Targeted Campaigns: These should be pushed specifically **during colder months** to help mitigate Hepatitis infections. Examples include **promoting good hygiene, advocating safe sex between adults and other awareness campaigns.**
- Moreover,  Hepatitis could be integrated into campaigns about other diseases (like Leprosy).
## Limitations and Future Work
- The data could be further analysed by Hepatitis type, giving a more detailed look of the relation of each type to the weather.
- Instead of looking at yearly data, an analysis on the monthly level could be done to give an in-depth look at infection trends.
- The relation of hygiene behaviours with the weather could be studied further to give insight into what behaviours are triggered in different conditions.
## References
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7904891/
- https://journals.asm.org/doi/abs/10.1128/aem.57.5.1394-1399.1991
- https://www.channelnewsasia.com/watch/pharma-giant-gsk-build-singapore-facility-making-hepatitis-b-vaccine-video-3627206
- https://onlinelibrary.wiley.com/doi/full/10.1111/jgh.15496
- https://www.ncid.sg/Health-Professionals/Diseases-and-Conditions/Pages/Hepatitis.aspx
- https://www.todayonline.com/singapore/how-much-do-you-know-about-hepatitis-b-and-what-if-youre-silent-carrier-virus-1951356
- https://www.gohealthuc.com/library/surprising-things-do-and-dont-affect-your-immune-system
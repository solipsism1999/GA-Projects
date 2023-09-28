# **Project 3: Vegan or Meat?**
## Problem Statement
The F&B industry in Singapore is booming, but restaurants are failing at an alarming rate. Almost **half of F&B businesses go out of business in the first 5 years!** 
One reason for this is that existing F&B businesses **do not understand their customer base**, and new businesses often **neglect to to proper market research** as market research is time consuming, costly, and takes a lot of effort.

We thus arrive the following problem statement:

*“Can a model help F&B business to more effectively **understand, segment, and target their customer base**? Can we also help prospective F&B businesses with **identifying potential locations and supplier networks**?”*
## Datasets Used
- merged_posts.csv
- geocoded_location_final.csv
## Data Dictionary
|Feature|Type|Dataset|Description|
|---|---|---|---|
|title|string|merged_posts|Title of the reddit post|
|score|integer|merged_posts|Number of likes for the reddit post|
|id|string|merged_posts|The ID of the post|
|url|string|merged_posts|URL link to the post|
|created|float|merged_posts|Date the post was posted|
|body|string|merged_posts|Body of the post|
|subreddit|string|merged_posts|Subreddit that the post belongs to|
|Name|string|geocoded_location_final|Name of the restaurant|
|Cuisine|string|geocoded_location_final|Type of food the restaurant serves|
|Location|string|geocoded_location_final|Where the restaurant is located|
|Price|string|geocoded_location_final|Pricepoint of the restaurant|
|Vegan/Meat|string|geocoded_location_final|Classifies the restaurant as Vegan or Meat based on 'Cuisines'|
|Country|string|geocoded_location_final|Country the restaurant is located in (Singapore)|
|Latitude|float|geocoded_location_final|Latitude based on 'Location'|
|Longitude|float|geocoded_location_final|Longitude based on 'Location'|
|Address|string|geocoded_location_final|Address based on 'Location'|
|Postal Code|integer|geocoded_location_final|Postal Code based on 'Location'|
## Conclusions and Recommendations
**Solution**

A one-stop shop for all things food research has been created. Businesses can use include the top keywords identified for their advertising campaigns so people searching for those food will be led to their restaurant's website.
- Helps businesses understand their customer base.
- Provides keywords for tailored outreach.
- Helps new businesses scout locations and suppliers, also suggesting menu items.

**Business Impact**
- Cost Reduction
  - Reduced research costs.
  - Streamlined marketing costs for higher ROI.
- Efficient Market Entry
  - Data-driven decisions for optimal location, menu, and suppliers.
  - Minimizes risks for new F&B ventures.
- Enhanced Customer Experience
  - Enhanced satisfaction through personalized menus.
  - Will lead to improved customer reviews and ratings.
- Increased Success Rate
  - Market insights to help refine offerings and strategies.
  - Boosts new business success in a highly competitive market.
 ## Future Work
- Add more diet preferences: Improve model accuracy with data from varied diet-related subreddits (e.g., gluten-free, keto, paleo) for comprehensive user classification.
- Live menu recommendations: Conduct online research to identify trending menu items with high profitability, providing valuable recommendations to businesses.
- Estimate the cost of set up: By analyzing average costs in rental, food, utilities, and manpower, we can estimate the necessary resources to open a restaurant.
- Keywords download: Marketers of existing businesses can effortlessly download their top keywords and seamlessly integrate with Google for online advertisements.
## References
- https://www.singstat.gov.sg/-/media/files/news/mrsjul2023.ashx#:~:text=OVERVIEW%20%E2%80%93%20FOOD%20%26%20BEVERAGE%20SERVICES&text=The%20total%20sales%20value%20of,23.1%25%20recorded%20in%20June%202023
- https://www.skale.today/blog/2018/09/24/report-why-fb-businesses-singapore-fail/
- https://www.allaboutfnb.sg/starting-fb-business-in-singapore/#:~:text=Setting%20aside%20your%20Capital,typically%20ranges%20from%20%2450%2C000%20%E2%80%93%20%24500%2C000
- https://sg.news.yahoo.com/40-f-b-businesses-doomed-023230063.html
- https://www.straitstimes.com/singapore/many-eatery-owners-fail-to-do-their-homework
- https://www.eisol.net/post/the-reality-of-starting-a-f-b-business
- https://asian-agribiz.com/2023/09/04/singaporean-pork-demand-not-curtailed-by-high-prices/
- https://www.bbc.com/news/business-65784505 
- https://www.mili.eu/insights/is-plant-based-milk-making-a-splashin-southeast-asia-market-research 
- https://www.straitstimes.com/singapore/first-locally-produced-vegan-cheese-to-offer-more-alternatives-to-consumers 
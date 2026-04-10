# CrashSense
Intelligent Accident Hotspot Analytics

Road traffic accidents remain one of the leading causes of fatalities and serious injuries 
worldwide, posing a major challenge to public safety and urban infrastructure planning. 
The rapid growth of vehicular traffic and urbanization has resulted in a massive volume of 
accident-related data that cannot be efficiently processed using traditional data analysis 
techniques. CrashSense: Intelligent Accident Hotspot Analytics aims to leverage Big 
Data Analytics to analyze large-scale road accident datasets and identify accident-prone 
locations, commonly referred to as accident hotspots. 

To identify accident hotspots, spatial aggregation and density-based analysis techniques are 
applied to detect regions with high accident concentration. Furthermore, machine learning 
models are employed to predict accident severity and assess the risk level of specific 
locations based on historical patterns. These predictive insights enable proactive 
identification of high-risk zones, allowing authorities to take preventive measures rather 
than reactive responses. 
The results of the analysis are visualized through intuitive charts, heatmaps, and geographic 
maps that provide clear and actionable insights. By combining large-scale data processing, 
intelligent analytics, and effective visualization, CrashSense demonstrates how Big Data 
technologies can be used to enhance road safety planning and decision-making. The 
proposed system can assist traffic authorities, urban planners, and policymakers in 
optimizing resource allocation, improving road infrastructure, and ultimately reducing 
accident rates and fatalities.

The dashboard is highly dynamic, not just a static visualization!

Instead of just showing you past data, it actively takes the user's input (like time of day or current live weather) and runs the Machine Learning (Random Forest) model in real-time to generate brand new predictions on the fly.

Here is a breakdown of what the dashboard is showing and the insights you can gain from it:

1. Dynamic Inputs & Real-Time ML Predictions
What it does: When you change the slider for "Hour of Day", select a new "Weather Condition", or automatically fetch live OpenWeatherMap data for a city, the dashboard overrides the original dataset features (Avg_Hour, Avg_Visibility, Rain_Accidents) with these new simulated values.
The ML Magic: It then feeds these updated numbers into your loaded risk_model_v2.pkl model to predict new Risk Levels (High, Medium, Low) for all 26,000+ grid cells in real-time.
2. 📍 Live Risk Map
What it shows: An interactive map of the US where every grid cell is plotted as a colored circle (Red = High Risk, Orange = Medium, Green = Low).
Insights: You can visually see where the dangerous accident hotspots are clustering under the given weather/time conditions. For example, you might notice that coastal cities light up red when you simulate "Heavy Rain".
3. 📊 Risk Distribution & Metric Cards
What it shows: The top cards and the bar chart show the percentage breakdown of grid cells by risk level.
Insights: You get an instant high-level overview. If you change the slider from Noon (12:00) to Midnight (00:00), you can instantly see if the overall percentage of "High Risk" areas jumps up across the country.
4. 🔄 Scenario Comparison (Before vs After)
What it shows: A side-by-side grouped bar chart comparing the "Baseline" (the original historical dataset risk) against your "Simulated" risk.
Insights: This is the most powerful feature. It tells you the exact impact of your simulation. It will literally tell you: "Impact on High-Risk Cells: 🔺 +1,500 cells" to quantify exactly how much significantly worse the roads become when it rains versus when it's clear.
5. 🧠 Feature Impact (Model Importances)
What it shows: A chart showing which factors the AI model cares about most.
Insights: It helps you trust the AI and understand the root causes of accidents. In your model, it clearly shows that Average Visibility (0.28) and Rain (0.24) are the biggest predictors of a location becoming a high-risk accident zone, followed closely by the number of Traffic Signals!
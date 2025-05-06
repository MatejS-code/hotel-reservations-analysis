"""
Created on Sat Apr 26 16:08:04 2025

@author: Matej S

Kaggle dataset analysis - 
https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =============================================================================
# import the dataset
# =============================================================================
hotels_df = pd.read_csv('Hotel Reservations.csv')


# =============================================================================
# set Booking ID to index for hotels_df 
# =============================================================================
hotels_df = hotels_df.set_index('Booking_ID')


# =============================================================================
# explore the dataset
# =============================================================================
hotels_df.info()          # Check data types & missing values
hotels_df.describe()      # Basic stats
hotels_df.isnull().sum()  # Count missing values
hotels_df.head()          # Display the first 5 rows of the dataset


# =============================================================================
# cleaning the dataset/remove whitespace
# =============================================================================
hotels_df.columns = hotels_df.columns.str.strip()


# =============================================================================
# analyzing the dataset
# =============================================================================
booking_status_perc = hotels_df['booking_status'].value_counts(normalize=True
                            ) * 100  

#reset the index of the new dataframe booking_status_perc, to get two rows
booking_status_perc = booking_status_perc.reset_index()

#renaming the two columns
booking_status_perc.columns = [['booking_status', 'percentage']]

#adding '%' to the values in column percentage
booking_status_perc['percentage'] = booking_status_perc[
                                        'percentage'].round(2).astype(str) + '%'

# analyzing with vizualisations


#1. Booking Status Distribution

sns.countplot(x='booking_status', data=hotels_df)
plt.title('Booking Status Distribution')
plt.show()


#2. Lead Time vs Cancellation

sns.boxplot(x='booking_status', y='lead_time', data=hotels_df)
plt.title('Lead Time by Booking Status')
plt.show()


#3. Market Segment vs Cancellation
sns.countplot(x='market_segment_type', hue='booking_status', data=hotels_df)
plt.xticks(rotation=0)
plt.title('Market Segment vs Booking Status')
plt.show()


#4. Correlation Heatmap
corr = hotels_df.select_dtypes(include='number').corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap='coolwarm', annot=True)
plt.xticks(rotation=90)
plt.title('Correlation Heatmap')
plt.show()

#5. Special Requests by Booking Status boxplot
sns.boxplot(x='booking_status', y='no_of_special_requests', data=hotels_df)
plt.title('Special Requests by Booking Status')
plt.show()


#6. Repeated Guest vs Booking Status boxplot
sns.countplot(x='repeated_guest', hue='booking_status', data=hotels_df)
plt.title('Repeated Guest vs Booking Status')
plt.show()



"""
Questions to answer

1. Booking Status Distribution	- How common are cancellations?
About 30–35% of the bookings are canceled, and 65–70% are not canceled.
→ Cancellations are fairly common, but most bookings are still completed.

2. Lead Time vs Cancellation - Do people who book earlier cancel more?
Yes.
→ Canceled bookings usually have longer lead times.
People who book very early are more likely to cancel than those who book closer to the check-in date.

3. Market Segment vs Cancellation - Which customer group cancels most often?
 The Online Travel Agency (OTA) segment cancels the most.
→ Bookings made through OTAs have a higher cancellation rate compared to direct bookings or corporate clients.

4. Correlation Heatmap - Which numerical variables are related?
 Most notable correlations:

- lead_time is positively correlated with cancellation 
(you might not see it directly if you didn't include booking_status numeric).
- no_of_special_requests might be slightly negatively correlated with cancellation.
- no_of_children might have a slight correlation with total guests,
 but overall correlations are quite weak (low values)
 → No extremely strong correlation between variables (no near 1.0 or -1.0).

5. Special Requests by Booking Status boxplot - Does the number 
of special requests affect cancellation?
Yes.
→ Guests with more special requests are less likely to cancel.
Canceled bookings usually have fewer or no special requests, while completed bookings often have more special requests.

6. Repeated Guest vs Booking Status boxplot - Does repeated guest behavior 
affect cancellation?
Yes.
→ Repeated guests are much less likely to cancel.
First-time guests cancel way more often than returning (loyal) guests.


# =============================================================================
# Conclusion:
# =============================================================================

The analysis shows that approximately one-third of hotel bookings are canceled,
with cancellations being more common among guests who book far in advance
and through online travel agencies. 
Guests with more special requests and returning customers 
are significantly less likely to cancel their bookings. 
Correlation between numerical features is generally weak,
but lead time shows a slight positive relationship with cancellations. 
Overall, customer type, booking behavior, and engagement level appear to play 
important roles in predicting cancellations.
"""





































































































































































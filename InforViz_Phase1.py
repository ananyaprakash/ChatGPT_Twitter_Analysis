##############################
# IMPORTS
##############################
import pandas as pd
import numpy as np
import datetime as datetime
import regex as re
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
from prettytable import PrettyTable
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from tabulate import tabulate
import matplotlib.pyplot as plt
from statsmodels.graphics import gofplots

########################################
#%% Step 1: Load dataframe
########################################

df = pd.read_csv('../ChatGPT_tweets.csv')
description = df.describe().round(2)
table = tabulate(description, headers='keys', tablefmt='pretty')
print(table)
print(df.head(5).to_string())

#########################################
#%% Step 2: Preprocess/ Clean
########################################

#########################################
#%% Step 2.1: Drop nan
########################################
print(df.isnull().sum())
print("Number of observations", df.shape[0])
df.dropna(inplace=True)
print(df.isnull().sum())
print("Number of observations after dropping nans", df.shape[0])



#########################################
#%% Step 2.2: Clean text data in field 'content' : ref. source: https://medium.com/@ka2612/the-chatgpt-phenomenon-unraveling-insights-from-500-000-tweets-using-nlp-8ec0ad8ffd37
########################################
def text_process(text):
    # Remove new line characters
    text = re.sub('[\r\n]+', ' ', text)

    # text = re.sub(r'@\w+', '', text)
    # text = re.sub(r'#\w+', '', text)

    # Remove links
    text = re.sub('http://\S+|https://\S+', '', text)
    text = re.sub('http[s]?://\S+', '', text)
    text = re.sub(r"http\S+", "", text)

    text = re.sub('&amp', 'and', text)
    text = re.sub('&lt', '<', text)
    text = re.sub('&gt', '>', text)

    # Remove multiple space characters
    text = re.sub('\s+',' ', text)

    # Convert to lowercase
    text = text.lower()
    return text

df['content'] = df['content'].apply(text_process)
print(df['content'].head(10).to_list())

#%% validating that there is no null text values
print(df[df['content'].isnull()== True])

#########################################
#%% Step 2.3: Feature Engineering
########################################

#########################################
#%% Step 2.3.1: Extract categorical
########################################
df['datetime'] = pd.to_datetime(df['date'], format="%Y-%m-%d %H:%M:%S%z",utc=True)
df['date'] = df['datetime'].dt.date
df['month'] = df['datetime'].dt.month_name()
df['time'] = df['datetime'].dt.time
df.head(5)



#%%
def extract_hashtags(text):
    hashtags = re.findall(r'#\w+', text)
    return hashtags

df['hashtags'] = df['content'].apply(extract_hashtags)
df['hashtags'] = df['hashtags'].fillna('').apply(lambda x: [] if x == '' else x)

#%%
def extract_tagged_users(text):
    users = re.findall(r'@\w+', text)
    return users

df['tagged_users'] = df['content'].apply(extract_tagged_users)
df['tagged_users'] = df['tagged_users'].fillna('').apply(lambda x: [] if x == '' else x)


#########################################
#%% Step 2.3.1: Extract numerical
########################################
df['content_length'] = df['content'].apply(len)
df['hashtag_count'] = df['hashtags'].apply(len)
df['tagged_users_count'] = [len(x) for x in df['tagged_users']]

#%%
print(df.head(5).to_string())

#########################################
#%% Step 2.4: Tabulate final dataset description
########################################
description = df.describe().round(2)
table = tabulate(description, headers='keys', tablefmt='pretty')
print(table)


#########################################
#%% Step 3: Outlier analysis
########################################
numerical = ['like_count' , 'retweet_count',  'content_length','hashtag_count','tagged_users_count']
#
# Function to remove outliers using IQR for all numerical columns
def remove_outliers_iqr_all_columns(df):
    for column_name in numerical:
        q1 = df[column_name].quantile(0.25)
        q3 = df[column_name].quantile(0.75)
        iqr = q3 - q1

        # Define the lower and upper bounds for outliers
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        # Filter the DataFrame to keep values within the bounds
        df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    return df

# Remove outliers using IQR for all numerical columns
df_no_outliers = remove_outliers_iqr_all_columns(df)

# Display the results
print("Original DataFrame shape:", df.shape)
print("DataFrame shape after removing outliers:", df_no_outliers.shape)

# Optional: Visualize the data before and after outlier removal
import matplotlib.pyplot as plt

# Boxplot for each feature before and after outlier removal
plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
df.boxplot()
plt.title("Boxplot Before Outlier Removal")
plt.xticks(rotation=90)

plt.subplot(1, 2, 2)
df_no_outliers.boxplot()
plt.title("Boxplot After Outlier Removal")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


#%%
description = df_no_outliers.describe().round(2)
table = tabulate(description, headers='keys', tablefmt='pretty')
print(table)

#########################################
#%% Step 3.1: Reduce database to focus on higher likes/retweets
########################################
print("Original DataFrame shape:", df.shape)
retweet_freq=df['retweet_count'].value_counts()
print(retweet_freq)
## we eliminate records which have only one observation with x number of retweets i.e, these are outliers in retweet_count
selected_retweets = retweet_freq[retweet_freq > 1].index
df_filtered= df[df['retweet_count'].isin(selected_retweets)]
print(df_filtered.shape)

print(df_filtered.describe())
#%%
print(df_filtered.var())
#%%
plt.style.use('dark_background')
sns.set_palette("rocket")
#########################################
#%% Step 3.1: Hist plots
########################################
plt.figure()
plt.hist(df_filtered['like_count'], bins=50)
plt.title("Histogram plot of like_count [raw data]")
plt.xlabel("Like count value")
plt.ylabel("Frequency")
# plt.grid()
plt.tight_layout()
plt.show()
#%%
plt.figure()
plt.hist(df_filtered['retweet_count'], bins=50)
plt.title("Histogram plot of retweet_count [raw data]")
plt.xlabel("Retweet count value")
plt.ylabel("Frequency")
plt.show()

#%%
# If we want to furhter narrow our scope, we can drop all those observations where retweet count is 0

df_filtered_1 = df[df['retweet_count']>0].copy()
df_filtered_1.describe()
#%%
print(df_filtered_1.var())
#This has mre variation in the data, so we choose to use this hereafter
#%%

#########################################
#%% Step 4: Transformation
########################################
from scipy.stats import boxcox
columns_to_transform = ['like_count', 'retweet_count']

# Perform Box-Cox transformation on the specified columns
for column in columns_to_transform:
    # Adding 1 to handle zero or negative values
    transformed_data, lambda_value = boxcox(df_filtered_1[column] + 1)
    df_filtered_1[column + '_transformed'] = transformed_data

# Display the DataFrame with transformed columns
print("DataFrame with Transformed Columns:")
print(df_filtered_1.head())

#%%

#%%
print(df_filtered_1.head().to_string())
# df_filtered.columns
print(df_filtered_1.describe().to_string())
#%%
plt.figure()
plt.hist(df_filtered_1['like_count_transformed'], bins=50)
plt.title("Histogram plot of like_count [box cox transformed data]")
plt.xlabel("Like count value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
#%%
plt.figure()
plt.hist(df_filtered_1['retweet_count_transformed'], bins= 50)
plt.title("Histogram plot of retweet_count [box cox transformed data]")
plt.title("Histogram plot of retweet_count [raw data]")
plt.xlabel("Retweet count value")
plt.tight_layout()
plt.show()

#%%
# df_filtered_1.drop(columns=['like_class'], axis=1, inplace=True)
#########################################
#%% Step 4: Creating Target variable class, categorical one-hot encoding
########################################
df_filtered_1.insert(5, 'like_class', pd.qcut(df_filtered_1['like_count_transformed'], q=3, labels=['Low','Medium', 'High']))
# categorical= ['month']
# df_dummy = pd.get_dummies(df_filtered_1, columns=categorical)
# print("####### First 5 rows of converted features ########")
# print(df_dummy.tail(5))
# print(df_dummy.shape)
#%%
print("Class distribution")
class_dist =df_filtered_1['like_class'].value_counts()
counts_df = pd.DataFrame({'Category': class_dist.index, 'Count': class_dist.values})
# Use tabulate to display the result in tabular format
table = tabulate(counts_df, headers='keys', tablefmt='pretty', showindex=False)
# table = tabulate(class_dist, headers='keys', tablefmt='pretty')
print(table)

#%%
df_filtered_1.to_csv('ChatGPT_tweets_processed.csv', index=False)
##############################
# Phase I : Static graphs
##############################
#%%
print(df_filtered_1.shape)


##############################
#%% 1. Line Plot
##############################
grouped_by_date= df_filtered_1.groupby(['date'])['id'].count()
print(grouped_by_date)

#%%
plt.figure(figsize=(14,10))
plt.plot(grouped_by_date.index,grouped_by_date.values, lw=7, color = 'cyan')
plt.xlabel('Date')
plt.ylabel('Number of posts')
plt.title('Number of posts per day')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
##############################
#%% 2. Bar Plot
##############################

# plot of top 20 most frequently posting users
grouped = df_filtered_1['username'].value_counts().sort_values(ascending=False)[:20]
top_usernames = grouped.keys()
# grouped = grouped[:20]
# print(grouped)
plt.figure()
sns.barplot(y =list(grouped.keys()), x =list(grouped.values), palette='rocket',orient='h')
plt.ylabel('Username')
plt.xlabel('Number of posts')
plt.title('Top 20 most active users')
plt.tight_layout()
plt.show()


##############################
#%% 2.b Bar Plot (grouped)
##############################
top_usernames_data= df_filtered_1[df_filtered_1['username'].isin(top_usernames)]
plt.figure(figsize=(10, 8))
sns.barplot(y='username', x='like_count', hue='month', data=top_usernames_data, estimator=sum, ci=None, palette='rocket',orient='h' )
plt.title('Grouped Bar Plot: Sum of Like Count by Username and Month')
plt.xlabel('Username')
plt.ylabel('Sum of Like Count')
plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.xlabel('Category')
plt.ylabel('Value')
plt.tight_layout()
plt.show()


##############################
#%% 3. Stacked Count plot
##############################
filtered_data= df_filtered_1[df_filtered_1['username'].isin(top_usernames)]
filtered_data_group = filtered_data.groupby('username')
# print(filtered_data_group.head(5))
plt.figure(figsize=(10, 8))
sns.countplot(x='username', hue='like_class', data=filtered_data, palette='rocket',dodge=False)
plt.title('Stacked Count Plot for Top 20 Users (by number of posts)')
plt.xlabel('Username')
plt.ylabel('Number of posts')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.legend(title='like_class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

##############################
#%% 4. Pie Chart
##############################
# extract top 10 hashtags used and print their percentage usage
hashtags = df_filtered_1['content'].str.findall(r'#\w+')
hashtags_count = hashtags.explode().value_counts()
top_ten_hashtags = hashtags_count[0:10]

plt.figure(figsize=(14, 10))
sns.set_palette("rocket")
ax = plt.subplot()
ax.pie(top_ten_hashtags, labels=top_ten_hashtags.index, autopct='%1.1f%%',textprops = {'color': 'white'}, startangle=40, pctdistance=0.85,labeldistance=1.05,explode=(0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0))
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Top Ten Hashtags Distribution')
plt.tight_layout()
plt.show()
#%%
top_ten_hashtags_no_chatgpt = hashtags_count[1:11]
plt.figure(figsize=(14, 10))
sns.set_palette("rocket")
ax = plt.subplot()
ax.pie(top_ten_hashtags_no_chatgpt, labels=top_ten_hashtags_no_chatgpt.index, autopct='%1.1f%%',textprops = {'color': 'white'}, startangle=40, pctdistance=0.85,labeldistance=1.05,explode=(0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0))
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Top Ten Hashtags Distribution (excluding #chatgpt)')
plt.tight_layout()
plt.show()

##############################
#%% 5. dist plot
##############################
plt.figure()
sns.displot(df_filtered_1, x="content_length", hue='like_class', palette='rocket')
plt.title('Distribution plot of content_length')
plt.tight_layout()
plt.show()

#%%
plt.figure()
sns.displot(df_filtered_1, x="like_count", hue='month', palette='rocket')
plt.title('Distribution plot of content_length')
plt.tight_layout()
plt.show()

##############################
#%% 6. pair plot
##############################
numerical_cols = ['like_count', 'retweet_count', 'content_length','hashtag_count','tagged_users_count']
print(df_filtered_1[numerical_cols +['like_class']][:50000].head(5))
#%%
plt.figure()
sns.pairplot(df_filtered_1[numerical_cols +['like_class']][:50000], hue="like_class")
plt.suptitle('Pair plot of numerical features (first 50K observation)')
plt.tight_layout()
plt.show()

##############################
#%% 7. Heatmap with cbar
##############################
# correlation matrix and heatmap
numerical= df_filtered_1[numerical_cols]
correlation_matrix = numerical.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
plt.title("Feature Correlation Matrix of Numerical Features")
plt.show()

##############################
#%% 8. Histogram plot with kde
##############################
plt.figure()
sns.histplot(df_filtered_1, x='like_count_transformed', kde=True, color='cyan')
plt.title('Distribution plot of like_count with kde')
plt.tight_layout()
plt.show()


##############################
#%% 8. QQ plot

##############################
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

gofplots.qqplot(df_filtered_1['like_count'], line='s', ax=axes[0, 0], color='cyan')
axes[0, 0].set_title('QQ Plot of like_count [raw data]')
gofplots.qqplot(df_filtered_1['like_count_transformed'], line='s', ax=axes[0, 1], color='cyan')
axes[0, 1].set_title('QQ Plot of like_count [transformed data]')
gofplots.qqplot(df_filtered_1['retweet_count'], line='s', ax=axes[1, 0], color='cyan')
axes[1, 0].set_title('QQ Plot of retweet_count [raw data]')
gofplots.qqplot(df_filtered_1['retweet_count_transformed'], line='s', ax=axes[1, 1], color='cyan')
axes[1, 1].set_title('QQ Plot of retweet_count [transformed data]')
plt.suptitle("Before and after transformation of like_count and retweet_count")
plt.show()

##############################
#%% 10. KDE Plot with fill
##############################
plt.figure()
sns.kdeplot(data=df_filtered_1, x="hashtag_count", fill= True, alpha=0.6, hue='like_class', linewidth=1.2, legend=True, palette = 'rainbow')
plt.title('KDE plot for hashtag count')
plt.show()

##############################
#%% 11. lm plot with scatter and regreession line
##############################
plt.figure(figsize=(10, 8))
sns.lmplot(data=df_filtered_1, x="retweet_count", y="like_count", palette='bright')
plt.title("lm plot of like_count and retweet_count")
plt.tight_layout()
plt.show()

##############################
#%% 12. Multivariate box plot
##############################
## completed above in outlier analysis

##############################
#%% 13. Area plot
##############################

monthly_df = df_filtered_1.groupby(['month', 'date']).agg(posts_per_date = ('date','count')).reset_index()
monthly_df['date_only']= pd.to_datetime(monthly_df['date']).dt.day
months= ['January','February','March']
print(monthly_df)
colors=['maroon', 'orange','purple']
jan_posts= monthly_df[monthly_df['month']=='January']
feb_posts= monthly_df[monthly_df['month']=='February']
mar_posts= monthly_df[monthly_df['month']=='March']

plt.figure(figsize=(14,10))
plt.plot(jan_posts['date'],jan_posts['posts_per_date'], color=colors[0], lw=3, label='January')
plt.fill_between(jan_posts['date'],0,jan_posts['posts_per_date'], alpha=0.4, label="Area of January posts", color=colors[0])
plt.plot(feb_posts['date'],feb_posts['posts_per_date'], color=colors[1], lw=3, label='February')
plt.fill_between(feb_posts['date'],0,feb_posts['posts_per_date'], alpha=0.4, label="Area of February posts", color=colors[1])
plt.plot(mar_posts['date'],mar_posts['posts_per_date'], color=colors[2], lw=3, label='March')
plt.fill_between(mar_posts['date'],0,mar_posts['posts_per_date'], alpha=0.4, label="Area of March posts", color=colors[2])
plt.title("Area plots of posts per day in each month")
plt.xlabel("Date")
plt.ylabel("Number of posts")
plt.legend()
# plt.tight_layout()
plt.xticks(rotation=90)

plt.show()

##############################
#%% 14. Violin plot
##############################
monthly_df.rename(columns={('date','count'): ('count','count')}, inplace=True)
plt.figure()
sns.catplot(x='month', y='posts_per_date', kind='violin', data= monthly_df)
plt.title('Violin plot of posts in each month')
plt.tight_layout()
plt.show()

##############################
#%% 15. Joint plot
##############################
plt.figure()
sns.jointplot(data= top_usernames_data, x="hashtag_count", y="retweet_count", hue='like_class', palette='rainbow')
plt.tight_layout()
plt.show()

##############################
#%% 16. Rug plot
##############################
plt.figure()
sns.scatterplot(data=df_filtered_1, x='like_count', y='retweet_count',hue='month', palette='rainbow')
sns.rugplot(data=df_filtered_1, x='like_count', y='retweet_count',hue='month', palette='rainbow')
plt.title('Scatter and rug plot of like_count vs retweet_count with hue as month')
plt.show()

##############################
#%% 17. 3D plot and contour plot
##############################
#%%
print(df_filtered_1.shape)
#%%
x = np.linspace(-4, 4, 800)
y = np.linspace(-4, 4, 800)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))
print(x)
print(z)


#%%
# n = 800
# x = np.linspace(-4, 4, n)
# y = np.linspace(-4, 4, n)

x = df_filtered_1['retweet_count_transformed'][50000:55000]
y= df_filtered_1['content_length'][50000:55000]
x, y = np.meshgrid(x, y)
z = df_filtered_1[['like_count_transformed']][50000:55000]

# z = np.sin(np.sqrt(x**2 + y**2))
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='coolwarm')
ax.contour(x, y, z, zdir='z', offset=-6, cmap='coolwarm',
lineweidth=1)
ax.contour(x, y, z, zdir='x', offset=-5, cmap='coolwarm',
linewidth=1)
ax.contour(x, y, z, zdir='y', offset=5, cmap='coolwarm',
linewidth=1)
ax.set_xticks(np.arange(-4,5,1))
ax.set_yticks(np.arange(-4,5,1))
ax.set_zticks(np.arange(-6,3,1))
ax.set_xlabel('X label', fontdict={'family':'serif',
'color':'darkred','size':15})
ax.set_ylabel('Y label', fontdict={'family':'serif',
'color':'darkred','size':15})
ax.set_zlabel('Z label', fontdict={'family':'serif',
'color':'darkred','size':15})
plt.title("Surface Plot of $z = \sin\sqrt{x^2 + y^2}$",
fontdict={'family':'serif', 'color':'blue', 'size':25})
plt.show()


##############################
#%% 18. 3D plot and contour plot
##############################

##############################
#%% 19. Cluster map
##############################
print(df_filtered_1[numerical_cols][92000:].shape)
#%%
plt.figure()
cluster_map = sns.clustermap(
    df_filtered_1[numerical_cols][92000:],
    figsize=(14, 12),
    metric="correlation",
    method="single",
    standard_scale =1,
)
for collection in cluster_map.ax_col_dendrogram.collections:
    collection.set_linewidth(4)

for collection in cluster_map.ax_row_dendrogram.collections:
    collection.set_linewidth(4)
# plt.tight_layout()
plt.suptitle("Cluster map of numerical features")
plt.show()

##############################
#%% 19. Hex bin
##############################
plt.figure(figsize=(14,10))
sns.jointplot(x=df_filtered_1['hashtag_count'], y=df_filtered_1['content_length'], kind="hex", color="#e81752")
plt.suptitle('Hex bin plot of hashtag_count versus tagged_users_count')
plt.tight_layout()
plt.grid()
plt.show()

##############################
#%% 20. Strip plot
##############################
# df_grouped = df_filtered_1.groupby(['date']).count().reset_index()
# print(df_grouped.head(10))
sns.stripplot(data=df_filtered_1, y="month", x="retweet_count")
plt.title('Strip plot of retweet_count for each month')
plt.grid()
plt.show()
##############################
#%% 21. Swarm plot
##############################
class_grouped_df = df_filtered_1.groupby(['like_class', 'date']).agg(posts_per_date = ('date','count')).reset_index()
plt.figure()
sns.swarmplot(x='like_class', y='posts_per_date', data= class_grouped_df)
plt.title('Swarm plot of posts_per_date for each like_class')
plt.grid()
plt.show()


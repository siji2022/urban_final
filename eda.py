# EDA of sold data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# read the sold data
fn='./urban_computing/final_project_hp/data/sold_chicago_5yrs_sample.csv'
sold=pd.read_csv(fn)
# convert column sold date to datetime
sold['SOLD DATE']=pd.to_datetime(sold['SOLD DATE'])
# fill missing value with 0
sold=sold.fillna(0)
sold['YEAR BUILT']=2023-sold['YEAR BUILT']
# normalize columns
normalize_columns=['PRICE','BEDS','BATHS','SQUARE FEET','LOT SIZE','YEAR BUILT','$/SQUARE FEET','HOA/MONTH']
sold[normalize_columns]=(sold[normalize_columns]-sold[normalize_columns].mean())/sold[normalize_columns].std()

corr_columns=['PRICE','BEDS','BATHS','SQUARE FEET','LOT SIZE','YEAR BUILT','$/SQUARE FEET','HOA/MONTH','ZIP OR POSTAL CODE']
# correlation on corr_columns

plt.figure(figsize=(15, 15))
plt.title('Correlation of features', size=16)
sns.heatmap(data=sold[corr_columns].corr(),cmap='RdBu_r', linecolor='white',annot=True,annot_kws={'size': 9, 'weight': 'bold'}, )

# save figure
plt.savefig('./urban_computing/final_project_hp/figures/sold_data_corr.png',dpi=300)
plt.close()


# sns.pairplot(data=sold[corr_columns],diag_kind='hist',kind='scatter',hue='ZIP OR POSTAL CODE')
# plt.savefig('./urban_computing/final_project_hp/figures/sold_data_pairplot.png',dpi=300)
# plt.close()

sns.stripplot(x='ZIP OR POSTAL CODE',y='PRICE',data=sold,hue='BEDS')
plt.title('Price vs Zip Code',size=16)
plt.xticks(rotation=90,fontsize=6)
plt.savefig('./urban_computing/final_project_hp/figures/price_vs_zipcode.png',dpi=300)

plt.close()

sns.stripplot(x='YEAR BUILT',y='PRICE',data=sold,hue='BATHS')
plt.title('Price vs Year build',size=16)
plt.xticks(rotation=90,fontsize=6)
plt.savefig('./urban_computing/final_project_hp/figures/price_vs_yearbuilt.png',dpi=300)

plt.close()

sns.scatterplot(x='SQUARE FEET',y='PRICE',data=sold,hue='BATHS')
plt.title('Price vs Square Feet',size=16)
plt.xticks(rotation=90,fontsize=6)
plt.savefig('./urban_computing/final_project_hp/figures/price_vs_squarefeet.png',dpi=300)
plt.close()

sns.scatterplot(x='SOLD DATE', y='PRICE', data=sold, hue='ZIP OR POSTAL CODE')
plt.title('Price vs Sold Date', size=16)
plt.xticks(rotation=90, fontsize=6)
plt.savefig('./urban_computing/final_project_hp/figures/price_vs_solddate.png', dpi=300)
plt.close()

# plot KDE
sns.displot(sold,x='PRICE',col='PROPERTY TYPE',kde=True,legend=True)
plt.legend()
plt.savefig('./urban_computing/final_project_hp/figures/KDE_price_vs_property_types.png')
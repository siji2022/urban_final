## Since the direct download form Redfin has a limit of 10000 records, we need to download the data region
import os
import pandas as pd

raw_data='./urban_computing/final_project_hp/data/__tempdata/'

### read all csv files under the raw_data folder
### combine all files into one dataframe
### remove the duplicate records

def read_all_csv_files():
       df=pd.DataFrame()
       idx=0
       for file in os.listdir(raw_data):
              idx+=1
              if file.endswith(".csv"):
                     print(f'processing {file}')
                     df_temp=pd.read_csv(raw_data+file)
                     # merge the dataframes df_temp to df
                     df=pd.concat([df,df_temp],axis=0)
                     df.drop_duplicates(inplace=True)
              if idx >300: # shouldn't have more than 300 files
                     break
       return df

def clean_files(df):
       # check missing values in the columns
       missing_values_check=['PRICE','SOLD DATE','MLS#','LATITUDE']
       df.dropna(subset=missing_values_check,inplace=True)
       print(df[missing_values_check].isnull().sum())
       return df

def drop_not_used_columns(df):
       columns_to_drop=['LOCATION','SALE TYPE','NEXT OPEN HOUSE START TIME','NEXT OPEN HOUSE END TIME','URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)','STATUS', 'SOURCE','MLS#','FAVORITE','INTERESTED']
       for column in columns_to_drop:
              if column in df.columns:
                     df.drop(columns=column,inplace=True,axis=1)
              else:
                     print(f'{column} not in the columns')
       # df.drop(columns=columns_to_drop,inplace=True)


# save all_files to a csv file
# all_files=read_all_csv_files()
# print(all_files.shape) #(157565, 27)
# all_files=clean_files(all_files)

# drop_not_used_columns(all_files)
# all_files.to_csv('./urban_computing/final_project_hp/data/sold_chicago_5yrs.csv',index=False)
# print(all_files.shape) #(128805, 16)

# get all listings in Chicago
# listing=pd.read_csv('./uc/final_project_hp/data/listing_chicago_20231113.csv')
# print(listing.shape)
# print(listing.isnull().sum())
# drop_not_used_columns(listing)
# # save to csv
# listing.to_csv('./uc/final_project_hp/data/listing_chicago_20231113_cleaned.csv',index=False)

# get all sold listings in Chicago
sold=pd.read_csv('./uc/final_project_hp/data/sold_chicago_5yrs_cleaned.csv')  # 128759


# clean property types
# sold.loc[sold['PROPERTY TYPE']=='Co-op','PROPERTY TYPE']='Condo/Co-op'
# sold.loc[sold['PROPERTY TYPE']=='Ranch','PROPERTY TYPE']='Other'
# sold.loc[sold['PROPERTY TYPE']=='Unknown','PROPERTY TYPE']='Other'
# sold.loc[sold['PROPERTY TYPE']=='Mobile/Manufactured Home','PROPERTY TYPE']='Other'
# sold=sold.sort_values(by='SOLD DATE')
# # remove the duplicate records
# sold.drop_duplicates(inplace=True)

# sold.to_csv('./uc/final_project_hp/data/sold_chicago_5yrs_cleaned.csv',index=False)
# print(sold.describe())
# remove price less than 500 records
sold=sold[sold['PRICE']>500]
sold['SOLD DATE']=pd.to_datetime(sold['SOLD DATE'])
# # Resample from SOLD data by property type and store in sample_df
sample_df=pd.DataFrame()
for property_type in ['Condo/Co-op','Single Family Residential','Multi-Family (2-4 Unit)','Multi-Family (5+ Unit)','Townhouse','Vacant Land','Parking']:
       select_property_type=sold[sold['PROPERTY TYPE']==property_type]
       print(f'{property_type}: {select_property_type.shape}')
       if select_property_type.shape[0]<10000:
              sample_df=pd.concat([sample_df,select_property_type],axis=0)
       else:
              # sort by price
              select_property_type=select_property_type.sort_values(by='PRICE',ascending=False)
              # include the top 1000 records
              sample1=select_property_type[:10]
              sample2=select_property_type[10:].sample(5000,replace=False,random_state=0)
              sample3=select_property_type[select_property_type['SOLD DATE'].dt.year==2023]  # sold date in 2023 are included in the sample
              sample_df=pd.concat([sample_df,sample1,sample2,sample3],axis=0)
             


print(sample_df.shape)
# # remove the duplicate records
sample_df.drop_duplicates(inplace=True)
print(sample_df.shape) #(52822, 16) (40577, 16) (40426, 16)
sample_df.to_csv('./uc/final_project_hp/data/sold_chicago_5yrs_sample_bytypes.csv',index=False)



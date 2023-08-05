#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#from google.colab import files
#df1 = files.upload()


# In[3]:


df = pd.read_excel("Supplychain train dataset.xlsx")


# In[4]:


df.info()


# In[5]:


df['Location_type'].unique()


# In[6]:


df['wh_owner_type'].unique()


# In[7]:


df.isnull().sum()


# ## Data Preprocessing

# Replacing null values for number of workers with median

# In[8]:


df['workers_num'] = df['workers_num'].fillna(value = np.nanmedian(df['workers_num']))


# In[9]:


df['workers_num'].isnull().sum()


# In[10]:


df['wh_est_year'].mode()


# Replacing null values for warehouse establishment year with the mode

# In[11]:


df['wh_est_year'] = df['wh_est_year'].fillna(value = 2000)


# In[12]:


df['wh_est_year'].isnull().sum()


# In[13]:


df['approved_wh_govt_certificate'].mode()


# Replacing null values in approved certificate with mode

# In[14]:


df['approved_wh_govt_certificate'] = df['approved_wh_govt_certificate'].replace(np.nan, 'C', regex = True)


# In[15]:


df['approved_wh_govt_certificate'].isnull().sum()


# In[16]:


df.isnull().sum()


# In[17]:


df.head()


# In[18]:


df_num = df.select_dtypes(include = 'number')
df_num


# In[19]:


df_cat = df.select_dtypes(exclude = 'number')
df_cat


# As there is no sognificant difference in mean and median values, outliers are not present

# ### Hypothesis testing for numeric variables(pearson's correlation test)

# 
# **1)No of refills in last 3 months**
# 

# In[20]:


# Null hypothesis(H0) : If the number of refills >=  4  ,num of refills does not affect weight.
# Alternate hypothesis(H1) : If number of refills < 4  num of refills are less than mean then weight decreases
# z-test
import random
import math
import numpy as np
from statsmodels.stats import weightstats as stats

#np.random.seed(123)

sample_data = df.loc[:,'num_refill_req_l3m'].sample(n = 2000)

null_value = round(np.mean(df['num_refill_req_l3m']),2)

sample_mean =round(np.mean(sample_data),2)

z_score, p_value = stats.ztest(sample_data, None, null_value, alternative = 'smaller')

alpha = 0.05

if p_value < alpha:
    print(round(p_value,2),"<", alpha,"Reject the null hypothesis")

else:
    print(round(p_value,2),">", alpha,"Fail to reject the null hypothesis")


# z-test gives different p-values based on different random seeds everytime we run, hence it is not reliable

# - Null hypothesis(H0) : p-value > 0.05 There is no correlation between number of refills and weight.
# - Alternate hypothesis(H1) : p-value < 0.05 ,there is a correlation between number of refills and weight.

# In[21]:


#pearsons correlation test
from scipy.stats import pearsonr
corr_coeff, p_value = pearsonr(df.num_refill_req_l3m, df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# - Hence p-value > 0.05, there is no correlation.
# 

# In[22]:


sns.scatterplot(x =df.num_refill_req_l3m,y = df.product_wg_ton)


# **2)Competitors in market**
# - Null hypothesis(H0) : p-value > 0.05, there is no correlation between no of competitors and weight.
# - Alternate hypothesis(H1) : p-value < 0.05, there is correlation between no of competitors and weight.

# In[23]:


from scipy.stats import pearsonr
corr_coeff, p_value = pearsonr(df.Competitor_in_mkt, df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# p-value > 0.05, there is no correlation between no of competitors and weight.

# In[24]:


sns.scatterplot(x = df.Competitor_in_mkt,y =  df.product_wg_ton)


# **3)Retail shop numbers**
# - Null hypothesis(H0): If p-value > 0.05, there is correlation betweeen retail shop number and weight.
# - Alternate hypothesis(H1) : If p-value < 0.05, there is correlation betweeen retail shop number and weight.

# In[25]:


from scipy.stats import pearsonr
corr_coeff, p_value = pearsonr(df.retail_shop_num, df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# p-value > 0.05 hence there is no corelation between number of retail shops and weight.

# In[26]:


sns.scatterplot(x = df.retail_shop_num,y = df.product_wg_ton)


# **4)distributor_num**
# - Null hypothesis(H0) : If p-value > 0.05, there is no correlation between number of distributors and weight.
# - Alternative hypothesis(H1) :If p-value < 0.05, there is correlation between number of distributors and weight.

# In[27]:


from scipy.stats import pearsonr
corr_coeff, p_value = pearsonr(df.distributor_num, df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# - pvalue > 0.05 , there is no relation between number of distributors and weight.

# In[28]:


sns.scatterplot(x = df.distributor_num,y = df.product_wg_ton)


# **5)Distance of warehouse from Production hub**
# - Null hypothesis : p-value > 0.05, there is no correlation between distance and weight
# - Alternate hypothesis:  p-value < 0.05, there is  correlation between distance and weight.

# In[29]:


from scipy.stats import pearsonr
corr_coeff, p_value = pearsonr(df['dist_from_hub'], df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# p-value > 0.05 , there is no correlation between distance from hub and weight.

# In[30]:


sns.scatterplot(x = df['dist_from_hub'], y = df.product_wg_ton)


# - Hence distance and weight are not related

# **6)Storage issue in last 3 months**
# - Null hypothesis(H0): p-value  > 0.05, no correlation between storage issue and product weight.
# - Alternate hypothesis(H1): If p-value < 0.05, there is a correlation between storage issue and product weight.

# In[31]:


from scipy.stats import pearsonr
corr_coeff, p_value = pearsonr(df['storage_issue_reported_l3m'], df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# In[32]:


sns.scatterplot(x = df['storage_issue_reported_l3m'], y = df.product_wg_ton, hue = df.storage_issue_reported_l3m  > 18 )


# This may be due to overstocking, poor inventory management.
# 

# **7)Warehouse breakdown in last 3 months**
# - Null hypothses(H0): If p-value > 0.05 then there is no correlation between warehouse breakdown and weight.
# - Alternate hypothesis(H1): If p-value < 0.05 then there is correlation between warehouse breakdown and weight.

# In[33]:


from scipy.stats import pearsonr
corr_coeff, p_value = pearsonr(df['wh_breakdown_l3m'], df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# In[34]:


sns.regplot(df['wh_breakdown_l3m'],df.product_wg_ton )


# In[35]:


sns.lineplot(df['wh_breakdown_l3m'],df.product_wg_ton , df['wh_breakdown_l3m'] < 3)


# This may be due to bad inventory management, poorly organized workflow, lack of damage control.

# Hence due to warehouse breakdown availability of product, safety, quality cannot be achieved and reputation of warehouse may go down.

# **8)Government checkup of warehouse in last 3 months(no of times)**
# - Null hypothesis(H0): If p-value > 0.05 , then there is no corelation between govt chekup and weight.
# - Alternate hypothesis(H1): If p-value < 0.05 , then there is no corelation between govt chekup and weight.

# In[36]:


corr_coeff, p_value = pearsonr(df.govt_check_l3m, df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# p-value > 0.05 , there is no corelation between govt chekup and weight

# In[37]:


sns.regplot(df.govt_check_l3m, df.product_wg_ton)


# **9)Number of workers**
# - Null hypothesis(H0): If p-value > 0.05, then there is no correlation between number of workers and weight.
# - Alternate hypothesis(H1): If p-value < 0.05, then there is correlation between number of workers and weight.

# In[38]:


corr_coeff, p_value = pearsonr(df['workers_num'], df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# In[39]:


sns.regplot(df['workers_num'], df.product_wg_ton)


# If p-value > 0.05, then there is no correlation between number of workers and weight

# **10)Warehouse established year**

# - Null hypothesis: p-value > 0.05, then there is no correlation between established year and weight.
# 
# - Alternate hypothesis: p-value < 0.05, then there is correlation between established year and weight.

# In[40]:


corr_coeff, p_value = pearsonr(df['wh_est_year'], df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# p-value < 0.05, there is correlation between established year and weight.

# In[41]:


sns.lineplot(df['wh_est_year'] ,df.product_wg_ton)


# - As warehouse established year increases weight of product is decreasing.
# - It may be due to low popularity hence low sales in newly established regions.

# ## Tests for categorical variables
# ## Two sample t-test(ind)

# **11)Flood impacted area**
# - sample1 = wg_in_flood_area(μ1)
# - sample2 = wg_in_no_flood_area(μ2)
# - Null hypothesis(H0):  μ1 = μ2 , flood impact and weight are not related.
# - Alternate hypothesis(H1):  μ1 ≠ μ2 , There exists a relation between flood impact and weight.
# 

# In[42]:


wg_in_flood_area = df.product_wg_ton[df.flood_impacted == 0]
wg_in_no_flood_area = df.product_wg_ton[df.flood_impacted == 1]


# In[43]:


print(wg_in_flood_area.head())
print(wg_in_no_flood_area.head())


# In[44]:


print(wg_in_flood_area.mean())
print(wg_in_no_flood_area.mean())


# Means are almost same

# In[45]:


import scipy

scipy.stats.ttest_ind(a = wg_in_flood_area,b =  wg_in_no_flood_area,equal_var = False)


# p-value (0.45 > 0.05) hence, fail to reject null hypothesis
# - There is no relationship between flood impact and weight of noodles.
# - There is no significant differnce in mean weight of product in flood impacted and no flood area.
# - Hence flood impact has no affect on weight of product.

# one way anova

# In[46]:


scipy.stats.f_oneway(wg_in_flood_area, wg_in_no_flood_area)


# In[47]:


sns.barplot(x = df.flood_impacted, y = df['product_wg_ton'])


# In[48]:


sns.boxplot(x = df.flood_impacted, y = df['product_wg_ton'])


# Hence flood impact and no impact in both cases the weight is increasing.

# **12)Flood proof warehouse**
# - sample1 = wg_in_flood_proof(μ1)
# - sample2 = wg_in_no_flood_proof(μ2)
# - Null hpothesis(H0):μ1 = μ2, flood proof and weight are not related.
# - Alternate hypothesis(H1): μ1 ≠ μ2 , flood proof and weight are related.
# 

# In[49]:


wg_in_flood_proof = df.product_wg_ton[df.flood_proof == 0]
wg_in_no_flood_proof = df.product_wg_ton[df.flood_proof == 1]


# In[50]:


print(wg_in_flood_proof.mean())
print(wg_in_no_flood_proof.mean())


# Means are almost same

# In[51]:


# t_test

scipy.stats.ttest_ind(a = wg_in_flood_proof, b = wg_in_no_flood_proof, equal_var = False)


# In[52]:


#anova

scipy.stats.f_oneway( wg_in_flood_proof, wg_in_no_flood_proof)


# p-value (0.59 > 0.05) hence, fail to reject null hypothesis
# - There no relationship between flood proof and weight of noodles.
# - There is no significant differnce in mean weight of product in flood proof and no flood proof area.
# - Hence flood impact has no affect on weight of product.

# In[53]:


sns.barplot(x = df.flood_proof, y = df.product_wg_ton )


# Hence weight of product increases with or without flood_proof.

# **13)Electric_supply**
# - sample1 = wg_with_elect_sup(μ1)
# - sample2 = wg_without_elect_sup(μ2)
# - Null hypothesis(H0): μ1 = μ2 , there is no relation between electric supply and weight.
# - Alternate hypothesis(H1): μ1 ≠ μ2 , there exists no relation between electric supply and weight.

# In[54]:


wg_with_elect_sup = df.product_wg_ton[df.electric_supply == 0]
wg_without_elect_sup = df.product_wg_ton[df.electric_supply == 1]


# In[55]:


print(wg_with_elect_sup.mean())
print(wg_without_elect_sup.mean())


# In[56]:


# t_test

scipy.stats.ttest_ind(a = wg_with_elect_sup, b = wg_without_elect_sup, equal_var = False )


# In[57]:


scipy.stats.f_oneway( wg_with_elect_sup, wg_without_elect_sup)


# - p_value(0.91 > 0.05), hence fail to reject null hypothesis
# - There is no relationship between electric supply and weight of product.
# - There is no significant difference in mean weight of warehouse with electric supply and without electric supply

# In[58]:


sns.barplot(x = df.electric_supply, y = df.product_wg_ton)


# Hence weight is increasing with or without electric supply

# **14)Temperature_regulating_machine**
# - sample1 = wg_with_reg_mach(μ1)
# - sample2 = wg_without_reg_mach(μ2)
# - Null hypothesis(H0): μ1 = μ2, there is no relation between temperature regualting machine and weight.
# - Alternate hypothesis(H1): μ1 ≠ μ2 there exists a  relation between temperature regualting machine and weight.

# In[59]:


wg_with_reg_mach = df.product_wg_ton[df.temp_reg_mach == 0]
wg_without_reg_mach = df.product_wg_ton[df.temp_reg_mach == 1]


# In[60]:


print(wg_with_reg_mach.mean())
print(wg_without_reg_mach.mean())


# In[61]:


wg_without_reg_mach.mean() - wg_with_reg_mach.mean()


# In[62]:


# t-test

scipy.stats.ttest_ind(a = wg_with_reg_mach, b = wg_without_reg_mach, equal_var = False)


# In[63]:


# anova

scipy.stats.f_oneway(wg_with_reg_mach, wg_without_reg_mach)


# p-value < 0.05 , hence reject null hypothesis
# - There exists a significant relationship between temperature regulating machine and weight of product.
# - There is significant difference(2487) in mean weights of product with machine and without machine.
# 

# In[64]:


sns.barplot( df.temp_reg_mach, df.product_wg_ton)


# Hence the differnce in product is seen with and without temperature regulating machine

# In[65]:


df['transport_issue_l1y'].unique()


# **15)Transport issues in last 1 year**
# -  μ1, μ2, μ3, μ4, μ5 , μ6 for transport issues for 0,1,2,3,4,5 number of times.Anova is suitable here.
# - Null hypothesis(H0):  μ0 = μ1 = μ2 = μ3 = μ4 = μ=5
# - Alternate hypothesis(H1): Anyone of the means are unequal

# In[66]:


wg_trans_0 = df.product_wg_ton[df.transport_issue_l1y == 0]
wg_trans_1 = df.product_wg_ton[df.transport_issue_l1y == 1]
wg_trans_2 = df.product_wg_ton[df.transport_issue_l1y == 2]
wg_trans_3 = df.product_wg_ton[df.transport_issue_l1y == 3]
wg_trans_4 = df.product_wg_ton[df.transport_issue_l1y == 4]
wg_trans_5 = df.product_wg_ton[df.transport_issue_l1y == 5]


# In[67]:


df.transport_issue_l1y.value_counts()


# In[68]:


scipy.stats.f_oneway(wg_trans_0,wg_trans_1,wg_trans_2,wg_trans_3,wg_trans_4,wg_trans_5)


# In[69]:


sns.barplot(df.transport_issue_l1y, df.product_wg_ton)


# In[70]:


sns.boxplot(df.transport_issue_l1y, df.product_wg_ton)


# - P-value < 0.05, hence reject null hypothesis.
# - There is exists a strong relationship between transport issue and weight of product.
# - There is significant difference in mean weights of products.

# **16)warehouse location_type**
# - sample1 = location_urban(μ1)
# - sample2 - location_rural(μ2)
# - Null hypothesis(H0): μ1 = μ2, there is no relation between location and weight.
# - Alternate hypothesis(H1): μ1 ≠ μ2, there exists a relation between location and weight.

# In[71]:


location_urban = df.product_wg_ton[df.Location_type == 'Urban']
location_rural = df.product_wg_ton[df.Location_type == 'Rural']


# In[72]:


print(location_urban.mean())
print(location_rural.mean())


# In[73]:


# t-test
scipy.stats.ttest_ind(a = location_urban, b = location_rural, equal_var = False)


# In[74]:


#anova
scipy.stats.f_oneway(location_urban, location_rural)


# - p_value < 0.05, hence null hypothesis is rejected.
# - There exists a relation between warehouse location type and weight of product.
# - There is significant difference between location urban and location rural.

# In[75]:


sns.barplot(df.Location_type, df.product_wg_ton)


# In[76]:


sns.boxplot(df.Location_type, df.product_wg_ton)


# Hence there is a differnce in urban and rural weights of products

# **17)Warehouse capacity size**
# - sample1 = capacity_large(μ1)
# - sample2 = capacity_medium(μ2)
# - sample3 = capacity_small(μ3)
# - Null hypothesis = μ1 = μ2 = μ3 ,there is no relation between capacity size and weight.
# - Alternate hypothesis = μ1 ≠ μ2 or μ2 ≠ μ3, there exists a relation between capacity size and weight.

# In[77]:


capacity_large = df.product_wg_ton[df.WH_capacity_size == 'Large']
capacity_medium = df.product_wg_ton[df.WH_capacity_size == 'Mid']
capacity_small = df.product_wg_ton[df.WH_capacity_size == 'Small']


# In[78]:


print(capacity_large.mean())
print(capacity_medium.mean())
print(capacity_small.mean())


# In[79]:


#anova
scipy.stats.f_oneway(capacity_large, capacity_medium , capacity_small)


# - p_value(0.39 > 0.05), fail to reject null hypothesis.
# - There is no relation between warehouse size and weight of product.
# - There is no significant difference in mean weight of product.

# In[80]:


sns.barplot(df.WH_capacity_size, df.product_wg_ton)


# **18)warehouse zone**
# - Null hypothesis :  μ1 = μ2 = μ3 = μ4, there is no relation between zones and weight.
# - Alternate hypothesis : Anyone of the means are unequal, there exists a relation between warehouse zones and weights.

# In[81]:


North_zone = df.product_wg_ton[df.zone == 'North']
South_zone = df.product_wg_ton[df.zone == 'South']
West_zone = df.product_wg_ton[df.zone == 'West']
East_zone = df.product_wg_ton[df.zone == 'East']


# In[82]:


print(North_zone.mean())
print(South_zone.mean())
print(West_zone.mean())
print(East_zone.mean())


# In[83]:


#anova
scipy.stats.f_oneway(North_zone ,South_zone, West_zone, East_zone)


# - P_value(0.275 > 0.05), fail to reject null hypothesis.
# - There is no relation between warehouse zone and weight of Product.
# - There is no significant differnce in means weight of product.

# In[84]:


sns.barplot(df.zone, df.product_wg_ton)


# **19)Warehouse_regional_zone**
# - Null hypothesis: μ1 = μ2 = μ3 = μ4 = μ5 = μ6 , NO relation between regional zone and weight.
# - Alternate hypothesis: Anyone of means are unequal. There exists a relation between regional zone and weight.

# In[85]:


regional_zone1 = df.product_wg_ton[df.WH_regional_zone == 'Zone 1']
regional_zone2 = df.product_wg_ton[df.WH_regional_zone == 'Zone 2']
regional_zone3 = df.product_wg_ton[df.WH_regional_zone == 'Zone 3']
regional_zone4 = df.product_wg_ton[df.WH_regional_zone == 'Zone 4']
regional_zone5 = df.product_wg_ton[df.WH_regional_zone == 'Zone 5']
regional_zone6 = df.product_wg_ton[df.WH_regional_zone == 'Zone 6']


# In[86]:


print(regional_zone1.mean())
print(regional_zone2.mean())
print(regional_zone3.mean())
print(regional_zone4.mean())
print(regional_zone5.mean())
print(regional_zone6.mean())


# In[87]:


scipy.stats.f_oneway(regional_zone1, regional_zone2, regional_zone3, regional_zone4, regional_zone5, regional_zone6)


# - p_value(0.4 > 0.05), fail to reject null hypothesis
# - There is no relation between warehouse regional zones and weight of product.
# - There is no significant difference in the mean weights of product in different regional zones.

# In[88]:


sns.barplot(df.WH_regional_zone, df.product_wg_ton)


# **20) Warehouse_owner type**
# - sample1 = Rented(μ1)
# - sample2 = Owned_by_company(μ2)
# - Null hypothesis: μ1 = μ2, no relation between warehouse owner type and weight of product.
# - Alternate hypothesis: μ1 ≠ μ2, there exists relation between warehouse owner type and weight of product.

# In[89]:


Rented = df.product_wg_ton[df.wh_owner_type == 'Rented']
Owned_by_company = df.product_wg_ton[df.wh_owner_type == 'Company Owned']


# In[90]:


print(Rented.mean())
print(Owned_by_company.mean())


# In[91]:


scipy.stats.f_oneway(Rented, Owned_by_company )


# - p_value(0.6 > 0.05) , fail to reject null hypothesis.
# - There is no relation between warehouse owner type and weight of product.
# - There is no significant difference in means of weights.

# In[92]:


sns.barplot(df.wh_owner_type, df.product_wg_ton)


# **21)Aprroved_govt_certificate of warehouse**
# - Null hypothesis(H0): μ1(A+) = μ2(A) = μ3(B+) = μ4(B) = μ5(C), there is no relation between govt certificate and weight of product.
# - Alternate hypothesis(H1): Anyone one of the means are unequal, there exists relationship.

# In[93]:


grade_1 = df.product_wg_ton[df.approved_wh_govt_certificate ==  'A+']
grade_2 = df.product_wg_ton[df.approved_wh_govt_certificate ==  'A']
grade_3 = df.product_wg_ton[df.approved_wh_govt_certificate ==  'B+']
grade_4 = df.product_wg_ton[df.approved_wh_govt_certificate ==  'B']
grade_5 = df.product_wg_ton[df.approved_wh_govt_certificate ==  'C']


# In[94]:


scipy.stats.f_oneway(grade_1, grade_2, grade_3, grade_4, grade_5)


# - p_value < 0.05, reject null hypothesis.
# - There exists a good relationship between govt_aprroved_certificate and weight of product.
# - There is significant difference in means weight of product.

# In[95]:


sns.barplot(df.approved_wh_govt_certificate, df.product_wg_ton)


# # Feature engineering
# 

# Hence warehouses having high grade, approved by govt have higher weights.

# In[96]:


df.head()


# In[97]:


# dropping ID columns
df = df.drop(['Ware_house_ID','WH_Manager_ID'],axis = 1)
df.head()


# In[98]:


df_cat = df.select_dtypes(exclude = np.number)


# In[99]:


df_cat


# Label encoding for columns
# - approved_wh_govt_certificate
# - nwh_capacity_size
# - wh_regional_zone
# - zone
# - Location_type
# - warehouse_owner_type

# In[100]:


df.approved_wh_govt_certificate.unique()


# In[101]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df.approved_wh_govt_certificate = label_encoder.fit_transform(df.approved_wh_govt_certificate)


# In[102]:


# label encoding works on most number of appearances 
df.approved_wh_govt_certificate.unique()


# In[103]:


df.WH_capacity_size.unique()


# In[104]:


df.WH_capacity_size = label_encoder.fit_transform(df.WH_capacity_size)
df.WH_capacity_size.unique()


# In[105]:


df.WH_regional_zone.unique()


# In[106]:


df.WH_regional_zone = label_encoder.fit_transform(df.WH_regional_zone)
df.WH_regional_zone.unique()


# In[107]:


df.Location_type.unique()


# In[108]:


df.Location_type = label_encoder.fit_transform(df.Location_type)
df.Location_type.unique()


# In[109]:


df.zone.unique()


# In[110]:


df.zone = label_encoder.fit_transform(df.zone)
df.zone.unique()


# In[111]:


df.wh_owner_type.unique()


# In[112]:


df.wh_owner_type = label_encoder.fit_transform(df.wh_owner_type)


# In[113]:


df.wh_owner_type.unique()


# In[114]:


df


# In[115]:


df.info()


# ## creating new features

# 1)storage issues when warehouse is breakdown
# 

# In[116]:


df['storage_with_break'] = round(df.storage_issue_reported_l3m/df.wh_breakdown_l3m)


# In[117]:


#filling nan with median values 
df['storage_with_break'] = df['storage_with_break'].fillna(value = np.nanmedian(df['storage_with_break']))


# **Hypothesis test** - Pearson's correlation test

# - Null hypothesis: If p>0.05, there is no correlation between 'storage issue with break' and weight of product
# - Alternate hypothesis: If p<0.05 ,there is  correlation between 'storage issue with break' and weight of product

# In[118]:


corr_coeff, p_value = pearsonr(df['storage_with_break'], df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# - P-value < 0.05, hence reject null hypothesis
# - There is good correlation between storage issues with breakdown and product weight

# 2) refills with temp_reg_mach

# In[119]:


# anova
refill_with_temp_mach = df.num_refill_req_l3m[df.temp_reg_mach == 1]
refill_without_temp_mach = df.num_refill_req_l3m[df.temp_reg_mach == 0]


# In[120]:


scipy.stats.f_oneway(refill_with_temp_mach, refill_without_temp_mach )


# p-value < 0.05 , there exists a relation between no of refills and temp_reg_mach

# In[121]:


sns.barplot(df.temp_reg_mach, df.num_refill_req_l3m)


# creating  columns with refills with temp_reg_mach

# In[122]:


# refills without temp mach had 60% of null values hence dropped
df['refills_with_temp_mach'] = df.num_refill_req_l3m[df.temp_reg_mach == 1]
df['refills_without_temp_mach'] = df.num_refill_req_l3m[df.temp_reg_mach == 0]


# In[123]:


df.drop(['refills_without_temp_mach'], axis = 1, inplace = True)


# In[124]:


df['refills_with_temp_mach'] = df['refills_with_temp_mach'].fillna(value = np.nanmedian(df['refills_with_temp_mach']))


# In[125]:


# correlation with target variable
corr_coeff, p_value = pearsonr(df['refills_with_temp_mach'], df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# 0.007 < 0.05, hence there is relation between **refill_with_temp_mach** and weight 

# 3)Transport issues

# In[126]:


sns.countplot(df.transport_issue_l1y)


# transport issues equal to zero is very high

# In[127]:


df['transport_encoded'] = np.where(df.transport_issue_l1y == 0,1,0)


# In[128]:


corr_coeff, p_value = pearsonr(df['transport_encoded'] == 1, df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# Transport issues equal to 1 has positive correlation with target

# In[129]:


corr_coeff, p_value = pearsonr(df['transport_encoded'] == 0, df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# Transport issues equal to 0 has negative correlation with target

# 4)Warehouse age

# In[130]:


df['wh_age'] = pd.Timestamp.now().year - df.wh_est_year


# In[131]:


corr_coeff, p_value = pearsonr(df['wh_age'] > 23 , df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# If warehouse age is > 23, it has positive correlation with weight

# In[132]:


corr_coeff, p_value = pearsonr(df['wh_age'] < 23 , df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# If warehouse age < 23 it has small negative or no correlation with weight

# In[133]:


df['wh_age_encoded'] = np.where(df['wh_age'] > 23, 1, 0)


# In[134]:


df['wh_age_encoded'].unique()


# 5)year with breakdown

# In[135]:


df['year_break'] = df.wh_age * df.wh_breakdown_l3m


# In[136]:


df['year_break'] = np.where(df['year_break']  == 0, 0, 1)


# In[137]:


corr_coeff, p_value = pearsonr(df['year_break'] == 1 , df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# years with breakdown has positive correlation(where breakdown is high)

# In[138]:


corr_coeff, p_value = pearsonr(df['year_break'] == 0 , df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# Years with breakdown has negative corrlation(where breakdown is low)

# 6)retail shops with distributors

# In[139]:


df['retails_distr'] = df.retail_shop_num / df.distributor_num


# In[140]:


corr_coeff, p_value = pearsonr(df['retails_distr']  , df.product_wg_ton)
print("Correlation coefficient:", corr_coeff)
print("p-value:", p_value)


# Hence no correlation with target variable

# - Total features created(6)
# - Storage with breakdown
# - Refills with temp reg mach
# - transport issues encoded
# - wh_year
# - year with breakdown
# - retails_distr
# 

# In[141]:


df.info()


# No correlation beween retail_Shops_distr and product weight

# **Outlier detection and treatment**

# In[142]:


for column in df_num.columns:
    plt.figure()
    sns.boxplot(data=df[column])
    plt.xlabel(column)
    plt.title(f'Boxplot - {column}')
    plt.show()


# Columns having outliers(as per above boxplot)
# - Transport issues in last 1 year
# - Competitors in market
# - Retail shop number
# - No of workers

# In[143]:


df.transport_issue_l1y.unique()


# 1)Transport_issues_l1y: 
# - If the transport issues are measured by number of delays, then delays may be due to high traffic in city areas and
# - and bad weather conditions(like fog, rain) in some regions.
# - Hence transport issues(3,4,5) in an year are not considered as outliers.

# In[144]:


# competitors in market
df.Competitor_in_mkt.unique()


# In[145]:


sns.distplot(df.Competitor_in_mkt)


# - No of competitors more than 8 are considered as outliers.
# - To find the potential opportunities to optimize the supply, number of competitors are important.
# - Hence outliers are not removed

# In[146]:


# 3)Retail shop numbers
df.retail_shop_num.describe()


# In[147]:


sns.distplot(df.retail_shop_num)


# - It is possible to have retails shops more than 8000(due to high demand for product in some regions)
# - As there are many values in that range,they are not considered as outliers

# 4)Number of workers

# In[148]:


df.workers_num.describe()


# In[149]:


sns.distplot(df.workers_num)


# - High number of workers are only due to seasonality, which is rare
# - Hence workers more than 80(also from above boxplot) are considered as outliers

# In[150]:


Q1 = df['workers_num'].quantile(0.25)
Q2 = df['workers_num'].quantile(0.5)
Q3 = df['workers_num'].quantile(0.75)
IQR = Q3 - Q1
LW = Q1 - 1.5*IQR
UW = Q3 + 1.5*IQR

df['workers_num'] = np.where(df['workers_num'] > UW,UW,np.where(df['workers_num'] < LW,LW,df['workers_num']))


# In[151]:


sns.distplot(df.workers_num)


# After removing outliers

# ## Grouping and Binning

# In[152]:


df['weight_bins'] = pd.cut(df.product_wg_ton, bins = 5)


# In[153]:


tp_with_dist = df.groupby(['transport_issue_l1y'])
tp_with_dist = tp_with_dist['dist_from_hub'].sum()
print(tp_with_dist)


# In[154]:


sns.barplot(df.wh_breakdown_l3m   ,df.storage_issue_reported_l3m  )


# In[155]:


storage_with_breakdown = df.groupby(['wh_breakdown_l3m'])['storage_issue_reported_l3m'].mean()
storage_with_breakdown


# If breakdown is high storage issues are also increasing

# In[156]:


transport_issue_wg = df.groupby(['weight_bins'])['transport_issue_l1y'].sum()
transport_issue_wg


# In[157]:


fig = sns.barplot(df['transport_issue_l1y'],df['weight_bins'])
fig.set_xticklabels(['0','1','2','3','4','5','6','7','8'])


# ### Transport issues are more in the lowest range of product weight

# In[158]:


fig = sns.barplot(df.storage_issue_reported_l3m,df.weight_bins)
fig


# In[159]:


fig = sns.barplot(df.approved_wh_govt_certificate , df.weight_bins)
fig.set_xticklabels(['0','1','2','3','4','5'])


# In[160]:


fig = sns.barplot(df.wh_breakdown_l3m,df.weight_bins)
fig


# breakdown is high in all bins except for first bin

# In[161]:


sns.barplot(x = df.temp_reg_mach == 0 , y= df.weight_bins)


# In[162]:


sns.barplot(x = df.temp_reg_mach == 1 , y= df.weight_bins)


# ## Multicollinearity detection 

# - VIF to determine multicollinearity.
# - Variance Inflation Factor(VIF) formula VIF is = 1/(1-R2)
# - VIF is more than 5, we say that multicollinearity exists

# In[163]:


df1 = df.drop(['product_wg_ton','retails_distr'], axis = 1)
df1.info()


# In[164]:


df1.head()


# In[165]:


def calculate_vif(dataset):
    vif = pd.DataFrame()
    vif['features'] = dataset.columns
    vif['VIF values'] = [variance_inflation_factor(dataset.values, i) for i in range(dataset.shape[1])]
    vif['VIF values'] = round(vif['VIF values'],2)
    return vif


# In[166]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error


# In[167]:


# VIF for categorical columns is infinite 
calculate_vif(df1.select_dtypes(include = np.number))


# - No existence of multicollinearity(except wh_est_year)

# ## Model building

# In[168]:


y = df['product_wg_ton']


# In[169]:


# removing retails distributed due to multicollinearity


# In[170]:


X = df.drop(['product_wg_ton','retails_distr','weight_bins'], axis = 1)


# In[171]:


X.info()


# In[172]:


# model evaluation libraries
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#model building libraries
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import xgboost
from xgboost import XGBRegressor
#model selection libraries
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV,learning_curve


# In[173]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[174]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[175]:


def model_builder(model_name, model, data):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    result = [model_name, r2, rmse, mae]
    return result


# In[176]:


model_builder(model_name = 'LinearRegression', model = LinearRegression(), data = df)


# In[177]:


def multiple_models(data):
    col_names = ['Model Name','R2 Score','RMSE','MAE']
    result = pd.DataFrame(columns = col_names)
    result.loc[len(result)] = model_builder('Linear Regression',LinearRegression(), data)
    result.loc[len(result)] = model_builder('Lasso Regression', Lasso() , data)
    result.loc[len(result)] = model_builder('Ridge Regression', Ridge() , data)
    result.loc[len(result)] = model_builder('DTR', DecisionTreeRegressor() , data)
    result.loc[len(result)] = model_builder('Random Forest', RandomForestRegressor(), data)
    result.loc[len(result)] = model_builder('Gboost', GradientBoostingRegressor(), data)
    result.loc[len(result)] = model_builder('XGboost', XGBRegressor() , data)
    result.loc[len(result)] = model_builder('AdaBoost', AdaBoostRegressor() , data)
    
    return result.sort_values('R2 Score', ascending = False)


# In[178]:


# Test accuracy
multiple_models(df)


# In[179]:


#import scikitplot as skplt

#pca = PCA(random_state=1)
#pca.fit(X_train)

#skplt.decomposition.plot_pca_component_variance(pca, figsize=(8,6));


# **Learning curves using Linear Regression**

# In[181]:


import scikitplot as skplt
skplt.estimators.plot_learning_curve( LinearRegression(), X_train, y_train,
                                     cv=10, shuffle=True, scoring="r2", n_jobs=-1,
                                     figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                     title="Noodles weight Learning Curve ");


# - In case of R2 score the training error and testing error converge early 
# - which indicates the model is generalizing for the unseen data which is good.

# In[182]:


from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error

def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=10, scoring='neg_mean_squared_error')
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores_mean, label='Training set')
    plt.plot(train_sizes, test_scores_mean, label='Testing set')
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()


# In[183]:


estimator = LinearRegression()
plot_learning_curve(estimator, X_train, y_train)


# - Lower the gap between train and test curve lower the variance. 
# - The variance deacreases(curves converge) till 11000 samples and thereafter it diverges again.

# In[184]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_absolute_error

def plot_learning_curve2(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=10, scoring='neg_mean_absolute_error')
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(8, 4))
    plt.plot(train_sizes, train_scores_mean, label='Training set')
    plt.plot(train_sizes, test_scores_mean, label='Testing set')
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()


# In[185]:


estimator = LinearRegression()
plot_learning_curve2(estimator, X_train, y_train)


# - The training error has decrased first and then increased again till it converges with test error.
# - The test error decreases gradually and trend further is also decreasing, which is a good model.

# ## Lets check overfitting 

# ## Hyperparameter Tuning with cross validation

# In[188]:


def cv_post_hpt(X, y, fold =10):
    score_LR = cross_val_score(LinearRegression(), X, y , cv = fold)
    score_LS = cross_val_score(Lasso(alpha = 0.1), X, y , cv = fold)
    score_RD = cross_val_score(Ridge(alpha = 10), X, y , cv = fold)
    score_DTR = cross_val_score(DecisionTreeRegressor(max_depth = 12),X, y , cv = fold)
    score_RandomForest = cross_val_score(RandomForestRegressor(max_depth = 16), X, y , cv = fold)
   
    score_Gboost = cross_val_score(GradientBoostingRegressor(), X, y , cv = fold)
    score_XGboost = cross_val_score(XGBRegressor(eta = 0.1), X, y , cv = fold)
    score_AdaBoost = cross_val_score(AdaBoostRegressor(learning_rate = 0.8), X, y , cv = fold)
                
    
    
    
    
    model_name = ['linearRegression' , 'Lasso', 'Ridge', 'DTR', 'RandomForest', 
                  'GBoost', 'XGBoost', 'AdaBoost']
    score = [ score_LR , score_LS, score_RD , score_DTR, score_RandomForest,score_Gboost, score_XGboost, score_AdaBoost]
    result = []
    
    for i in range(len(model_name)):
        score_mean = np.mean(score[i])
        score_std = np.std(score[i])
        m_name = model_name[i]
        temp = [m_name , score_mean , score_std]
        result.append(temp)
    k_fold_df = pd.DataFrame(result , columns = ["Model Name" , 'CV Accuracy' , 'CV STD'])
    return k_fold_df.sort_values('CV Accuracy' , ascending = False)
        
              


# In[189]:


cv_post_hpt(X_train,y_train)


# XG Boost is the best model with 99.45% accuracy. 

# In[192]:


skplt.estimators.plot_learning_curve( GradientBoostingRegressor(), X_train, y_train,
                                    cv=10, shuffle=True, scoring="r2", n_jobs=-1,
                                    figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                    title="Noodles weight Learning Curve ")


# In[190]:


plot_learning_curve(GradientBoostingRegressor(), X_train, y_train)


# In[191]:


plot_learning_curve2(GradientBoostingRegressor(), X_train, y_train)


# In[193]:


for model, i in [(LinearRegression(),1),(Lasso(),2),(Ridge(),3),(DecisionTreeRegressor(),4),(RandomForestRegressor(),5),(GradientBoostingRegressor(),6),(XGBRegressor(),7),(AdaBoostRegressor(),8)]:
    plt.subplot(8,2,i)
  
    
    skplt.estimators.plot_learning_curve( model, X_train, y_train,
                                     cv=10, shuffle=True, scoring="r2", n_jobs=-1,
                                     figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                     title="Noodles weight Learning Curve ")
    
    plot_learning_curve(model, X_train, y_train)
    plot_learning_curve2(model, X_train, y_train)
    print("model_name: ",model )
    


# - Linear Regression, Lasso, Ridge perform in the manner of learning curve for all the three losses(R2 Score, MSE, MAE).
# - Decision tree and Random Forest Regressor are the models with low bias and high variance hence overfitting.
# - Gradient Boost is the best algorithm where R2score, MSE , MAE are performing in similar trend.
# - XGBoost is showing high variance which indicates overfitting.
# - AdaBoost show high bias and high variance and the worst performing model.

# In[199]:


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

sns.regplot(x =  y_test, y =  y_pred, scatter_kws={"color": "yellow"},  line_kws={"color": "red"})


# ### Feature importance using XG Boost

# In[237]:


import xgboost as xgb
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
xgb.plot_importance(model, importance_type='weight')
plt.show()


# In[236]:


X.info()


# **Recommendations**

# - Higher the number of retail shops, more noodles can be supplied to different parts of region and segments of market. Level of service of noodles to customers is also important.
# - Storage issues affect the delivery time of noodles to customers causing delay and may also affect the quality and safety of noodles.Hence special attention is needed to in managing high quantity.
# - Distance of warehouse affects the delivery time to shops sometimes due to weather and traffic.
# - Higher the number of distributors more choices to supply the product in time , hence able to meet the demand.
# - Recently(2020) established warehouse only can store high quantity of product. Hence old warehouses need to be renovated.
# - Approved warehouse certificate has good facilities to store high quantity of noodles

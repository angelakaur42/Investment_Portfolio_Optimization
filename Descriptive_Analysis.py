import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Load the DataFrame from the pickle file
df = pd.read_pickle("Yahoo-Finance-Scraper.pkl")


def absHigh(df, num):
    c = df.columns.values #stores column names
    a = np.abs(df.values) #converts dataframe values to a NumPy array and takes absolute values
    np.fill_diagonal(a, 0) #sets diagonal elements to 0
    i = (-a).argpartition(num, axis=None)[:num] #finds strongest negative relationships
    i, _ = np.unravel_index(i, a.shape) #converts linear indices to row and column indices within the dataframe
    i = sorted(i) #sort indices
    return df.loc[c[i],c[i]] #selects only columns and rows corresponding to the identified strongest negative relationships, forming a new DataFrame.


def selLow(df, num):
    c=df.columns.values
    a=np.abs(df.values)
    np.fill_diagonal(a,0)
    i=(a).argpartition(num, axis=None)[:num]
    i,_=np.unravel_index(i, a.shape)
    i=sorted(i)
    return df.loc[c[i], c[i]]

#displaying high correlation
corr=df.corr()
mat = absHigh(corr,8)
mask = np.triu(np.ones_like(mat))
fig, ax = plt.subplots(figsize=(20, 20))
ax.set_title("High Correlations", fontsize = 24)
sns.heatmap(mat, annot=True, mask=mask, cmap="viridis")
#plt.show()

#displaying low correlation
mat = selLow(corr,10) #select the 10 lowest correlations
mask = np.triu(np.ones_like(mat)) #creating a mask matrix with the same shape as mat and make the upper triangle True
fig, ax = plt.subplots(figsize=(20, 20)) #creating a figure and axis object for the plot
ax.set_title("Low Correlations", fontsize = 24)
sns.heatmap(mat, annot=True, mask=mask, cmap="viridis")
#plt.show()


#Calculating the Efficient Frontier
#calculate the annualized rate of return and covariance matrix

#annualized average return for each asset
#Annualized average return=Daily avg return*252 days
ra=np.mean(df,axis=0)*252 #(series data type)

#create a covariance matrix
covar=df.cov()*252

#calculate the annualized volatility (std) for each asset
vols=np.sqrt(np.diagonal(covar)) #covariance of a variable with itself is its variance

#create the weights array
# weights=np.concatenate([np.linspace(start=2,stop=1,num=200), 
#                         np.zeros(1600),
#                         np.linspace(start=-1, stop=-2, num=200)])


#Calculate the sharpe ratio for each asset
sr=(ra/vols).reset_index().rename(columns={0:'SR'})
sr['Rank0']=sr["SR"].rank(method="first", ascending=False).astype('int')-1 #ranking assets based on their sharpe ratios in descending order and assigns a numerical rank to each asset
#-1 adjusts rank to start from 0 (as opposed to 1 for easier indexing)
sr=sr.sort_values('Rank0')
#sr['weights']=weights



plt.figure(figsize=(16,6))
plt.title('All Stocks', fontsize=24)
plt.xlabel('Risk/Volatility')
plt.ylabel('Return')
plt.plot(vols, ra,'bo', markersize=2)
plt.show()

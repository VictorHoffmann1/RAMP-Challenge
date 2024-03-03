import numpy as np
import pandas as pd

def to_float(x):
  try : 
    return float(x)
  except :
    return float(x.replace(',', '.'))
def to_string(x):
  try : return str(x)
  except : return x
def transform_dataset(df):
    df['Date mutation'] = pd.to_datetime( df['Date mutation'], format='%d/%m/%Y')
    df['Year'] = df['Date mutation'].dt.year
    df['Month'] = df['Date mutation'].dt.month
    df['Day'] = df['Date mutation'].dt.day
    df = df.drop(columns=['Date mutation', 'Nature culture', 'Nature mutation'])
    df['Valeur fonciere'] = df['Valeur fonciere'].apply(lambda x : to_float(x))
    df['Code departement'] = df['Code departement'].apply(lambda x : to_string(x))
    df['Code commune'] = df['Code commune'].apply(lambda x : to_string(x))
    df['Nombre pieces principales'] = df['Nombre pieces principales'].fillna(0)
    df['Nombre pieces principales'] = df['Nombre pieces principales'].apply(lambda x : int(x))
    df['Surface terrain'] = df['Surface terrain'].apply(lambda x : float(x))
    df['Surface reelle bati'] = df['Surface reelle bati'].apply(lambda x : float(x))
    df.drop(df[df['Surface terrain'] <= 9].index, inplace = True)
    df.drop(df[df['Valeur fonciere'] <= 10].index, inplace = True)
    df['Prix m2'] = df['Valeur fonciere']/df['Surface terrain']
    df['Proportion terrain bati'] = df['Surface reelle bati']/df['Surface terrain']
    df.drop(df[df['Prix m2'] <= 0.1].index, inplace = True)
    df.drop_duplicates(subset=['Valeur fonciere', 'Code departement'], inplace=True)
    return df

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train = transform_dataset(train)
test = transform_dataset(test)

train['Code departement'] = train['Code departement'].replace({'2A': '2', '2B': '2'})
test['Code departement'] = test['Code departement'].replace({'2A': '2', '2B': '2'})

train.dropna(inplace=True)
test.dropna(inplace=True)

train['Code commune'] = pd.to_numeric(train['Code commune'], errors='coerce')
test['Code commune'] = pd.to_numeric(test['Code commune'], errors='coerce')

train = pd.get_dummies(train, columns=['Type local'])
test = pd.get_dummies(test, columns=['Type local'])
train.to_csv('data/train.csv')
test.to_csv('data/test.csv')
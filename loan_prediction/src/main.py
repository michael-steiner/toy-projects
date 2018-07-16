import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data
loan_train = pd.read_csv("data/train_data.csv", header=0)
loan_test = pd.read_csv("data/test_data.csv", header=0)


# peeking at the data
peek = False
if peek:
    print(loan_test.head())
    print(loan_train.describe())
    print(loan_train['Property_Area'].value_counts())

# plotting histogram
hist_plot = False
if hist_plot:
    plt.show(loan_train['ApplicantIncome'].hist(bins=50))
    plt.show(loan_train.boxplot(column='ApplicantIncome'))
    plt.show(loan_train.boxplot(column='ApplicantIncome', by = 'Education'))
    plt.show(loan_train['LoanAmount'].hist(bins=50))
    plt.show(loan_train.boxplot(column='LoanAmount'))


# Categorical variable analysis

# pivot table
pivot = False
if pivot:
    temp_1 = loan_train['Credit_History'].value_counts(ascending=True)
    temp_2 = loan_train.pivot_table(values='Loan_Status', index=['Credit_History'], aggfunc=lambda x: x.map({'Y': 1, 'N': 0}).mean())
    print('Frequency Table for Credit History')
    print(temp_1)
    print('\nProbility of getting loan for each Credit History class:' )
    print(temp_2)
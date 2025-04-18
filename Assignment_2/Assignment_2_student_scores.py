#!/usr/bin/env python
import csv
import numpy as np
from datetime import datetime
import pandas as pd

def get_student_list(csv_file_name):
    with open(csv_file_name, 'r') as f:
        reader = csv.reader(f)
        column_data = next(reader)
        colmap = dict(zip(column_data, range(len(column_data))))
        # print(colmap)
        colmap_list = [None] * (max(colmap.values()) + 1)
        ## code to change colmap to a list with student_names as values in the list indexes by the column number
        ## e.g. colmap[0] = 'Name', colmap[1] = 'Subject1', colmap[2] = 'Subject2', colmap[3] = 'Subject3'.
        for k, v in colmap.items():
            # print(k)
            # print(v)
            colmap_list[v] = k        
        # print(colmap_list)
    return colmap, colmap_list[1:]

### ASSIGNMENT 2.1
# Function that reads the CSV file and returns subject wise topper(s) and overall topper(s), such that:
# Toppers shall have at least 60% attendance
# Toppers shall have the project submitted.
## Data from file - student_scores.csv
def get_toppers(csv_file='students.csv'):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert 'Attendance (%)' to float and filter
    df['Attendance (%)'] = pd.to_numeric(df['Attendance (%)'], errors='coerce')
    df_filtered = df[(df['Attendance (%)'] >= 60) & (df['Project Submitted'] == True)].copy()

    # List of subjects
    subjects = ['Math', 'Science', 'English']
    
    # Subject-wise toppers
    subject_toppers = {}
    for subject in subjects:
        max_score = df_filtered[subject].max()
        toppers = df_filtered[df_filtered[subject] == max_score]['Name'].tolist()
        subject_toppers[subject] = toppers

    # Overall topper(s): sum of all subjects
    df_filtered.loc[:, 'Total'] = df_filtered[subjects].sum(axis=1)
    max_total = df_filtered['Total'].max()
    overall_toppers = df_filtered[df_filtered['Total'] == max_total]['Name'].tolist()

    return subject_toppers, overall_toppers

### ASSIGNMENT 2.2
# A function that returns a data frame with the following columns added along with original data:
# ‘Average Score’ : For each student
# ‘Grade’ :  based on average score (A : >= 90; B : 75 .. 89.99; C : 60 .. 74.99; D : <60)
# ‘Performance’ : 
# ‘Excellent’ : Grade A and attendance > 90%, project submitted
# ‘Needs Attention’ : Grade D OR project not submitted OR attendance < 60%
# ‘Satisfactory’ : All others

def add_student_performance(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Ensure numeric columns are treated as numbers
    subjects = ['Math', 'Science', 'English']
    for subj in subjects:
        df[subj] = pd.to_numeric(df[subj], errors='coerce')
    df['Attendance (%)'] = pd.to_numeric(df['Attendance (%)'], errors='coerce')
    
    # Calculate Average Score
    df['Average Score'] = df[subjects].mean(axis=1)
    
    # Assign Grade
    def get_grade(avg):
        if avg >= 90:
            return 'A'
        elif avg >= 75:
            return 'B'
        elif avg >= 60:
            return 'C'
        else:
            return 'D'
    df['Grade'] = df['Average Score'].apply(get_grade)
    
    # Assign Performance
    def get_performance(row):
        if (row['Grade'] == 'A' and row['Attendance (%)'] > 90 and row['Project Submitted'] == True):
            return 'Excellent'
        elif (row['Grade'] == 'D' or row['Project Submitted'] == False or row['Attendance (%)'] < 60):
            return 'Needs Attention'
        else:
            return 'Satisfactory'
    df['Performance'] = df.apply(get_performance, axis=1)
    
    return df



def export_summary_statistics(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Ensure numerical columns are treated as numbers
    columns = ['Math', 'Science', 'English', 'Attendance (%)']
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Compute summary statistics
    summary = df[columns].agg(['mean', 'median', 'min', 'max', 'std'])
    
    # Reset index to turn the statistic names into a regular column
    summary = summary.reset_index().rename(columns={'index': 'Statistic'})
    
    # Export to CSV
    summary.to_csv(output_csv, index=False)
    print(f"Summary statistics exported to {output_csv}")


## 2.1 usage
csv_file_name = 'student_scores.csv'
subject_toppers, overall_toppers = get_toppers(csv_file_name)
print("Subject-wise toppers:", subject_toppers)
print("Overall toppers:", overall_toppers)

print("--------------------")
# 2.2 usage:
enhanced_df = add_student_performance(csv_file_name)
print(enhanced_df.head())

print("--------------------")

# 2.3 usage:
export_summary_statistics(csv_file_name, 'summary_statistics.csv')

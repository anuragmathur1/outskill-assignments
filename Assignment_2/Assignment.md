## Assignment 2

The file ‘student_scores.csv’ provides sample data of students and their test score along with attendance and project status. This is available in a CSV file. Use this content and create a python program (using pandas) for the following requirements. Make them as functions in a module.

Function that reads the CSV file and returns subject wise topper(s) and overall topper(s), such that:
Toppers shall have at least 60% attendance
Toppers shall have the project submitted.

A function that returns a data frame with the following columns added along with original data:
‘Average Score’ : For each student
‘Grade’ :  based on average score (A : >= 90; B : 75 .. 89.99; C : 60 .. 74.99; D : <60)
‘Performance’ : 
‘Excellent’ : Grade A and attendance > 90%, project submitted
‘Needs Attention’ : Grade D OR project not submitted OR attendance < 60%
‘Satisfactory’ : All others

A function that exports the summary statistics of the subject wise marks, attendance to a CSV file.

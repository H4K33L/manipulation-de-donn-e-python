import pandas as pd 
df = pd.read_csv("titanic/train.csv")
from ydata_profiling import ProfileReport
 
profile = ProfileReport(df, title="Profiling Report")
profile.to_file("your_report.html")
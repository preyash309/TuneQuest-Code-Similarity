import pandas as pd
import numpy as np

# Load the probabilities you already calculated
df = pd.read_csv("test_probabilities_tta.csv")

# ==========================================
# CHANGE THIS TO TEST NEW LEADERBOARD SCORES
# ==========================================
NEW_THRESHOLD = 0.37

# Apply new threshold
df['prediction'] = (df['p_final'] >= NEW_THRESHOLD).astype(int)

# Save the new submission
sub_filename = f"submission_thresh_{NEW_THRESHOLD:.2f}.csv"
df[['id', 'prediction']].rename(columns={'prediction': 'label'}).to_csv(sub_filename, index=False)

print(f"Generated {sub_filename}!")
print(df['prediction'].value_counts())
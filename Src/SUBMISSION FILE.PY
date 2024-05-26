import pandas as pd

# Load the test data
test_data = pd.read_csv(r"C:\Users\91800\Desktop\Health care\data\test_data.csv")

# Assuming predictions is a list of predicted LOS classes for the test data
# This would typically come from your trained model's predictions
predictions = [0] * len(test_data)  # Placeholder for actual predictions

# Create a DataFrame for the submission
submission = pd.DataFrame({
    'case_id': test_data['case_id'],
    'Stay': predictions
})

# Save the submission file

submission.to_csv('C:\\Users\\91800\\Desktop\\Health care\\data\\submission.csv', index=False)
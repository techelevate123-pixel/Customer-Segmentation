import pandas as pd
import numpy as np

# for reproducibility
np.random.seed(42)

# number of customers
n = 300

# create dataset
df = pd.DataFrame({"age": np.random.randint(18,70,n), "income": np.random.randint(20000,120000,n), "spending_score": np.random.randint(1,100,n), "family_size": np.random.randint(1,6,n)})

# save files
df.to_csv("data/customers.csv", index=False)

print("dataset created successfully!")
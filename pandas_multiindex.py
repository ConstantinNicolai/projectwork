import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create a multi-index
index = pd.MultiIndex.from_product(
    [range(3), range(3), range(3)],
    names=['dim1', 'dim2', 'dim3']
)

# Create sample data
data = np.random.rand(27)

# Create DataFrame
df = pd.DataFrame(data, index=index, columns=['value'])

print(df)

# Reset index to convert MultiIndex to columns for easier manipulation
df_reset = df.reset_index()

# Example of data manipulation: calculating mean values grouped by dim1 and dim2
mean_values = df_reset.groupby(['dim1', 'dim2']).mean().reset_index()

# Pivot the data for plotting
pivot_table = mean_values.pivot(index='dim1', columns='dim2', values='value')

# Plotting the data
pivot_table.plot(kind='bar')
plt.ylabel('Mean Value')
plt.title('Mean Values by dim1 and dim2')
plt.show()

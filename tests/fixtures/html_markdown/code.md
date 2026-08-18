# Code Example

The following Python snippet applies a window function to a DataFrame and is followed by an SQL variant for comparison.

```python
from pyspark.sql import Window
from pyspark.sql.functions import row_number

w = Window.partitionBy("dept").orderBy("salary")
df = df.withColumn("rn", row_number().over(w))
```

```sql
SELECT dept, salary,
       row_number() OVER (PARTITION BY dept ORDER BY salary) AS rn
FROM employees;
```

Both variants produce identical row numbering over the same logical window definition, which is the expected behavior for this documentation example.
# Window Functions

## Ranking

Use `dense_rank` with a partition and ordering window.

```python
from pyspark.sql import Window
from pyspark.sql import functions as F

window = Window.partitionBy("customer_id", "category").orderBy(F.col("amount").desc())
ranked = sales_df.withColumn("amount_rank", F.dense_rank().over(window))
```

# Window Functions

Introductory paragraph that sets up the section and references `row_number()` inline for prose consistency.

## Ranking

Use `dense_rank` when you want no gaps in the rank sequence.

```python
from pyspark.sql import Window
w = Window.partitionBy("dept").orderBy("salary")
```

### Example Output

| Function | Output |
| --- | --- |
| row_number | 1, 2, 3 |
| rank | 1, 2, 2 |
| dense_rank | 1, 2, 2 |

## Frames

Frames bound the window using `rangeBetween` and `rowsBetween`.

```sql
SELECT SUM(amount) OVER (
  ORDER BY ts ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
) FROM sales;
```

The frame clause is applied after partitioning in the query plan.

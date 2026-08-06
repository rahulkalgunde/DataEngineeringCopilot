from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()
orders = spark.createDataFrame(
    [
        ("o1", [{"item_id": "i1", "price": 10.0, "discount": 0.10}]),
        ("o2", [{"item_id": "i2", "price": 20.0, "discount": 0.30}]),
    ],
    schema="order_id STRING, items ARRAY<STRUCT<item_id: STRING, price: DOUBLE, discount: DOUBLE>>",
)

filtered = orders.withColumn(
    "eligible_items",
    F.filter("items", lambda item: item.discount <= F.lit(0.20)),
)
result = filtered.withColumn(
    "net_total",
    F.aggregate(
        "eligible_items",
        F.lit(0.0),
        lambda total, item: total + item.price * (1.0 - item.discount),
    ),
)
result.show()

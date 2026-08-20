# Table Reference

This guide describes how window function results are laid out in result tables for the documentation set.

| Function | Partition | Order |
| --- | --- | --- |
| row_number | dept | salary desc |
| rank | dept | salary asc |
| dense_rank | team | hire_date |

Every function above assigns a stable rank to each row within its partition window and is commonly used in reporting pipelines.

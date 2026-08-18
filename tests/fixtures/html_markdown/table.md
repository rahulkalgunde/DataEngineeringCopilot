# Table Reference

This guide describes how window function results are laid out in result tables for the documentation set.

| Function | Partition | Order |
| --- | --- | --- |
| row\_number | dept | salary desc |
| rank | dept | salary asc |
| dense\_rank | team | hire\_date |

Every function above assigns a stable rank to each row within its partition window and is commonly used in reporting pipelines.
# Malformed Page

This page contains unclosed and mis-nested tags to prove the converter keeps its behavior deterministic in the face of broken markup.

**Unclosed bold *and italic*** text still renders.

| | |
| --- | --- |
| Orphan cell without header | Second cell |

A paragraph opened but never closed plus a stray attribute value below this line.

```python
print("balanced")
```
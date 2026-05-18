import sys
print("A")
try:
    import torch
    print("B")
except Exception as e:
    print(e)

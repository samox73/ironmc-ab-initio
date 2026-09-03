#!/usr/bin/env python

n = 8

print("K_POINTS crystal")
print(n**3)
for i in range(n):
    for j in range(n):
        for k in range(n):
            print(f"  {i/n:.8f} {j/n:.8f} {k/n:.8f} {1/n**3:.9e}")

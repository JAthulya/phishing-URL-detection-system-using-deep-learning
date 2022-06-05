import pandas as pd

df = pd.read_csv('features1.csv')
a=0
b=0
c=0
d=0
e = 0
for i in df.cert_by:
    if e < 401:
        if i == 1:
            a = a + 1
        elif i == 0:
            b = b + 1
    elif e > 400:
        if i == 1:
            c = c + 1
        elif i ==0:
            d = d + 1
    e = e + 1
            
print("feature detected in legitimate: " + str(a))
print("feature not detected in legitimate: " + str(b))
print("feature detected in phishing :" + str(c))
print("feature not detected in phishing :" + str(d))
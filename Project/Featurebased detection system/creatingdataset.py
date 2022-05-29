import pandas as pd
from urllib.parse import urlparse, urlencode
import ipaddress
import re
import requests
from bs4 import BeautifulSoup
import whois
import urllib
import urllib.request
from datetime import datetime
from urllib.parse import urlparse, urlencode

df = pd.read_csv('dataset.csv')

# def havingIP(url):
#   try:
#     ipaddress.ip_address(url)
#     ip = 1
#   except:
#     ip = 0
#   return ip


# list=[]
# for i in df.url:
#     ip = havingIP(i)
#     list.append(ip)
# df['Have_IP']=list
# df.to_csv('features.csv',index=False)

# def havingAtsign(url):
#     if "@" in url:
#         at = 1
#     else:
#         at = 0
#     return at

# list=[]
# for i in df.url:
#     at = havingAtsign(i)
#     list.append(at)
# df['Have_At']=list
# df.to_csv('test1.csv',index=False)

# def getLength(url):
#   if len(url) < 54:
#     length = 0
#   else:
#     length = 1
#   return length

# #phishing
# list = []
# for i in df.url:
#     l = getLength(i)
#     list.append(l)
# df['URL_Length']=list
# df.to_csv('test1.csv',index=False)

# def getDepth(url):
#   s = urlparse(url).path.split('/')
#   depth = 0
#   for j in range(len(s)):
#     if len(s[j]) != 0:
#       depth = depth+1
#   return depth

# list=[]
# for i in df.url:
#     d= getDepth(i)
#     list.append(d)
# df['URL_Depth']=list
# df.to_csv('test1.csv',index=False)

# def redirection(url):
#     pos = url.rfind('//')
#     if pos > 6:
#         if pos > 7:
#             return 1
#         else:
#             return 0
#     else:
#         return 0

# list=[]
# for i in df.url:
#     p = redirection(i)
#     list.append(p)
# df['Redirection']=list
# df.to_csv('test1.csv',index=False)

# https in domain
# def httpDomain(url):
#   domain = urlparse(url).netloc
#   if 'https' in domain:
#     return 1
#   else:
#     return 0

# list=[]
# for i in df.url:
#     d = httpDomain(i)
#     list.append(d)
# df['https_Domain']=list
# df.to_csv('test1.csv',index=False)

# shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
#                       r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
#                       r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
#                       r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
#                       r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
#                       r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
#                       r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
#                       r"tr\.im|link\.zip\.net"
# def tinyURL(url):
#     match=re.search(shortening_services,url)
#     if match:
#         return 1
#     else:
#         return 0

# list=[]
# for i in df.url:
#     t = tinyURL(i)
#     list.append(t)
# df['TinyURL']=list
# df.to_csv('test1.csv',index=False)

# def prefixSuffix(url):
#     if '-' in urlparse(url).netloc:
#         return 1
#     else:
#         return 0

# list=[]
# for i in df.url:
#     p = prefixSuffix(i)
#     list.append(p)
# df['Prefix/Suffix']=list
# df.to_csv('test1.csv',index=False)

# def dnsrecord(url):
#     dns = 0
#     try:
#         domain_name = whois.whois(url)
#         if domain_name.get('domain_name') == None:
#             dns = 1
#     except:
#         dns =1
#     return dns
        
# list=[]
# for i in df.url:
#     dns = dnsrecord(i)
#     list.append(dns)
# df['DNS_Record']=list
# df.to_csv('test1.csv',index=False)

# def ageofdomain(url):
#     try:
#         whois_info = whois.whois(url)
#         if whois_info.get('domain_name')== None:
#             days = 1
#         ex = whois_info.get('expiration_date')
#         cd = whois_info.get('creation_date')
#         expiration_date = ex[0]
#         creation_date = cd[0]
#         days = abs((expiration_date - creation_date).days)
#         if days > 360:
#             days = 0
        
#         return days
#     except :
#         pass
#     try:
#         whois_info = whois.whois(url)
#         if whois_info.get('domain_name')== None:
#             days = 1
#         ex = whois_info.get('expiration_date')
#         cd = whois_info.get('creation_date')
#         expiration_date = ex
#         creation_date = cd
#         days = abs((expiration_date - creation_date).days)
#         if days > 360:
#             days = 0
        
#         return days
#     except:
#         pass
    
#     try:
#         whois_info = whois.whois(url)
#         if whois_info.get('domain_name')== None:
#             days = 1
#         ex = whois_info.get('expiration_date')
#         cd = whois_info.get('creation_date')
#         expiration_date = ex
#         creation_date = cd[0]
#         days = abs((expiration_date - creation_date).days)
#         if days > 360:
#             days = 0
        
#         return days
#     except:
#         pass
#     try:
#         whois_info = whois.whois(url)
#         if whois_info.get('domain_name')== None:
#             days = 1
#         ex = whois_info.get('expiration_date')
#         cd = whois_info.get('creation_date')
#         expiration_date = ex[0]
#         creation_date = cd
#         days = abs((expiration_date - creation_date).days)
#         if days > 360:
#             days = 0
        
#         return days
#     except:
#         return 1

# def domaintime(url):
#     try:
#         whois_info = whois.whois(url)
#         if whois_info.get('domain_name')== None:
#             day = 1
#         ex = whois_info.get('expiration_date')
#         expiration_date = ex[0] 
#         nt = datetime.now()
#         days = abs((expiration_date - nt).days)
#         if days > 180:
#             day = 0
#         else:
#             day = 1
#         return day
#     except:
#         pass
    
#     try:
#         whois_info = whois.whois(url)
#         if whois_info.get('domain_name')==None:
#             day = 1
#         ex = whois_info.get('expiration_date')
#         expiration_date = ex
#         nt = datetime.now()
#         days = abs((expiration_date - nt).days)
#         if days > 180:
#             day = 0
#         else:
#             day = 1
#         return day
   
#     except:
#         return 1
    
# list=[]
# for i in df.url:
#     dns = domaintime(i)
#     list.append(dns)
# df['Domain_time']=list
# df.to_csv('test1.csv',index=False)

# def iframe(response):
#     if response == '':
#         return 1
#     else:
#         if re.findall((r'[<iframe>|<frameBorder>]'), response.text):
#             return 0
#         else:
#             return 1

# list=[]
# for i in df.url:
#     try:
#         response = requests.get(i)
#     except:
#         response = ''
#     frame = iframe(response)
#     list.append(frame)
# df['iFrame']=list
# df.to_csv('test1.csv',index=False)

# def mouseOver(response):
#     if response == '':
#         return 1
#     else:
#         if re.findall("<script>.+onmouseover.+</script>", response.text):
#             return 1
#         else:
#             return 0

# list=[]
# for i in df.url:
#     try:
#         response = requests.get(i)
#     except:
#         response = ''
#     fake = mouseOver(response)
#     list.append(fake)
# df['Mouse_Over'] = list
# df.to_csv('test1.csv',index=False)

def forwading(response):
    if response == '':
        return 1
    else:
        if len(response.history)<= 2:
            return 0
        else:
            return 1
        

list= []
for i in df.url:
    try:
        response = requests.get(i)
    except:
        response = ''
    forward = forwading(response)
    list.append(forward)
df['Web_Forwards'] = list
df.to_csv('test1.csv',index=False)
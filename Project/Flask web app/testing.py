import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from urllib.parse import urlparse, urlencode
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import whois
from datetime import datetime
import urllib.request
import requests

def ageofdomain(url):
    try:
        whois_info = whois.whois(url)
        if whois_info.get('domain_name')== None:
            days = 1
        ex = whois_info.get('expiration_date')
        cd = whois_info.get('creation_date')
        expiration_date = ex[0]
        creation_date = cd[0]
        days = abs((expiration_date - creation_date).days)
        if days > 360:
            days = 0
        
        return days
    except :
        pass
    try:
        whois_info = whois.whois(url)
        if whois_info.get('domain_name')== None:
            days = 1
        ex = whois_info.get('expiration_date')
        cd = whois_info.get('creation_date')
        expiration_date = ex
        creation_date = cd
        days = abs((expiration_date - creation_date).days)
        if days > 360:
            days = 0
        
        return days
    except:
        pass
    
    try:
        whois_info = whois.whois(url)
        if whois_info.get('domain_name')== None:
            days = 1
        ex = whois_info.get('expiration_date')
        cd = whois_info.get('creation_date')
        expiration_date = ex
        creation_date = cd[0]
        days = abs((expiration_date - creation_date).days)
        if days > 360:
            days = 0
        
        return days
    except:
        pass
    try:
        whois_info = whois.whois(url)
        if whois_info.get('domain_name')== None:
            days = 1
        ex = whois_info.get('expiration_date')
        cd = whois_info.get('creation_date')
        expiration_date = ex[0]
        creation_date = cd
        days = abs((expiration_date - creation_date).days)
        if days > 360:
            days = 0
        
        return days
    except:
        return days
    
def domaintime(url):
    try:
        whois_info = whois.whois(url)
        if whois_info.get('domain_name')== None:
            day = 1
        ex = whois_info.get('expiration_date')
        expiration_date = ex[0] 
        nt = datetime.now()
        days = abs((expiration_date - nt).days)
        if days > 180:
            day = 0
        else:
            day = 1
        return day
    except:
        pass
    
    try:
        whois_info = whois.whois(url)
        if whois_info.get('domain_name')==None:
            day = 1
        ex = whois_info.get('expiration_date')
        expiration_date = ex
        nt = datetime.now()
        days = abs((expiration_date - nt).days)
        if days > 180:
            day = 0
        else:
            day = 1
        return day
   
    except:
        return 1

def dnsrecord(url):
    dns = 0
    try:
        domain_name = whois.whois(url)
        if domain_name.get('domain_name') == None:
            dns = 1
    except:
        dns =1
    return dns

          
url = input('enter the url= ')
response = requests.get(url)
result = re.search('<iframe>|<frameBorder>',response.text)
#print(result)
#print(response.text)
print(whois.whois(url))
# print(datetime.now())
print(ageofdomain(url), dnsrecord(url),domaintime(url))

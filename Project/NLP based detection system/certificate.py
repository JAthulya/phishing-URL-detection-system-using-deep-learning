import ssl, socket
from urllib.parse import urlparse, urlencode

messages = 'https://www.google.com'
hostname = urlparse(messages).netloc
print(hostname)
ctx = ssl.create_default_context()
with ctx.wrap_socket(socket.socket(), server_hostname=hostname) as s:
    s.connect((hostname, 443))
    cert = s.getpeercert()

subject = dict(x[0] for x in cert['subject'])
issued_to = subject['commonName']
issuer = dict(x[0] for x in cert['issuer'])
issued_by = issuer['commonName']

print(issued_to)
print(issued_by)

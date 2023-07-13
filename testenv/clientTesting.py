import http.client

conn = http.client.HTTPConnection('localhost', 8881)
conn.request("GET", "/")
response = conn.getresponse()
print(response.status, response.reason)
data = response.read()
print(data.decode())
conn.close()
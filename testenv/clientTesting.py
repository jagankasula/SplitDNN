import http.client
import time

def measure_latency():
    conn = http.client.HTTPConnection('KC-SCER-FCCT0M3', 8881)
    start_time = time.time()
    conn.request("GET", "/")
    response = conn.getresponse()
    latency = time.time() - start_time

    print(f"Response Status: {response.status} {response.reason}")
    print(f"Response Data: {response.read().decode()}")
    print(f"Network Latency: {latency:.6f} seconds")

    conn.close()

if __name__ == "__main__":
    measure_latency()

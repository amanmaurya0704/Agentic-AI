import requests

with open("valid_proxies.txt") as f:
    proxies = f.read().split("\n")

product = ["https://www.amazon.com/Smartphone-Unlocked-Processor-Manufacturer-Warranty/dp/B0DP3G4GVQ/"]


counter = 0
for p in range(len(proxies)):
    try:
        print(f"using the proxy: {proxies[counter]}")
        print(counter)
        res = requests.get(product,proxies={"http":proxies[counter],
                                            "https":proxies[counter]})
        
        print(res.status_code)
        break

    except:
        print("Failed")

    finally:
        counter += 1
        counter % len(proxies)

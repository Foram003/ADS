import requests

url = "http://localhost:5000/api"

test_data = {
    "buying_low": 0.0,
    "buying_med": 1.0,
    "buying_vhigh": 0.0,
    "maint_low": 0.0,
    "maint_med": 0.0,
    "maint_vhigh": 1.0,
    "doors_3": 0.0,
    "doors_4": 1.0,
    "doors_5more": 0.0,
    "persons_4": 1.0,
    "persons_more": 0.0,
    "lug_boot_med": 0.0,
    "lug_boot_small": 0.0,
    "safety_low": 1.0,
    "safety_med": 0.0,
}

# 'unacc'

r = requests.post(url, json=test_data)
print("Predicted class: ",r.json())

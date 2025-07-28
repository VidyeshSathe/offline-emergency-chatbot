import requests, zipfile, os

url = "https://www.dropbox.com/scl/fi/sxqa3qh9unvn3etv4rwcq/models.zip?rlkey=1oqzd6ybbyd84s6hgs0gvp97y&dl=1"
output = "models.zip"
target_dir = "models"

print(" Downloading models from Dropbox...")
r = requests.get(url)
with open(output, "wb") as f:
    f.write(r.content)

print(" Download complete. Extracting...")
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(target_dir)

os.remove(output)
print(f" Model setup complete. Files ready in ./{target_dir}")

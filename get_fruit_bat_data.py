#download data for An annotated dataset of Egyptian fruit bat vocalizations across varying contexts and during vocal ontogeny (https://www.nature.com/articles/sdata2017143)
import requests
import json
import wget

url= "https://api.figshare.com/v2/collections/3666502/articles?page_size=70"
response = requests.get(url)
data = response.text
data = json.loads(data)
for file in data:
    title = file["title"]
    print(title)
    url = file["url"]
    file_data = json.loads((requests.get(url)).text)
    wget.download(file_data["files"]["download_url"])
    print(f"{title} downloaded")
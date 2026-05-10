import json
import re
from pathlib import Path

def extract_center(xml_path):
    try:
        with open(xml_path, 'r') as f:
            content = f.read()
        match = re.search(r'<EXT_POS_LIST>(.*?)</EXT_POS_LIST>', content)
        if match:
            coords = [float(x) for x in match.group(1).split()]
            lats = coords[0::2]
            lons = coords[1::2]
            return sum(lats)/len(lats), sum(lons)/len(lons)
    except:
        pass
    return 0.0, 0.0

def update_labels():
    processed_path = Path("data/processed/dataset_labels.json")
    if not processed_path.exists():
        return
        
    with open(processed_path, 'r') as f:
        data = json.load(f)
        
    for item in data:
        folder = Path(item["folder"])
        # Find any XML that might have footprint
        xmls = list(folder.rglob("MTD_MSIL*.xml")) + list(folder.rglob("INSPIRE.xml"))
        if xmls:
            lat, lon = extract_center(xmls[0])
            item["lat"] = lat
            item["lon"] = lon
            
    with open(processed_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Updated {len(data)} items with coordinates.")

if __name__ == "__main__":
    update_labels()

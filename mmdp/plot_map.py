import pandas as pd
import folium

if __name__ == '__main__':

    folium_map = folium.Map(location=[40.738, -73.98],
                            zoom_start=13,
                            tiles="CartoDB dark_matter")

    folium.CircleMarker(location=[40.738, -73.98],fill=True).add_to(folium_map)
    folium_map


    bike_data  = pd.read_csv("/home/lucifer/Documents/Git/MMDP/data/13_01_01_sorted.csv")
    # bike_data["Start Time"] = pd.to_datetime(bike_data["Start Time"])
    # bike_data["Stop Time"] = pd.to_datetime(bike_data["Stop Time"])
    # bike_data["hour"] = bike_data["Start Time"].map(lambda x: x.hour)

    print(bike_data.head())
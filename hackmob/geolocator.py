import requests
import json
import numpy as np

class Geolocator:

    url_current_location = 'https://www.googleapis.com/geolocation/v1/geolocate'
    api_key = 'AIzaSyD-OH0UooW7Wu19JVv1THCNHYHWqvl2IC8'
    url_gas_stations = 'https://sedeaplicaciones.minetur.gob.es/ServiciosRESTCarburantes/PreciosCarburantes/EstacionesTerrestres/FiltroMunicipio/881'

    @staticmethod
    def current_location():
        request_url = Geolocator.url_current_location + '?key=' + Geolocator.api_key
        req = requests.post(request_url)
        req_json = json.loads(req.text)
        return (req_json['location']['lat'], req_json['location']['lng'])
    
    @staticmethod
    def distance(latitude1, longitude1, latitude2, longitude2):
        a = np.power(np.sin(np.deg2rad(abs(latitude1-latitude2)/2)),2) + np.cos(np.deg2rad(latitude1))*np.cos(np.deg2rad(latitude2))*np.power(np.sin(np.deg2rad(abs(longitude1-longitude2)/2)),2)
        c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
        d = Geolocator.R*c
        return d

    @staticmethod
    def euclidean_distance(latitude1, longitude1, latitude2, longitude2):
        d = np.sqrt(np.power(latitude2 - latitude1, 2) + np.power(longitude2 - longitude1, 2))
        return d

    @staticmethod
    def get_contour_circ(latitude1, longitude1, latitude2, longitude2):
        origin_latitude = (latitude1 + latitude2) / 2
        origin_longitude = (longitude1 + longitude2) / 2
        rad = np.sqrt(np.power(longitude2 - origin_longitude, 2) + np.power(latitude2 - origin_latitude, 2))
        return origin_latitude, origin_longitude, rad

    @staticmethod
    def is_inside_contour_circ(origin_latitude, origin_longitude, r, latitude, longitude):
        dist = Geolocator.euclidean_distance(origin_latitude, origin_longitude, latitude, longitude)
        return dist <= r

    @staticmethod
    def get_gas_stations_list():
        req = requests.get(Geolocator.url_gas_stations)
        file_json = json.loads(req.text)
        gas_stations_list = []
        for el in file_json['ListaEESSPrecio']:
            current_el = {}
            current_el['name'] = el['Rótulo']
            current_el['lat'] = el['Latitud'].replace(',', '.')
            current_el['long'] = el['Longitud (WGS84)'].replace(',', '.')
            current_el['address'] = el['Dirección']
            gas_stations_list.append(current_el)
        return gas_stations_list

    @staticmethod
    def get_gas_stations_route(latitude1, longitude1, latitude2, longitude2):
        stations_route = []
        gas_stations_list =Geolocator.get_gas_stations_list()
        origin_lat, origin_long, rad = Geolocator.get_contour_circ(latitude1, longitude1, latitude2, longitude2)
        for gs in gas_stations_list:
            inside = Geolocator.is_inside_contour_circ(origin_lat, origin_long, rad, float(gs['lat']), float(gs['long']))
            if inside:
                stations_route.append(gs)
        if stations_route == []:
            return None
        else:
            return stations_route
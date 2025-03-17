import time
import numpy as np
import requests
from phue import Bridge, Light


def get_bridge_ip():
    response = requests.get('https://discovery.meethue.com/')
    bridges = response.json()[0]

    if bridges:
        id = bridges['id']
        ip = bridges['internalipaddress']
        port = bridges['port']

        return id, ip, port
    else:
        print("No bridges found.")
        return 0


def get_light_with_name(lights: list[Light], name: str):
    for light in lights:
        if light.name == name:
            return light


def kelvin_cycle(light: Light):
    min_kelvin = 2000
    max_kelvin = 6000

    kelvin_range = np.linspace(min_kelvin, max_kelvin, num=21, dtype=int)

    light.on = True
    time.sleep(0.5)

    for kelvin in kelvin_range:
        print(kelvin)
        light.colortemp_k = kelvin
        time.sleep(0.5)

    light.on = False


if __name__ == '__main__':
    id, ip, port = get_bridge_ip()

    bridge = Bridge(ip)
    bridge.connect()

    lamp_name = 'Lampe mod tv'
    light = get_light_with_name(bridge.lights, lamp_name)

    kelvin_cycle(light)

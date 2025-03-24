import math
import time
from typing import Any

import colour
import requests
import numpy as np
import pandas as pd
from numpy import ndarray, dtype
from phue import Bridge, Light


def get_bridge_ip() -> tuple[str, str, int] | None:
    response = requests.get('https://discovery.meethue.com/')
    bridges = response.json()[0]

    if bridges:
        id = bridges['id']
        ip = bridges['internalipaddress']
        port = bridges['port']

        return id, ip, port
    else:
        raise "No bridges found."


def get_light_with_name(lights: list[Light], name: str) -> Light | None:
    for light in lights:
        if light.name == name:
            return light

    return None


def xyY_to_sRGB(x: float, y: float, Y: int) -> ndarray[Any, dtype[Any]] | tuple[
    int, int, int]:

    Y = Y / 254
    X = x / y * Y
    Z = (1 - x - y) / y * Y

    return (np.array(colour.XYZ_to_sRGB([X, Y, Z])) * 255).astype(int)


def kelvin_cycle(light: Light) -> list[list[int | float]]:
    min_kelvin = 2000
    max_kelvin = 6000

    rgb_at_kelvin = []

    kelvin_range = np.linspace(min_kelvin, max_kelvin, num=21, dtype=int)

    light.on = True
    time.sleep(0.5)

    for kelvin in kelvin_range:
        light.colortemp_k = kelvin

        time.sleep(0.5)
        print(f'kelvin is {light.colortemp_k}')
        print('----')
        sRGB = xyY_to_sRGB(*light.xy, light.brightness)
        rgb_at_kelvin.append([light.colortemp_k, *sRGB])

    light.on = False

    return rgb_at_kelvin


kelvin_table_self = {
    2000: (255,210,39),
    2198: (255,218,72),
    2398: (255,224,94),
    2597: (255,229,113),
    2801: (255,233,128),
    3003: (255,236,142),
    3205: (255,239,154),
    3401: (255,241,164),
    3597: (255,243,173),
    3802: (255,245,182),
    4000: (255,246,190),
    4202: (255,248,198),
    4405: (255,249,205),
    4608: (255,250,211),
    4808: (255,250,217),
    5000: (255,251,222),
    5208: (255,252,228),
    5405: (255,252,233),
    5587: (255,253,237),
    5814: (255,253,242),
    5988: (255,254,245)
}
kelvin_table_colour = {
    2000: (294, 161, 27),
    2198: (283, 167, 53),
    2398: (274, 172, 70),
    2597: (266, 175, 85),
    2801: (258, 179, 97),
    3003: (252, 181, 108),
    3205: (246, 183, 117),
    3401: (241, 185, 125),
    3597: (236, 187, 132),
    3802: (231, 188, 139),
    4000: (227, 189, 145),
    4202: (224, 190, 151),
    4405: (220, 191, 157),
    4608: (217, 192, 162),
    4808: (214, 192, 166),
    5000: (211, 193, 170),
    5208: (209, 193, 175),
    5405: (206, 194, 178),
    5587: (204, 194, 182),
    5814: (202, 194, 185),
    5988: (200, 195, 188)
}

if __name__ == '__main__':
    id, ip, port = get_bridge_ip()

    bridge = Bridge(ip)
    bridge.connect()

    lamp_name = 'Lampe mod tv'
    light = get_light_with_name(bridge.lights, lamp_name)

    rgb_at_kelvin = kelvin_cycle(light)

    kelvin_table = pd.DataFrame(rgb_at_kelvin, columns=['kelvin', 'R', 'G', 'B'])
    print(kelvin_table)

    kelvin_table.to_csv('kelvin_table.csv', index=False)

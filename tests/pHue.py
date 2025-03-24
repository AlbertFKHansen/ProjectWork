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


def xyY_to_sRGB(x: float, y: float, Y: int) -> ndarray:
    if y == 0:
        raise 'y can not be zero'

    Y = Y / 255
    X = x / y * Y
    Z = (1 - x - y) / y * Y

    sRGB = colour.XYZ_to_RGB(np.array([X, Y, Z]), 'sRGB') * 255

    sRGB = np.clip(sRGB.astype(int), 0, 255)

    return sRGB


def kelvin_to_sRGB(K: float) -> ndarray:
    df = pd.read_csv('kelvin_table.csv')

    def get_rgb(data):
        R = data['R'].iloc[0]
        G = data['G'].iloc[0]
        B = data['B'].iloc[0]

        return np.array([R, G, B])

    try:
        row = df.loc[df['kelvin'] == K]
        return get_rgb(row)

    except IndexError:
        low = math.floor(K / 100) * 100
        low_row = df.loc[df['kelvin'] == low]
        lowRGB = get_rgb(low_row)

        high = math.ceil(K / 100) * 100
        high_row = df.loc[df['kelvin'] == high]
        highRGB = get_rgb(high_row)

        print(lowRGB, highRGB)

        r = (high - K) / 100
        interpolated = (1 - r) * lowRGB + r * highRGB

        return np.round(interpolated).astype(int)


def kelvin_cycle(light: Light) -> list[list[int | float]]:
    min_kelvin = 2000
    max_kelvin = 6500

    rgb_at_kelvin = []

    kelvin_range = np.linspace(min_kelvin, max_kelvin, num=41, dtype=int)
    #kelvin_range = range(2000, 6500+1)
    light.on = True
    time.sleep(0.5)

    for kelvin in kelvin_range:
        light.colortemp_k = kelvin

        time.sleep(0.5)
        sRGB = xyY_to_sRGB(*light.xy, light.brightness)
        color = [f'\033[38;2;{sRGB[0]};{sRGB[1]};{sRGB[2]}m', '\033[0m']
        print(f'kelvin is {color[0]}{kelvin}{color[1]}')
        print('----')

        rgb_at_kelvin.append([kelvin, *sRGB])

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
table = {
    2000: (255, 164, 5),
    2200: (255, 178, 16),
    2400: (255, 190, 29),
    2600: (255, 199, 42),
    2800: (255, 207, 55),
    3000: (255, 214, 69),
    3200: (255, 220, 82),
    3400: (255, 225, 94),
    3600: (255, 229, 107),
    3800: (255, 232, 120),
    4000: (255, 235, 131),
    4200: (255, 238, 144),
    4400: (255, 240, 155),
    4600: (255, 242, 166),
    4800: (255, 244, 177),
    5000: (255, 246, 187),
    5200: (255, 247, 198),
    5400: (255, 248, 207),
    5600: (255, 250, 215),
    5800: (255, 251, 226),
    6000: (255, 251, 233)
}


if __name__ == '__main__':
    print(kelvin_to_sRGB(4630))
    exit()
    id, ip, port = get_bridge_ip()
    bridge = Bridge(ip)
    bridge.connect()

    lamp_name = 'Lampe mod tv'
    light = get_light_with_name(bridge.lights, lamp_name)

    rgb_at_kelvin = kelvin_cycle(light)

    kelvin_table = pd.DataFrame(rgb_at_kelvin, columns=['kelvin', 'R', 'G', 'B'])
    print(kelvin_table)

    kelvin_table.to_csv('kelvin_table.csv', index=False)

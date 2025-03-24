import math
import time

import requests
import numpy as np
import pandas as pd
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
        print("No bridges found.")
        return None


def get_light_with_name(lights: list[Light], name: str) -> Light | None:
    for light in lights:
        if light.name == name:
            return light

    return None


def xyY_to_sRGB(x: float, y: float, Y: int) -> np.ndarray[tuple[int, ...], np.dtype]:
    if Y == 0:
        return 0, 0, 0

    Y = Y / 254

    X = x / y * Y
    Z = (1 - x - y) / y * Y

    # sRGB inverse matrix
    M = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    XYZ = np.array([X, Y, Z])

    linear_sRGB = M @ XYZ

    def linear_to_sRGB(c_linear: np.ndarray) -> np.ndarray:
        return np.where(
            c_linear <= 0.0031308,
            12.92 * c_linear,
            1.055 * (c_linear ** (1 / 2.4)) - 0.055
        )

    sRGB = linear_to_sRGB(linear_sRGB)

    sRGB_bounded = np.clip(sRGB, 0, 1) * 255

    return sRGB_bounded.astype(int)


def xyBriToRgb(x: float, y: float, bri: int) -> tuple[int, int, int]:
    # Calculate z from chromaticity coordinates
    z = 1.0 - x - y
    # Normalize brightness from 0-255 to a 0-1 range
    Y = bri / 255.0
    # Calculate X and Z based on chromaticity
    X = (Y / y) * x
    Z = (Y / y) * z

    # Convert to linear RGB using the given coefficients
    r_lin = X * 1.612 - Y * 0.203 - Z * 0.302
    g_lin = -X * 0.509 + Y * 1.412 + Z * 0.066
    b_lin = X * 0.026 - Y * 0.072 + Z * 0.962

    # Apply gamma correction to each channel
    def gamma_correct(c: float) -> float:
        if c <= 0.0031308:
            return 12.92 * c
        else:
            return (1.0 + 0.055) * math.pow(c, 1/2.4) - 0.055

    r = gamma_correct(r_lin)
    g = gamma_correct(g_lin)
    b = gamma_correct(b_lin)

    # Normalize so that the maximum channel equals 1
    max_value = max(r, g, b)
    if max_value != 0:
        r /= max_value
        g /= max_value
        b /= max_value

    # Scale to 0-255 and handle negative values as per your JS code
    r = r * 255
    g = g * 255
    b = b * 255

    if r < 0:
        r = 0
    if g < 0:
        g = 0
    if b < 0:
        b = 0

    # Round and convert to integers
    return int(round(r)), int(round(g)), int(round(b))


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
        print(xyBriToRgb(*light.xy, light.brightness))
        print(xyY_to_sRGB(*light.xy, light.brightness))
        print('----')
        sRGB = xyY_to_sRGB(*light.xy, light.brightness)
        rgb_at_kelvin.append([light.colortemp_k, *sRGB])

    light.on = False

    return rgb_at_kelvin


kelvin_table = {
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

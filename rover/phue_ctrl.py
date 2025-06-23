import math
import time
from typing import Any

import colour
import requests
import numpy as np
import pandas as pd
from numpy import ndarray
from phue import Bridge, Light

class LightController:
    def __init__(self, ip=None):
        if ip:
            self.ip = ip
        else:
            self.id, self.ip, self.port = self.get_bridge_ip()        
        self.bridge = Bridge(self.ip)
        self.lights = self.bridge.lights
    
    def get_bridge_ip(self) -> tuple[str, str, int] | None:
        '''
        used for the initialization of the hue hub.
        This function should not be used unless you are doing custom
        networking and need the network details of the hub.

        Even then, you can just read the controller objects values for id, ip and port
        '''
        response = requests.get('https://discovery.meethue.com/')
        try:
            bridge = response.json()[0]
            id = bridge['id']
            ip = bridge['internalipaddress']
            port = bridge['port']
            return id, ip, port
            
        except:
            return "id", "192.168.1.100", 553

    def on(self):
        for light in self.lights:
            light.on = True
    def off(self):
        for light in self.lights:
            light.on = False
    
    
    def set_temp(self, k: int):
        '''
        Used for changing the temperature of the lights on the network.
        The hue lamps we are using only have the range of 2000k-6500k

        input:
            k - Temperature (kelvin)
        '''
        min_k = 2000
        max_k = 6500
        
        if min_k > k or max_k < k:
            raise "temp not within correct range"
        
        for light in self.lights:
            light.colortemp_k = k
        time.sleep(0.5)
        
    def set_brightness(self, b):
        '''
        Used for changing the temperature of the lights on the network.
        The hue lamps we are using only have the range of 2000k-6500k
        
        input:
            b - brightness
        '''
        min_b = 0
        max_b = 254
        
        if min_b > b or max_b < b:
            raise "brightness not within correct range"
        
        for light in self.lights:
            light.brightness = b
        time.sleep(0.5)

    

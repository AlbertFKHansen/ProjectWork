import serial  
import json
import queue
import threading
import yaml
import os
import time
import glob
import numpy as np
from base_ctrl import BaseController
import argparse
from phue import PhueRequestTimeout


# our own scripts
from phue_ctrl import LightController
from dslr_ctrl import capture_image


def capture(args):
    print(f"\033[94mCollecting dataset for {args.name}")
    print("-----")

    if args.temp is None:
        print("default temp:    4000k")
    elif len(args.temp) == 0:
        print("default temp:    4000k")
        args.temp.append(4000)
        print("start temp:      2000k")
        args.temp.append(2000)
        print("end temp:        6500k")
        args.temp.append(6500)
    elif len(args.temp) == 1:
        print(f"default temp:    {args.temp[0]}k")
        print("start temp:      2000k")
        args.temp.append(2000)
        print("end temp:        6500k")
        args.temp.append(6500)
    elif len(args.temp) == 3:
        print(f"default temp:    {args.temp[0]}k")
        print(f"start temp:      {args.temp[1]}k") 
        print(f"end temp:        {args.temp[2]}k") 

    if args.rotate is None:
        print("rotation:        off")
    elif len(args.rotate) == 0:
        print("steps_size:      5")
        args.rotate.append(5)
        print("start position:  0")
        args.rotate.append(0)
        print("end position:    360")
        args.rotate.append(360)
    elif len(args.rotate) == 1:
        print(f"steps_size:      {args.rotate[0]}")
        print("start position:  0")
        args.rotate.append(0)
        print("end position:    360")
        args.rotate.append(360)
    elif len(args.rotate) == 3:
        print(f"steps_size:      {args.rotate[0]}")
        print(f"start position:  {args.rotate[1]}") 
        print(f"end position:    {args.rotate[2]}")
    print("-----")
    
    
    if args.temp is None:
        light_ctrl.on()
        light_ctrl.set_temp(4000)
        default_temp = 4000
    elif args.rotate is None:
        start_pos = 0
    else:
        start_pos = args.rotate[1]
        default_temp = args.temp[0]
        light_ctrl.on()
        base.gimbal_ctrl(start_pos-180,0,0,0)
        time.sleep(1)


    
    if args.temp:
        for k in range(args.temp[1],args.temp[2]+100,100):
            light_ctrl.set_temp(k)
            print(f"\r\033[KAngle: {start_pos}     Temp: {k}", end="")
            time.sleep(0.5)
            fileName = f'/Data/temp/{args.name}/{k}.png'
            capture_image(thisPath + fileName)
            time.sleep(0.5)
        
        light_ctrl.set_temp(default_temp)
        time.sleep(1.0)

    if args.rotate:
        for i in range(args.rotate[1], args.rotate[2], args.rotate[0]):
            base.gimbal_ctrl(i-180,0,0,0)
            print(f"\r\033[KAngle: {i}     Temp: {default_temp}", end="")
            time.sleep(0.5)
            fileName = f'/Data/rot/{args.name}/{i}.png'
            capture_image(thisPath + fileName)
            time.sleep(0.5)
            
    if not args.rotate and not args.temp:
        print("No actions were specified, so no images will be captured")
        print("Use \033[92m-r \033[94mor \033[92m-t \033[94m to specify physical interventions")
        
    print("\n")


# customargparser to handle no exit on "command --help"
class NoExitArgumentParser(argparse.ArgumentParser):
    def exit(self, status=0, message=None):
        if message:
            print(message)
        # Don't exit the program
    
    def error(self, message):
        self.print_usage()
        print(f"Error: {message}")
        





def main():
    parser = argparse.ArgumentParser(
        description="The main control environment for datacollection with the UGV Rover"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # --- Help ---
    help_parser = subparsers.add_parser("help", help="Displays this message")


    # --- Quit ---
    quit_parser = subparsers.add_parser("exit", help="Exit the UGV rover interface")

    # --- Dataset collection ---
    dataset_parser = subparsers.add_parser("capture", help="Collect images of an object from sequences of physical interventions")
    dataset_parser.add_argument("-n", "--name", type=str, required=True, help="The name of object to name/sort data after")
    dataset_parser.add_argument("-r","--rotate", nargs="*", type=int, help="Default is -r 5 0 360, but you can parse custom values for either the step-size(5) or all three arguments")
    dataset_parser.add_argument("-t","--temp", nargs="*", type=int, help="Default is -t 4000 2000 6500, but you can parse custom values for either the default(4000) or all three arguments")


    # --- rotation controller ---
    rotate_parser = subparsers.add_parser("rotate", help="manual rotation control")
    rotate_parser.add_argument("-r","-a","--angle", type=int, help="angle ranges from 0 to 360")


    # --- lights controller ---
    light_parser = subparsers.add_parser("light", help="command used for manual light control")
    light_parser.add_argument("--on", action="store_true", help="Turn lights on")
    light_parser.add_argument("--off", action="store_true", help="Turn lights off")
    light_parser.add_argument("-t","--temp", type=int, help="set temperature of lights (2000k - 6500k)")
    light_parser.add_argument("-b","--brightness", type=int, help="set brightness of lights (0-254)")


    # --- single image ---
    image_parser = subparsers.add_parser("image", help="take a single picture")
    image_parser.add_argument("-n", "--name", type=str, required=True, help="The name of the image")
    

    
    
    # user input loop
    while True:
        try:
            user_input = input("\033[92m\033[1m> \033[0m").strip().lower()
            
            args = parser.parse_args(user_input.split(" "))

            if not user_input.strip():
                continue
            elif args.command == "exit":
                break
            elif args.command == "help":
                parser.print_help()
            elif args.command == "rotate":
                print(f"\033[94mRotating to {args.angle}... ", end="")
                base.gimbal_ctrl(args.angle-180,0,0,0)
                time.sleep(0.5)
                print("Done\033[0m")
            elif args.command == "capture":
                capture(args)
            elif args.command == "light":
                if args.brightness:
                    print(f"\033[94mSetting brightness to {args.brightness}... ", end="")
                    light_ctrl.set_brightness(args.brightness)
                    light_ctrl.on()
                    print("Done\033[0m")
                if args.temp:
                    print(f"\033[94mSetting temperature to {args.temp}... ", end="")
                    light_ctrl.set_temp(args.temp)
                    light_ctrl.on()
                    print("Done\033[0m")
                elif args.on:
                    print("\033[94mTurning light on... ", end="")
                    light_ctrl.on()
                    print("Done\033[0m")
                elif args.off:
                    print("\033[94mTurning light off... ", end="")
                    light_ctrl.off()
                    print("Done\033[0m")

            elif args.command == "image":
                fileName=f"/Data/individual/{args.name}.png"
                capture_image(thisPath + fileName)

        except Exception as err:
            print("\033[91mSomething went wrong\n")
            print(type(err))
            print(err)
    







if __name__ == '__main__':
    os.system('clear')

    # Checking if base_ctrl works
    try:
        print("\033[94mSetting up base controller... ", end="")
        base = BaseController('/dev/ttyAMA0', 115200) # only works for rpi
    except:
        print("\033[91mBase controller not functional. Can't do motor control.\033[0m")

    

    # Connecting to Philips Hue Bridge
    try:
        print("\033[94mConnecting to pHue Bridge... ", end="")
        light_ctrl = LightController()
        light_ctrl.on()
        light_ctrl.set_temp(4000)
        light_ctrl.set_brightness(254)
        time.sleep(1)
        print("Done\033[0m\n")
    except PhueRequestTimeout:
        print("\033[91mThe request time out while trying to connect to the bridge, and the default ip \033[0m192.168.1.100\033[91m did not work.")
        ip = input("\033[92mProvide bridge ip: \033[0m")
        try:
            light_ctrl = LightController(ip)
            light_ctrl.on()
            light_ctrl.set_temp(4000)
            light_ctrl.set_brightness(254)
            light_ctrl.off()
        except PhueRequestTimeout:
            print("\033[91mThe request time out again. Continuing without light control\033[0m")

    # Getting path
    curpath = os.path.realpath(__file__)
    thisPath = os.path.dirname(curpath)
    print(f"\033[94mData will be located in \033[95m{thisPath}/Data")
    print("\033[94mUse \033[95mscp -r ws@[ROVER IP]:/home/ws/ugv_rpi/Data/* /path/to/your/data/directory\033[94m to copy all data via ssh into your local Data dicrectory\n")

    Title="""\033[92m\033[1m  _    _  _______      __  _____                     
 | |  | |/ ____\ \    / / |  __ \                    
 | |  | | |  __ \ \  / /  | |__) |_____   _____ _ __ 
 | |  | | | |_ | \ \/ /   |  _  // _ \ \ / / _ \ '__|
 | |__| | |__| |  \  /    | | \ \ (_) \ V /  __/ |   
  \____/ \_____|   \/     |_|  \_\___/ \_/ \___|_|\033[0m
    """
    print(Title)
    
    print("Type \"\033[3m[command] -h\033[0m\" or \"\033[3m[command] --help\033[0m\" for help with commands or \"\033[3mhelp\033[0m\" for a list of commands...\n")
    
    main()

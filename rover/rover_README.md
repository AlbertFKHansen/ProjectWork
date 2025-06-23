# Guide to using the UGV Rover
This is both a general guide on how to use the system and a mention of the things to be aware of that can be quirky and problematic at times.
I'm not sure how much anyone using this already knows about networking and messing around in a terminal, so i'm just writing what might be relevant.

If there are any questions relating to unexplained details about the system, email me at: akoha@dtu.dk


## Connecting and using the CLI
Let's say the local rover ip is 192.168.1.113

then connect via ssh with the user "ws" in terminal. The password is: raspberry

```bash
ssh ws@192.168.1.113
```

jump to the environment directory

```bash
ws@192.169.1.113:~$ cd rpi_env
```

press the big center on the Philips Hue Hub. this puts it in some sort of paring mode (not bluetooth, wifi based)

```bash
ws@192.169.1.113:~/rpi_env$ python3 main.py
```

Wait for the rover to connect to the hub.

You can type --help or -h for a list of commands and "[command] --help" for details on the command.



## Design choices and quirks
### Philips Hue Hub
The rover is designed to be used with a Philips Hue Hub, which is connected to the local network via LAN. But the CLI will work either way. You just can't do the light commands without the hub.

The hub is not bluetooth absed, but wifi based. For some reason there is an attempt limit for the automatic discovery of the hub in a 10-15 minute time frame. if that happens, you can either wait and start the CLI again after pressing the big center button on the hub, or you can try to connect to the hub manually by using its ip when the CLI prompts you for it.

### Camera
We are using a DSLR EOS 2000D from Canon, which is connected via USB to the rover. The camera is controlled by the gphoto2 tool, which is installed on the rover. There are some quirks. It is not strictly necesary to use manual mode on the camera, but i highly recomend it, both for time effeciency and for the consistency of photos. The "deamon" (im not sure if its an actual deamon, but it practiucally works like one) have a few issues. The main issue is retaining camera control. We only experienced this when we used the camera - looking at images or maybe adjusting optical zoom. If you do that, the camera can simply be turend off and back on.

### Why a router is needed
- To connect to the UGV Rover, you need to know it's ip on the local network
- You need to be on a network that allows intercommunication between devices
- The Philips Hue HUB is required to be connected via LAN, but is blocked by DTU's Network
- it is possible to connect the rover to a wifi, but you have to provide authentication details in a yaml file, which is not ideal for DTU networks, since auth for DTU is based on user accounts.

All of the above can be solved by using a router, which is connected to the Philips Hue Hub via LAN, and the rover connects to the router via wifi (which i already did for the router we used) or LAN.

To connect via wifi:
name (ssid):    Syslink
password:       1234567890

You can control and see everything related to the router by typing 192.168.1.1 as the URL in your browser.
username:   admin
password:   admin

A quick hack to quickly find the ip of devices on your network, if you are on Linux or MacOS, is to use nmap.
```bash
sudo nmap -sn 192.168.1.0/24
```
This will "Scan Ping" all ip addresses with the first 24 bits fixed (192.168.1.XXX) and report which devices responds.



### Designed for Failure
The CLI tells you what the rotation and temperature is at all times, if a problem happens - like camera not responding or the battery dying - you can simply stop the script, and start a new capture session from where you left off by defining it:

```
capture -n [name] -r -t
```

The temperature intervention fineshes, and rotation begins.. in the middle of rotation the camera stops taking images at about 250 degrees.

you stop the session with ctrl+c, change the battery of the camera, and then you can start a new session with the same name.

```
capture -n [name] -r 5 230 360
```

using 230 degrees as the start, you make sure that no angles are skipped. The new image files will override the old ones, so you can just continue where you left off and complete the capture session.





## File overview
For legal reasons (I don't want to risk misunderstanding the license) i'm only sharing the code that i wrote for the rover. The CLI also depends on motor control. This comes from the base_ctrl.py found on the rover. The base controller depends on at least one config file as far as i could tell. But i have not investigated the control system in detail.

### phue_ctrl.py
This file has a controller object than on init tries to automatically find the ip of the hub. The controller object has functions for light control.

### dslr_ctrl.py
has a function that takes file name as argument. This is the function that uses gphoto2 and stores the captured photos as PNG files on the rover. (the CLI makes the path and file name be based on ./Data/{intervention}/{object}/{value}.png)

### main.py
The CLI. there are a couple of different commands to be used here. most of them are pretty self explanatory based on the --help flag. I dont think i ever fixed a small issue with the program ending when you use "[command] --help". But you can just run the script again.

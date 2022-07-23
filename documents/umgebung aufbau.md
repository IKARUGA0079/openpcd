

#  Umgebung zum Erstellen

###Zuerst müssen die virtuelle Maschine herunterladen
[virtuelle Maschine](https://www.virtualbox.org/)
###Laden das Ubuntu-Image, iso herunter
Beachten Sie hier, dass die neueste Ubuntu-Version 22.04 (15.05.2022) ist und die neueste Ubuntu-Version keine kompatible ROS-Version hat. Hier müssen wir also die Ubuntu 20.04-Version herunterladen
[Ubuntu-Version 20.04](https://releases.ubuntu.com/20.04.4/)
Korrespondenz zwischen Ubuntu- und ROS-Versionen:
- Ubuntu 16.04（Xenial） + ROS 1 Kinetic    +  ROS 2  Ardent
- Ubuntu 18.04（Bionic） + ROS 1 Melodic   +  ROS 2  Dashing LTS
- Ubuntu 20.04（Focal）  + ROS 1 Noetic     +  ROS 2  Foxy LTS

###ROS 安装
Öffnen Sie zunächst den Dialog Software und Updates, der über die Suchschaltfläche in der oberen linken Ecke von Ubuntu durchsucht werden kann.
Konfigurieren Sie nach dem Öffnen wie unten gezeigt (stellen Sie sicher, dass „restricted“, „universe“ und „multiverse“ angekreuzt sind)
![enter image description here](https://img-blog.csdnimg.cn/2020111009114464.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xlc2xpZV9fX0NoZXVuZw==,size_16,color_666666,t_70#pic_center)


#### Quelle hinzufügen
Öffnen Sie eine Konsole (Strg + Alt + T) und geben Sie den folgenden Befehl ein:
`sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'`

####Legen den geheimen Schlüssel fest
`sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654`

Nachdem wir eine neue Quelle hinzugefügt haben, müssen wir die Quellenliste einmal aktualisieren, geben Sie `sudo apt-get update` in das Terminal ein, um zu aktualisieren.

#### ROS installieren
Dann können wir ROS installieren.Hier haben Sie die Wahl, die vollständige Desktop-Version zu installieren.
`sudo apt install ros-noetic-desktop-full`

###Konfigurierendie ROS-Umgebung für das System
`sudo rosdep init`
`rosdep update`
Wenn bei der Installation des ROS-Initialisierungsschritts rosdep ein Fehler auftritt, dass der Befehl nicht gefunden werden kann, lautet die Lösung ：
`sudo apt-get install python-rosdep` 
or   `sudo apt-get install python3-rosdep` 

Umgebungsvariablen initialisieren:
`echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc`
`source ~/.bashrc`

### ROS2 installieren
####1. Stellen die Codierung ein
`sudo locale-gen en_US en_US.UTF-8`
`sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8`
`export LANG=en_US.UTF-8`
 
####2. Stellen die Softwarequelle ein
`sudo apt update && sudo apt install curl gnupg2 lsb-release`
`curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -`
 
`sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources. list.d/ros2-latest.list'`
 
####3. Installieren ROS2
`sudo apt aktualisieren`
`sudo apt installiere ros-foxy-desktop`

####4. Legen Umgebungsvariablen fest
`Quelle /opt/ros/foxy/setup.bash`
Wenn ROS1 installiert ist, wird eine Warnung ausgegeben，Dies kann ignoriert werden

###5. Installieren das Tool zur automatischen Vervollständigung
`sudo apt install python3-argcomplete`

### End

export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=true

echo $1

xvfb-run -a webots --mode=fast --stdout --stderr --batch --minimize --no-rendering --port=$1 empty_room.wbt &
export WEBOTS_HOME=/usr/local/webots
$WEBOTS_HOME/webots-controller --robot-name=HamBot --port=$1 ./TestSAC.py

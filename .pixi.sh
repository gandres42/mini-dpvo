export SPACE_ID="pablovela5620/mini-dpvo-zero"
export WHL_PATH="mini_dpvo-0.1.1-cp311-cp311-linux_x86_64.whl"

if [ -z "${DISPLAY:-}" ] && [ -S /tmp/.X11-unix/X1 ]; then
    export DISPLAY=":1"
fi

if [ -z "${XAUTHORITY:-}" ] && [ -f /run/user/1000/gdm/Xauthority ]; then
    export XAUTHORITY="/run/user/1000/gdm/Xauthority"
fi

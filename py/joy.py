import time
import uinput

def main():
    events = (
        uinput.BTN_A,
        uinput.BTN_B,
        uinput.BTN_X,
        uinput.BTN_Y,
        uinput.BTN_TL,
        uinput.BTN_TR,
        uinput.BTN_THUMBL,
        uinput.BTN_THUMBR,
        uinput.ABS_X + (0, 255, 0, 0),
        uinput.ABS_Y + (0, 255, 0, 0),
    )

    device = uinput.Device(
        events,
        vendor=0x045e,
        product=0x028e,
        version=0x110,
        name="Microsoft X-Box 360 pad",
    )

    with device as dev:
        # Center joystick
        # syn=False to emit an "atomic" (128, 128) event.
        dev.emit(uinput.ABS_X, 128, syn=False)
        dev.emit(uinput.ABS_Y, 128, syn=False)

        while True:
            for i in range(256):
                dev.emit(uinput.ABS_X, i)
                dev.emit(uinput.ABS_Y, i)

                time.sleep(.01)
            
            for i in range(255, -1, -1):
                dev.emit(uinput.ABS_X, i)
                dev.emit(uinput.ABS_Y, i)

                time.sleep(.01)

if __name__ == "__main__":
    main()
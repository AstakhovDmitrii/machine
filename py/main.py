import cv2 as cv
from cv2 import aruco
import numpy as np
import serial
import time
import argparse
from enum import Enum


CRSF_SYNC = 0xC8
PACKET_LENGTH = 24
COMPORT = "/dev/ttyUSB0"
BAUDRATE = 921600
CHANNEL_LENGTH = 11
CHANNELS_NUM = 16


class PacketsTypes(int, Enum):
    GPS = 0x02
    VARIO = 0x07
    BATTERY_SENSOR = 0x08
    BARO_ALT = 0x09
    HEARTBEAT = 0x0B
    VIDEO_TRANSMITTER = 0x0F
    LINK_STATISTICS = 0x14
    RC_CHANNELS_PACKED = 0x16
    ATTITUDE = 0x1E
    FLIGHT_MODE = 0x21
    DEVICE_INFO = 0x29
    CONFIG_READ = 0x2C
    CONFIG_WRITE = 0x2D
    RADIO_ID = 0x3A


def crc8_dvb_s2(crc, a) -> int:
  crc = crc ^ a
  for ii in range(8):
    if crc & 0x80:
      crc = (crc << 1) ^ 0xD5
    else:
      crc = crc << 1
  return crc & 0xFF


def crc8_data(data) -> int:
    crc = 0
    for a in data:
        crc = crc8_dvb_s2(crc, a)
    return crc


def crsf_validate_frame(frame) -> bool:
    return crc8_data(frame[2:-1]) == frame[-1]


def signed_byte(b):
    return b - 256 if b >= 128 else b


def pack_CRSF_to_bytes(channels) -> bytes:
    # Channels is in CRSF format! (0-1984)
    # Values are packed little-endianish
    # such that bits BA987654321 -> 87654321, 00000BA9
    # 11 bits per channel x 16 channels = 22 bytes
    if len(channels) != CHANNELS_NUM:
        raise ValueError(f"CRSF must have {CHANNELS_NUM} channels")
    
    result = bytearray()
    dest_shift = 0
    new_value = 0
    for ch in channels:
        # Put the low bits in any remaining dest capacity.
        new_value |= (ch << dest_shift) & 0xff
        result.append(new_value)

        # Shift the high bits down and place them into the next dest byte.
        src_bits_left = CHANNEL_LENGTH - 8 + dest_shift
        new_value = ch >> (CHANNEL_LENGTH - src_bits_left)
        # When there's at least a full byte remaining, consume that as well.
        if src_bits_left >= 8:
            result.append(new_value & 0xff)
            new_value >>= 8
            src_bits_left -= 8

        # Next dest should be shifted up by the bits consumed.
        dest_shift = src_bits_left

    return result


def channels_CRSF_to_packet(channels) -> bytes:
    result = bytearray([
        CRSF_SYNC, PACKET_LENGTH, PacketsTypes.RC_CHANNELS_PACKED
    ])
    result += pack_CRSF_to_bytes(channels)
    result.append(crc8_data(result[2:]))
    
    return result


def handle_CRSF_packet(ptype, data):
    match ptype:
        case PacketsTypes.RADIO_ID:
            if data[5] == 0x10:
                # print(f"OTX sync")
                pass

        case PacketsTypes.LINK_STATISTICS:
            rssi1 = signed_byte(data[3])
            rssi2 = signed_byte(data[4])
            lq = data[5]
            snr = signed_byte(data[6])
            antenna = data[7]
            mode = data[8]
            power = data[9]

            # Telemetry strength.
            downlink_rssi = signed_byte(data[10])
            downlink_lq = data[11]
            downlink_snr = signed_byte(data[12])
            print(
                f"RSSI={rssi1}/{rssi2}dBm LQ={lq:03} mode={mode} "
                f"ant={antenna} snr={snr} power={power} drssi={downlink_rssi} "
                f"dlq={downlink_lq} dsnr={downlink_snr}"
            )
        
        case PacketsTypes.ATTITUDE:
            pitch = int.from_bytes(data[3:5], byteorder="big", signed=True) / 10000.0
            roll = int.from_bytes(data[5:7], byteorder="big", signed=True) / 10000.0
            yaw = int.from_bytes(data[7:9], byteorder="big", signed=True) / 10000.0
            print(f"Attitude: Pitch={pitch:0.2f} Roll={roll:0.2f} Yaw={yaw:0.2f} (rad)")

        case PacketsTypes.FLIGHT_MODE:
            packet = "".join(map(chr, data[3:-2]))
            print(f"Flight Mode: {packet}")

        case PacketsTypes.BATTERY_SENSOR:
            vbat = int.from_bytes(data[3:5], byteorder="big", signed=True) / 10.0
            curr = int.from_bytes(data[5:7], byteorder="big", signed=True) / 10.0
            mah = data[7] << 16 | data[8] << 7 | data[9]
            pct = data[10]
            print(f"Battery: {vbat:0.2f}V {curr:0.1f}A {mah}mAh {pct}%")

        case PacketsTypes.BARO_ALT:
            print(f"BaroAlt: ")

        case PacketsTypes.DEVICE_INFO:
            packet = " ".join(map(hex, data))
            print(f"Device Info: {packet}")

        case PacketsTypes.VARIO:
            vspd = int.from_bytes(data[3:5], byteorder="big", signed=True) / 10.0
            print(f"VSpd: {vspd:0.1f}m/s")
        
        case PacketsTypes.RC_CHANNELS_PACKED:
            # print(f"Channels: (data)")
            pass
        
        case _:
            packet = " ".join(map(hex, data))
            print(f"Unknown 0x{ptype:02x}: {packet}")

# load in the calibration data
calib_data_path = "MultiMatrix.npz"
calib_data = np.load(calib_data_path)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]
print(calib_data.files)

MARKER_SIZE = 6  # centimeters (measure your printed marker size)
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
param_markers = aruco.DetectorParameters()
cap = cv.VideoCapture(0)


def map(x, in_min, in_max, out_min, out_max):
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def clamp(n, minn, maxn):
    if n < minn:
        return minn
    elif n > maxn:
        return maxn
    else:
        return n
        
        

with serial.Serial(COMPORT, BAUDRATE, timeout=2) as ser:
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		marker_corners, marker_IDs, reject = aruco.detectMarkers(gray_frame, marker_dict, parameters=param_markers)
		
		
		
		
		if marker_corners:
			rVec, tVec, _ = aruco.estimatePoseSingleMarkers(marker_corners, MARKER_SIZE, cam_mat, dist_coef)
			total_markers = range(0, marker_IDs.size)
			if (len(marker_IDs) > 0):
				i = 0
				corners = marker_corners[i]
				cv.polylines(
					frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
				)
				corners = corners.reshape(4, 2)
				corners = corners.astype(int)
				top_right = corners[0].ravel()
				top_left = corners[1].ravel()
				bottom_right = corners[2].ravel()
				bottom_left = corners[3].ravel()

				# Calculating the distance
				distance = np.sqrt(
					tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
				)
				rotate = 992
				speed = 992
				if (distance > 10): # исправить дистанцию
					if (tVec[i][0][0] != 0):
						rotate = clamp(map(tVec[i][0][0], -6, 6, 800, 1100), 800, 1100) # исправить значения для мапа
					speed = clamp(900 + map(distance, 10, 20, 0, 300), 992, 1200) # исправить значения для мапа
					
				
				ch0 = [speed, rotate, 992, 992, 992, 992, 992, 992, 992, 992, 992, 992, 992, 992, 992, 992]	
				
				ser.write(channels_CRSF_to_packet(ch0))
				

				point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
				cv.putText(
					frame,
					f"id: {marker_IDs[i]} Dist: {round(distance, 2)}",
					top_right,
					cv.FONT_HERSHEY_PLAIN,
					1.3,
					(0, 0, 255),
					2,
					cv.LINE_AA,
				)
				cv.putText(
					frame,
					f"x:{round(tVec[i][0][0],1)} r: {round(rotate,1)} s: {round(speed, 1)} ",
					bottom_right,
					cv.FONT_HERSHEY_PLAIN,
					1.0,
					(0, 0, 255),
					2,
					cv.LINE_AA,
				)
		
		
		
		
		
		
		
		cv.imshow("frame", frame)
		key = cv.waitKey(1)
		if key == ord("q"):
        		break							


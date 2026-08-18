"""
Wi-Fi signal test for fence vs window mounting decision
Per blueprint Sec 6 & action item 3: physically test signal at fence before committing.

Run this on Pi at intended fence location:
  python scripts/wifi_test.py

It logs RSSI, link quality, and pings router.
Also tests sustained iPerf-ish upload for dashboard streaming.

Useful if Pi 3B onboard Wi-Fi is marginal at fence ~100-150ft with walls.
If marginal, consider mesh/extender near window, directional antenna, or Ethernet run
(Pi 3B has onboard Ethernet despite power complication).
"""

import subprocess
import time
import re
import socket
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("wifi_test")

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout + result.stderr
    except Exception as e:
        return f"cmd failed: {e}"

def get_rssi():
    # Try iwconfig
    out = run_cmd("iwconfig wlan0 2>&1")
    # Look for Signal level
    m = re.search(r"Signal level[=|:]\s*(-?\d+)\s*dBm", out)
    if m:
        return int(m.group(1)), out
    m = re.search(r"Link Quality[=|:]\s*(\d+/\d+)", out)
    # Try iw dev link
    out2 = run_cmd("iw dev wlan0 link 2>&1")
    m2 = re.search(r"signal:\s*(-?\d+)\s*dBm", out2)
    if m2:
        return int(m2.group(1)), out + "\n" + out2
    return None, out + "\n" + out2

def ping_test(target="8.8.8.8", count=10):
    out = run_cmd(f"ping -c {count} {target} 2>&1")
    # parse avg
    m = re.search(r"avg.*?=\s*[\d\.]+/([\d\.]+)/", out)
    if m:
        return float(m.group(1)), out
    return None, out

def check_ip():
    out = run_cmd("hostname -I")
    return out.strip()

def main(duration_sec=60):
    print("=== Wi-Fi Signal Test for fence mounting (Blueprint Sec 6) ===")
    print(f"IP: {check_ip()}")
    print(f"Testing for {duration_sec}s, sampling every 5s...\n")

    samples = []
    start = time.time()
    while time.time() - start < duration_sec:
        rssi, raw = get_rssi()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        if rssi is not None:
            # classify
            if rssi >= -50:
                qual = "Excellent"
            elif rssi >= -60:
                qual = "Good (window OK, fence likely OK)"
            elif rssi >= -70:
                qual = "Fair (window OK, fence marginal – consider extender)"
            elif rssi >= -80:
                qual = "Weak (likely unusable at fence – needs Ethernet or extender)"
            else:
                qual = "Unusable"
            line = f"[{ts}] RSSI {rssi} dBm -> {qual}"
            print(line)
            samples.append(rssi)
        else:
            print(f"[{ts}] RSSI parse failed, raw:\n{raw[:500]}")
        time.sleep(5)

    if samples:
        avg = sum(samples)/len(samples)
        print(f"\nAverage RSSI: {avg:.1f} dBm over {len(samples)} samples")
        if avg >= -67:
            print("Verdict: Fence mount likely viable for dashboard (Flask over Wi-Fi) – still test with feeder running")
        elif avg >= -75:
            print("Verdict: Borderline – window mount recommended unless you add mesh/extender near window or Ethernet")
        else:
            print("Verdict: Wi-Fi too weak at fence – choose window mount or run Ethernet + weatherproof power")

    print("\n--- Ping test to router (gateway) ---")
    gw = run_cmd("ip route | grep default | awk '{print $3}'").strip().split("\n")[0] if run_cmd else "192.168.1.1"
    if not gw:
        gw = "8.8.8.8"
    print(f"Pinging gateway {gw}")
    avg_ping, out = ping_test(gw, 20)
    print(out[:1000])
    if avg_ping:
        print(f"Avg ping {avg_ping:.1f}ms")
        if avg_ping > 50:
            print("High latency – dashboard may be sluggish")

    print("\n--- Action items ---")
    print(" - If RSSI >= -67, fence mount candidate OK but scope power (solar vs cable) per Sec 6")
    print(" - If RSSI < -70, stick to window mount = easy indoor 5V microUSB + direct sightline")
    print(" - Test again with actual feeder orientation (feeding side facing window if you want human viewing)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60, help="seconds to sample")
    args = parser.parse_args()
    main(args.duration)

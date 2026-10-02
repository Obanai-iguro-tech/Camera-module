# V380 Smart Recorder

Desktop companion app for Windows that watches the V380 live-view panel, detects people on-device, and clicks V380's native Record button so clips are saved by V380 to the camera SD card/cloud.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m v380_smart_recorder
```

On first launch, configure each camera:

1. Open V380 and show the camera tab.
2. In this app, select the matching camera.
3. Click `Set Capture Region`, enter the V380 video panel coordinates as `x,y,width,height`, and save.
4. Click `Calibrate Record Button`, hover over V380's Record button, then press `F8`.
5. Start monitoring.

The app does not create local video files. It only screen-captures pixels for detection and sends Record/Stop clicks to V380.

## Notes

- YOLOv8n weights download on first run through `ultralytics`.
- V380 must remain visible. Minimized windows cannot be captured reliably.
- Coordinates are stored per camera in `config.json`.
- Manual Record/Stop buttons send the same calibrated click as automatic detection.

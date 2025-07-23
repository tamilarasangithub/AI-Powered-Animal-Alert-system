"hello everyone, this is a simple animal alert system that uses YOLOv8 to detect specific animals in real-time from a webcam feed and sends alerts via Telegram. It is designed to help monitor wildlife or pets by sending notifications when certain animals are detected. The code includes functionality for capturing images, processing them with a pre-trained YOLO model, and sending alerts with the detected animal information. Let's take a look at the code below:"
"this my animal alert system using YOLOv8 and Telegram"
"my name is [Tamilarasan], and I am excited to share this project with you. The system is built using Python, OpenCV, and the YOLOv8 model for object detection. It continuously monitors a webcam feed, detects specified animals, and sends alerts through Telegram when these animals are detected. This can be particularly useful for wildlife monitoring or pet surveillance. Below is the complete code for the animal alert system."
import asyncio
import time
from pathlib import Path
import cv2
from telegram import Bot
from ultralytics import YOLO
"paste your token and chat id below"
TOKEN   = "paste_here_token"    # bot token
CHAT_ID = paste here chatid                                          # chat ID
ANIMAL_CLASSES = {"elephant", "tiger", "lion", "leopard"}     # what to get alert on
COOLDOWN_SEC   = 30                                           # min seconds between pings
MODEL_WEIGHTS  = "yolov8n.pt"                                 # auto‑downloads
CAM_INDEX      = 0                                            # default webcam
SNAP_PATH      = Path("detected.jpg")                         # temp image path
bot   = Bot(TOKEN)
model = YOLO(MODEL_WEIGHTS)
cap   = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open webcam – check camera index or permissions.")

print("🚀 Animal‑Alert started – press ‘x’ to quit.")
last_sent = 0.0

async def send_alert(caption: str, photo_path: Path) -> None:
    """Send Telegram photo + caption asynchronously."""
    async with bot:
        with photo_path.open("rb") as img:
            await bot.send_photo(chat_id=CHAT_ID, photo=img, caption=caption)

while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️  Failed to read frame.")
        break

    # YOLO inference
    results = model(frame, verbose=False)[0]
    found = []

    for box in results.boxes:
        cls_name = model.names[int(box.cls)]
        conf     = float(box.conf)
        if cls_name not in ANIMAL_CLASSES or conf < 0.35:
            continue

        found.append(cls_name)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_name} {conf:.2f}",
                    (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if found and (time.time() - last_sent) > COOLDOWN_SEC:
        last_sent = time.time()
        caption   = f"⚠️ Detected: {', '.join(set(found)).title()}"
        cv2.imwrite(str(SNAP_PATH), frame)
        print("🟢 " + caption + " – sending to Telegram…", flush=True)
        try:
            asyncio.run(send_alert(caption, SNAP_PATH))
        except Exception as exc:
            print("❌ Telegram error:", exc, flush=True)

    cv2.imshow("Animal‑Alert", frame)
    if cv2.waitKey(1) & 0xFF == ord('X'):
        print("👋 Exiting…")
        break

cap.release()
cv2.destroyAllWindows()

"first run this code in powershell pip install opencv-python ultralytics python-telegram-bot"
" again run this code in powershell pip install opencv-python ultralytics python-telegram-bot"
" this code is run in powershell"

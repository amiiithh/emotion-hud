import cv2
import numpy as np
from deepface import DeepFace

cap = cv2.VideoCapture(0)

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False
        )

        emotion_scores = result[0]["emotion"]
        top_emotion = max(emotion_scores, key=emotion_scores.get)

    except Exception:
        emotion_scores = {e: 0 for e in EMOTIONS}
        top_emotion = "neutral"

    # HUD panel
    panel_x, panel_y = 20, 60
    line_h = 28
    max_bar_w = 160

    cv2.putText(frame, "Tracked emotions:",
                (panel_x, panel_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for i, emo in enumerate(EMOTIONS):
        y = panel_y + i * line_h
        score = emotion_scores.get(emo, 0)
        bar_w = int((score / 100) * max_bar_w)

        cv2.putText(frame, f"{emo:8s}",
                    (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.rectangle(frame,
                      (panel_x + 90, y - 12),
                      (panel_x + 90 + max_bar_w, y + 4),
                      (60, 60, 60), -1)

        cv2.rectangle(frame,
                      (panel_x + 90, y - 12),
                      (panel_x + 90 + bar_w, y + 4),
                      (255, 100, 180), -1)

    cv2.putText(frame,
                f"Top emotion: {top_emotion}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)

    cv2.putText(frame,
                "Press 'q' to quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1)

    cv2.imshow("AI Face Emotion HUD", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindo

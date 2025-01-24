import cv2
import math
import numpy as np

# Video dosyasını yükle
video_path = 'Lane Detection Test Video 01.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video dosyası açılamadı.")
    exit()

# Video boyutları ve FPS değerini al
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video Boyutları: {frame_width}x{frame_height}")
print(f"FPS: {fps}")

# Yeni boyutlar
new_width, new_height = 640, 480

# Çıktı videosu ayarları
output_path = 'final_output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

# ROI (Region of Interest) için üçgen alanı belirleme
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

# Şeritleri filtreleme fonksiyonu (slope bazlı)
def filter_lines(lines, width):
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            # Eğimi belirli bir aralığa filtrele
            if 0.5 < abs(slope) < 2:  # Şeritlerin eğimi genelde bu aralıkta olur
                if slope < 0 and x1 < width // 2 and x2 < width // 2:  # Sol şerit
                    left_lines.append((x1, y1, x2, y2))
                elif slope > 0 and x1 > width // 2 and x2 > width // 2:  # Sağ şerit
                    right_lines.append((x1, y1, x2, y2))
    return left_lines, right_lines

# Ortalama çizgiyi hesaplama
def average_line(lines, prev_line=None, alpha=0):
    if len(lines) == 0:
        return prev_line  # Eğer çizgi bulunamazsa önceki çizgiyi kullan
    x1, y1, x2, y2 = np.mean(lines, axis=0, dtype=int)
    if prev_line is not None:
        # Moving average ile yumuşatma
        x1 = int(alpha * prev_line[0] + (1 - alpha) * x1)
        y1 = int(alpha * prev_line[1] + (1 - alpha) * y1)
        x2 = int(alpha * prev_line[2] + (1 - alpha) * x2)
        y2 = int(alpha * prev_line[3] + (1 - alpha) * y2)
    return x1, y1, x2, y2

# Hafıza için önceki çizgiler
prev_left_line = None
prev_right_line = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Frame'i yeniden boyutlandır
    frame_resized = cv2.resize(frame, (new_width, new_height))

    # Gri tonlamalı hale getir
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur ile gürültüyü azalt
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Canny kenar algılama
    edges = cv2.Canny(blurred_frame, 250, 250)

    # ROI alanını tanımla
    height, width = edges.shape
    roi_vertices = [(50, height), (width // 2, height // 2 + 50), (width - 50, height)]
    roi_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))

    # Hough dönüşümü ile doğruları tespit et
    lines = cv2.HoughLinesP(roi_edges, 1, math.pi / 180, threshold=31, minLineLength=100, maxLineGap=70)

    # Şeritleri filtrele
    left_lines, right_lines = filter_lines(lines, width)

    # Ortalama çizgileri hesapla
    prev_left_line = average_line(left_lines, prev_left_line)
    prev_right_line = average_line(right_lines, prev_right_line)

    # Çizgileri çiz
    if prev_left_line is not None:
        x1, y1, x2, y2 = prev_left_line
        cv2.line(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 5)
    if prev_right_line is not None:
        x1, y1, x2, y2 = prev_right_line
        cv2.line(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Frame'i kaydet
    out.write(frame_resized)

    # Görüntüyü göster
    cv2.imshow('Şerit Takibi ve İyileştirmeler', frame_resized)

    # Çıkmak için 'q' tuşuna basılmasını bekle
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

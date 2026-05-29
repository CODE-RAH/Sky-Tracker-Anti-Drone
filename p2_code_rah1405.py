import cv2
import numpy as np


class SkyTracker:
    def __init__(self):
        # راه‌اندازی دوربین (0 یعنی وب‌کم پیش‌فرض، می‌تونی آدرس فایل ویدیو هم بدی)
        self.cap = cv2.VideoCapture(0)

        # Background Subtractor برای تشخیص حرکت در آسمان (حذف ابرهای ثابت)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=25,
            detectShadows=False
        )

        # ردیاب CSRT (خیلی دقیق برای اجسام کوچک هوایی)
        self.tracker = None
        self.tracking = False

        # تنظیمات نمایش
        self.box_color = (0, 0, 255)  # قرمز در فرمت BGR
        self.box_thickness = 3

    def detect_moving_object(self, frame):
        """تشخیص اجسام متحرک در آسمان"""
        # تبدیل به grayscale و blur برای حذف نویز
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # اعمال Background Subtraction
        fg_mask = self.bg_subtractor.apply(gray)

        # عملیات مورفولوژی برای پر کردن حفره‌ها و حذف نویز
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # پیدا کردن کانتورها
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_box = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # فیلتر کردن بر اساس اندازه (حذف نویزهای کوچک و ابرهای بزرگ)
            if 100 < area < 20000:  # این اعداد رو می‌تونی تنظیم کنی
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h

                # فیلتر نسبت ابعاد (پهپادها و پرنده‌ها معمولاً مربع یا مستطیل هستند)
                if 0.3 < aspect_ratio < 3.0:
                    if area > max_area:
                        max_area = area
                        best_box = (x, y, w, h)

        return best_box, fg_mask

    def select_roi_manual(self, frame):
        """انتخاب دستی ناحیه با موس"""
        roi = cv2.selectROI("Select Target (Drone/Bird)", frame, fromCenter=False, showCrosshair=True)
        if roi[2] > 0 and roi[3] > 0:  # اگر کاربر انتخاب کرد
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(frame, roi)
            self.tracking = True
        cv2.destroyWindow("Select Target (Drone/Bird)")

    def run(self):
        print("دستورالعمل:")
        print("'a' -> حالت خودکار (تشخیص خودکار هر شیء متحرک)")
        print("'m' -> انتخاب دستی شیء با موس")
        print("'r' -> ریست کردن ردیاب")
        print("'q' -> خروج")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # آینه‌ای کردن تصویر (احساس طبیعی‌تر)
            frame = cv2.flip(frame, 1)

            key = cv2.waitKey(1) & 0xFF

            # کنترل کلیدها
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.select_roi_manual(frame)
            elif key == ord('r'):
                self.tracking = False
                self.tracker = None
            elif key == ord('a'):
                self.tracking = False  # غیرفعال کردن ردیاب قبلی

            if self.tracking and self.tracker is not None:
                # بروزرسانی ردیاب
                success, box = self.tracker.update(frame)

                if success:
                    x, y, w, h = [int(v) for v in box]

                    # رسم کادر قرمز خطر با گوشه‌های تیز
                    cv2.rectangle(frame, (x, y), (x + w, y + h), self.box_color, self.box_thickness)

                    # نوشتن متن خطر
                    cv2.putText(frame, "!!! DANGER !!!", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.box_color, 2)

                    # رسم خطوط ضربدری (شبیه نشانه‌گیری)
                    center_x, center_y = x + w // 2, y + h // 2
                    cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), self.box_color, 2)
                    cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), self.box_color, 2)
                else:
                    # اگر شیء گم شد، دوباره حالت خودکار فعال می‌شه
                    self.tracking = False
                    cv2.putText(frame, "Target Lost - Scanning...", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            else:
                # حالت خودکار: پیدا کردن شیء متحرک
                box, mask = self.detect_moving_object(frame)

                if box:
                    x, y, w, h = box
                    # رسم کادر قرمز
                    cv2.rectangle(frame, (x, y), (x + w, y + h), self.box_color, self.box_thickness)
                    cv2.putText(frame, "AUTO DETECT", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.box_color, 2)

                # نمایش ماسک (برای دیباگ، می‌تونی کامنت کنی)
                # cv2.imshow("Motion Mask", mask)

            # راهنمای روی تصویر
            cv2.putText(frame, "Press 'm' for manual | 'a' for auto | 'q' quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Sky Tracker - Anti Drone System", frame)

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = SkyTracker()
    tracker.run()

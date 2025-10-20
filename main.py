import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import numpy as np
import os


class ImageViewerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Analyzer - YOLO11")
        self.root.geometry("1000x750")
        self.root.configure(bg='#f0f0f0')

        # تنظیم تم
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('Title.TLabel', background='#f0f0f0', font=('Arial', 16, 'bold'))
        self.style.configure('Header.TLabel', background='#f0f0f0', font=('Arial', 12, 'bold'))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('Accent.TButton', background='#007acc', foreground='white')

        # بارگذاری مدل YOLO
        self.model = None
        self.load_model()

        self.setup_ui()

    def load_model(self):
        """بارگذاری مدل YOLO"""
        try:
            self.model = YOLO("YOLO11/yolo11x.pt")
            print("مدل YOLO با موفقیت بارگذاری شد")
        except Exception as e:
            print(f"خطا در بارگذاری مدل: {e}")
            self.model = None

    def setup_ui(self):
        # هدر برنامه
        header_frame = ttk.Frame(self.root, style='TFrame')
        header_frame.pack(fill=tk.X, padx=20, pady=10)

        title_label = ttk.Label(header_frame, text="🎯 AI Image Analyzer", style='Title.TLabel')
        title_label.pack(side=tk.LEFT)

        subtitle_label = ttk.Label(header_frame, text="Powered by YOLO11", foreground='#666666')
        subtitle_label.pack(side=tk.LEFT, padx=(10, 0))

        # فریم اصلی
        main_frame = ttk.Frame(self.root, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # فریم سمت چپ برای کنترل‌ها
        left_frame = ttk.Frame(main_frame, style='TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 15))

        # فریم سمت راست برای نمایش تصویر
        self.right_frame = ttk.Frame(main_frame, style='TFrame')
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # کارت انتخاب تصویر
        file_card = ttk.LabelFrame(left_frame, text="📁 Image Selection", padding=15)
        file_card.pack(fill=tk.X, pady=(0, 15))

        file_input_frame = ttk.Frame(file_card)
        file_input_frame.pack(fill=tk.X)

        self.file_path = tk.StringVar()
        self.file_entry = ttk.Entry(file_input_frame, textvariable=self.file_path, font=('Arial', 10))
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        self.browse_btn = ttk.Button(file_input_frame, text="Browse", command=self.browse_image)
        self.browse_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # برچسب وضعیت تصویر
        self.image_status = ttk.Label(file_card, text="❌ No image selected", foreground="red")
        self.image_status.pack(anchor=tk.W, pady=(10, 0))

        # دکمه‌های اصلی - در دسترس‌تر
        quick_action_frame = ttk.Frame(left_frame, style='TFrame')
        quick_action_frame.pack(fill=tk.X, pady=(0, 15))

        # دکمه پیش‌بینی بزرگ و قابل مشاهده
        self.predict_btn = ttk.Button(quick_action_frame, text="🚀 PREDICT IMAGE",
                                      command=self.analyze_image,
                                      style='Accent.TButton',
                                      width=20)
        self.predict_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=12)

        self.clear_btn = ttk.Button(quick_action_frame, text="🔄 CLEAR",
                                    command=self.clear_form,
                                    width=10)
        self.clear_btn.pack(side=tk.RIGHT, padx=(10, 0), ipady=12)

        # کارت وضعیت مدل
        model_card = ttk.LabelFrame(left_frame, text="⚙️ System Status", padding=15)
        model_card.pack(fill=tk.X, pady=(0, 15))

        model_status = "✅ Model Loaded" if self.model else "❌ Model Not Available"
        model_color = "green" if self.model else "red"
        self.model_status_label = ttk.Label(model_card, text=f"YOLO11: {model_status}", foreground=model_color)
        self.model_status_label.pack(anchor=tk.W)

        # کارت آمار تشخیص
        self.stats_card = ttk.LabelFrame(left_frame, text="📈 Detection Statistics", padding=15)
        self.stats_card.pack(fill=tk.X, pady=(0, 15))

        # آمار کلی
        self.total_objects_label = ttk.Label(self.stats_card, text="Total Objects: 0",
                                             font=('Arial', 11, 'bold'), foreground='#007acc')
        self.total_objects_label.pack(anchor=tk.W)

        self.detection_time_label = ttk.Label(self.stats_card, text="Detection Time: -",
                                              font=('Arial', 10), foreground='#666666')
        self.detection_time_label.pack(anchor=tk.W, pady=(2, 0))

        # کارت نتایج پیش‌بینی
        self.results_card = ttk.LabelFrame(left_frame, text="📊 Analysis Results", padding=15)
        self.results_card.pack(fill=tk.BOTH, expand=True)

        # بهترین پیش‌بینی
        self.top_prediction_frame = ttk.Frame(self.results_card)
        self.top_prediction_frame.pack(fill=tk.X, pady=(0, 15))

        self.prediction_label = ttk.Label(self.top_prediction_frame, text="Top Detection: -",
                                          font=('Arial', 14, 'bold'), foreground='#007acc')
        self.prediction_label.pack(anchor=tk.W)

        self.confidence_label = ttk.Label(self.top_prediction_frame, text="Confidence: -",
                                          font=('Arial', 11), foreground='#666666')
        self.confidence_label.pack(anchor=tk.W, pady=(2, 0))

        # جداکننده
        separator = ttk.Separator(self.results_card, orient='horizontal')
        separator.pack(fill=tk.X, pady=10)

        # عنوان رتبه‌بندی
        ttk.Label(self.results_card, text="🏆 Top Detections:",
                  font=('Arial', 11, 'bold')).pack(anchor=tk.W)

        # فریم برای نمایش رتبه‌بندی با اسکرول
        ranking_container = ttk.Frame(self.results_card)
        ranking_container.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        # ایجاد اسکرول‌بار
        scrollbar = ttk.Scrollbar(ranking_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.ranking_canvas = tk.Canvas(ranking_container, yscrollcommand=scrollbar.set,
                                        bg='#f0f0f0', highlightthickness=0)
        self.ranking_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.ranking_canvas.yview)

        # فریم داخلی برای آیتم‌های رتبه‌بندی
        self.ranking_inner_frame = ttk.Frame(self.ranking_canvas)
        self.ranking_canvas_window = self.ranking_canvas.create_window((0, 0), window=self.ranking_inner_frame,
                                                                       anchor="nw")

        # کارت نمایش تصویر
        image_card = ttk.LabelFrame(self.right_frame, text="🖼️ Image Preview", padding=15)
        image_card.pack(fill=tk.BOTH, expand=True)

        # برچسب برای نمایش تصویر
        self.image_label = ttk.Label(image_card, text="📷 No image selected\n\nSelect or drop an image to analyze",
                                     background="white", relief="solid", justify=tk.CENTER,
                                     font=('Arial', 11), foreground='#666666')
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # متغیر برای ذخیره تصویر
        self.current_image = None
        self.image_tk = None
        self.image_path = None
        self.annotated_image = None

        # تنظیم event binding برای تغییر سایز
        self.ranking_inner_frame.bind("<Configure>", self.on_frame_configure)
        self.ranking_canvas.bind("<Configure>", self.on_canvas_configure)

        # اضافه کردن کلیدهای میانبر
        self.root.bind('<Return>', lambda event: self.analyze_image())
        self.root.bind('<Control-o>', lambda event: self.browse_image())
        self.root.bind('<Control-l>', lambda event: self.clear_form())

    def on_frame_configure(self, event):
        """به روز رسانی scrollregion هنگام تغییر سایز فریم"""
        self.ranking_canvas.configure(scrollregion=self.ranking_canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """به روز رسانی عرض فریم داخلی هنگام تغییر سایز کانواس"""
        self.ranking_canvas.itemconfig(self.ranking_canvas_window, width=event.width)

    def browse_image(self):
        """باز کردن دیالوگ برای انتخاب تصویر"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )

        if file_path:
            self.file_path.set(file_path)
            self.image_path = file_path
            self.load_and_display_image(file_path)

    def load_and_display_image(self, file_path):
        """لود و نمایش تصویر"""
        try:
            # لود تصویر
            self.current_image = Image.open(file_path)

            # تغییر سایز برای نمایش
            display_image = self.resize_image(self.current_image, 500)
            self.image_tk = ImageTk.PhotoImage(display_image)

            # نمایش تصویر
            self.image_label.configure(image=self.image_tk, text="")
            self.image_status.configure(text="✅ Image loaded successfully", foreground="green")

        except Exception as e:
            self.image_status.configure(text=f"❌ Error: {str(e)}", foreground="red")

    def resize_image(self, image, max_size):
        """تغییر سایز تصویر برای نمایش"""
        width, height = image.size

        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def draw_bounding_boxes(self, image, results):
        """رسم bounding box و برچسب‌ها روی تصویر"""
        draw = ImageDraw.Draw(image)

        # فونت برای نوشتن متن
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        if results and len(results) > 0:
            result = results[0]

            if result.boxes and len(result.boxes) > 0:
                boxes = result.boxes

                for box in boxes:
                    # مختصات bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    class_name = self.model.names[class_id]

                    # رنگ بر اساس کلاس
                    color = self.get_color_for_class(class_id)

                    # رسم مستطیل
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                    # متن برچسب
                    label = f"{class_name} {confidence:.2f}"

                    # اندازه متن
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # پس‌زمینه برای متن
                    draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 10, y1],
                                   fill=color)

                    # نوشتن متن
                    draw.text((x1 + 5, y1 - text_height - 2), label, fill='white', font=font)

        return image

    def get_color_for_class(self, class_id):
        """تولید رنگ بر اساس کلاس"""
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
                  '#00FFFF', '#FFA500', '#800080', '#008000', '#800000']
        return colors[class_id % len(colors)]

    def analyze_image(self):
        """آنالیز تصویر با استفاده از YOLO"""
        if self.current_image is None:
            self.image_status.configure(text="❌ Please select an image first", foreground="red")
            return

        if self.model is None:
            self.image_status.configure(text="❌ YOLO model not available", foreground="red")
            return

        try:
            self.image_status.configure(text="⏳ Analyzing image with YOLO...", foreground="orange")
            self.predict_btn.config(state='disabled', text="⏳ PROCESSING...")
            self.root.update()

            # انجام پیش‌بینی با YOLO
            import time
            start_time = time.time()
            results = self.model.predict(source=self.image_path, save=False, verbose=False,conf = 0.2)
            #####😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅
            end_time = time.time()
            detection_time = end_time - start_time

            # پردازش نتایج
            self.process_yolo_results(results, detection_time)

            # نمایش تصویر با bounding box
            self.display_annotated_image(results)

            self.image_status.configure(text="✅ Analysis complete", foreground="green")
            self.predict_btn.config(state='normal', text="🚀 PREDICT IMAGE")

        except Exception as e:
            self.image_status.configure(text=f"❌ Analysis error: {str(e)}", foreground="red")
            self.predict_btn.config(state='normal', text="🚀 PREDICT IMAGE")

    def display_annotated_image(self, results):
        """نمایش تصویر با bounding box"""
        if results and len(results) > 0:
            # لود مجدد تصویر اصلی
            original_image = Image.open(self.image_path)

            # رسم bounding box روی تصویر
            annotated_image = self.draw_bounding_boxes(original_image.copy(), results)

            # تغییر سایز برای نمایش
            display_image = self.resize_image(annotated_image, 500)
            self.annotated_image = ImageTk.PhotoImage(display_image)

            # نمایش تصویر
            self.image_label.configure(image=self.annotated_image, text="")

    def process_yolo_results(self, results, detection_time):
        """پردازش نتایج YOLO و نمایش رتبه‌بندی کلاس‌ها"""
        # پاک کردن رتبه‌بندی قبلی
        self.clear_ranking_display()

        if results and len(results) > 0:
            result = results[0]

            # جمع‌آوری تمام detection‌ها
            class_predictions = {}
            class_counts = {}

            if result.boxes and len(result.boxes) > 0:
                boxes = result.boxes
                total_objects = len(boxes)

                # آمار کلی
                self.total_objects_label.configure(text=f"Total Objects: {total_objects}")
                self.detection_time_label.configure(text=f"Detection Time: {detection_time:.2f}s")

                for box in boxes:
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    class_name = self.model.names[class_id]

                    # شمارش تعداد هر کلاس
                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                    class_counts[class_name] += 1

                    # استفاده از بالاترین confidence برای هر کلاس
                    if class_name not in class_predictions or confidence > class_predictions[class_name]:
                        class_predictions[class_name] = confidence

            # اگر detection پیدا شد
            if class_predictions:
                # مرتب‌سازی کلاس‌ها بر اساس confidence (نزولی)
                sorted_predictions = sorted(class_predictions.items(), key=lambda x: x[1], reverse=True)

                # نمایش بهترین پیش‌بینی
                best_class, best_confidence = sorted_predictions[0]
                count = class_counts.get(best_class, 1)
                self.prediction_label.configure(text=f"Top Detection: {best_class} (x{count})")
                self.confidence_label.configure(text=f"Confidence: {best_confidence:.3f}")

                # نمایش رتبه‌بندی کامل
                self.display_class_rankings(sorted_predictions, class_counts)

            else:
                # اگر هیچ کلاسی تشخیص داده نشد
                self.prediction_label.configure(text="Top Detection: No objects detected")
                self.confidence_label.configure(text="Confidence: -")
                self.total_objects_label.configure(text="Total Objects: 0")
                self.detection_time_label.configure(text=f"Detection Time: {detection_time:.2f}s")
        else:
            self.prediction_label.configure(text="Top Detection: No results")
            self.confidence_label.configure(text="Confidence: -")
            self.total_objects_label.configure(text="Total Objects: 0")

    def display_class_rankings(self, sorted_predictions, class_counts):
        """نمایش رتبه‌بندی کلاس‌ها"""
        # نمایش 10 کلاس برتر
        max_display = min(10, len(sorted_predictions))

        for i, (class_name, confidence) in enumerate(sorted_predictions[:max_display]):
            count = class_counts.get(class_name, 1)

            # ایجاد کارت برای هر آیتم
            item_frame = ttk.Frame(self.ranking_inner_frame, relief='solid', borderwidth=1)
            item_frame.pack(fill=tk.X, pady=2, padx=2)

            # رنگ‌بندی بر اساس رتبه
            if i == 0:
                bg_color = '#e8f5e8'  # سبز برای رتبه اول
            elif i == 1:
                bg_color = '#e8f0ff'  # آبی برای رتبه دوم
            elif i == 2:
                bg_color = '#fff8e8'  # زرد برای رتبه سوم
            else:
                bg_color = '#f8f8f8'  # خاکستری روشن برای بقیه

            # محتوای کارت
            content_frame = ttk.Frame(item_frame, style='TFrame')
            content_frame.pack(fill=tk.X, padx=8, pady=6)

            # شماره رتبه
            rank_label = ttk.Label(content_frame, text=f"{i + 1}", font=('Arial', 12, 'bold'),
                                   width=3, background=bg_color)
            rank_label.pack(side=tk.LEFT)

            # نام کلاس و تعداد
            class_text = f"{class_name} (x{count})"
            class_label = ttk.Label(content_frame, text=class_text, font=('Arial', 10),
                                    background=bg_color, width=18, anchor=tk.W)
            class_label.pack(side=tk.LEFT, padx=(8, 0))

            # مقدار confidence
            confidence_label = ttk.Label(content_frame, text=f"{confidence:.3f}",
                                         font=('Arial', 10, 'bold'), foreground='#007acc',
                                         background=bg_color, width=8)
            confidence_label.pack(side=tk.RIGHT)

            # نوار پیشرفت
            progress = ttk.Progressbar(content_frame, length=80, value=confidence * 100,
                                       maximum=100, style='TProgressbar')
            progress.pack(side=tk.RIGHT, padx=(5, 10))

            # تنظیم رنگ پس‌زمینه برای تمام ویجت‌ها
            for widget in [rank_label, class_label, confidence_label]:
                widget.configure(background=bg_color)

    def clear_ranking_display(self):
        """پاک کردن نمایش رتبه‌بندی"""
        for widget in self.ranking_inner_frame.winfo_children():
            widget.destroy()

    def clear_form(self):
        """پاک کردن فرم"""
        self.file_path.set("")
        self.current_image = None
        self.image_tk = None
        self.annotated_image = None
        self.image_path = None
        self.image_label.configure(image="", text="📷 No image selected\n\nSelect or drop an image to analyze")
        self.image_status.configure(text="❌ No image selected", foreground="red")

        # پاک کردن رتبه‌بندی
        self.clear_ranking_display()

        self.prediction_label.configure(text="Top Detection: -")
        self.confidence_label.configure(text="Confidence: -")
        self.total_objects_label.configure(text="Total Objects: 0")
        self.detection_time_label.configure(text="Detection Time: -")


def main():
    root = tk.Tk()
    app = ImageViewerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
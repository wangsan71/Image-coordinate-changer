import os
import sys
import cv2
import numpy as np
import cvui
import random
import string
from PyQt5.QtWidgets import QApplication, QFileDialog

WINDOW_NAME = "Photo Labeller Pro v3.0"
UI_WIDTH = 300

class PhotoLabeller:
    def __init__(self):
        self.app = QApplication([])
        # --- 啟動設置變數 ---
        self.src_dir = ""
        self.dst_dir = ""
        self.display_scale = [1.0] 
        self.setup_complete = False
        
        # --- 標註資料變數 ---
        self.img_list = []
        self.current_idx = 0
        self.save_count = 0
        self.raw_img = None
        self.display_img = None
        self.scale_factor = 1.0 
        self.orig_x, self.orig_y = -1, -1
        self.hit_x, self.hit_y = -1, -1
        self.pred_x, self.pred_y = -1, -1
        self.status_msg = "Ready"
        
        # --- 模型相關 ---
        self.model = None
        self.device = None
        self.torch_loaded = False
        self.auto_predict = False
        
        cvui.init(WINDOW_NAME)

    def select_src(self):
        path = QFileDialog.getExistingDirectory(None, "Select Source Folder")
        if path: self.src_dir = path

    def select_dst(self):
        path = QFileDialog.getExistingDirectory(None, "Select Destination Folder")
        if path: self.dst_dir = path

    def start_labelling(self):
        if not os.path.isdir(self.src_dir) or not os.path.isdir(self.dst_dir):
            self.status_msg = "Error: Invalid Directories!"
            return
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        self.img_list = [os.path.join(self.src_dir, f) for f in os.listdir(self.src_dir) 
                         if f.lower().endswith(valid_exts)]
        self.img_list.sort()
        if not self.img_list:
            self.status_msg = "Error: No images found!"
            return
        self.setup_complete = True
        self.load_image_data()

    def load_image_data(self):
        if not self.img_list: return
        path = self.img_list[self.current_idx]
        self.raw_img = cv2.imread(path)
        if self.raw_img is None: return
        
        try:
            basename = os.path.basename(path).split('.')[0]
            parts = basename.split('_')
            self.orig_x, self.orig_y = int(parts[0]), int(parts[1])
        except:
            h, w = self.raw_img.shape[:2]
            self.orig_x, self.orig_y = w//2, h//2

        self.hit_x, self.hit_y = self.orig_x, self.orig_y
        self.pred_x, self.pred_y = -1, -1
        
        # 動態計算顯示高度
        base_h = 800 * self.display_scale[0]
        h, w = self.raw_img.shape[:2]
        self.scale_factor = base_h / h
        self.display_img = cv2.resize(self.raw_img, (int(w * self.scale_factor), int(base_h)))
        
        if self.auto_predict and self.model: self.predict_logic()

    def lazy_load_torch(self):
        if self.torch_loaded: return True
        self.status_msg = "System: Loading PyTorch..."
        try:
            global torch, transforms, models
            import torch
            import torchvision.transforms as transforms
            import torchvision.models as models
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.torch_loaded = True
            return True
        except Exception as e:
            self.status_msg = f"Load Error: {str(e)}"
            return False

    def load_model_file(self):
        if not self.lazy_load_torch(): return
        path, _ = QFileDialog.getOpenFileName(None, "Select .pth Model", "", "PTH Files (*.pth)")
        if not path: return
        try:
            self.model = models.resnet18(weights=None)
            self.model.fc = torch.nn.Linear(512, 2)
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model = self.model.to(self.device).eval()
            self.auto_predict = True
            self.status_msg = "Model Loaded"
            self.predict_logic()
        except: self.status_msg = "Model Error"

    def predict_logic(self):
        if self.model is None or self.raw_img is None: return
        from PIL import Image
        img_rgb = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(input_tensor).cpu().numpy().flatten()
        h, w = self.raw_img.shape[:2]
        self.pred_x = int(((112 + out[0] * 112) / 224.0) * w)
        self.pred_y = int(((112 + out[1] * 112) / 224.0) * h)

    def save_and_next(self):
        if self.raw_img is None: return
        uid = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        cv2.imwrite(os.path.join(self.dst_dir, f"{self.hit_x}_{self.hit_y}_{uid}.jpg"), self.raw_img)
        self.save_count += 1
        if self.current_idx < len(self.img_list) - 1:
            self.current_idx += 1
            self.load_image_data()
        else: self.status_msg = "Done!"

    def render_setup(self, frame):
        frame[:] = (50, 50, 50)
        y = 60
        cvui.text(frame, 50, y, "1. Directories Setup", 0.6); y += 40
        cvui.text(frame, 60, y, f"Source: ...{self.src_dir[-30:]}"); y += 30
        if cvui.button(frame, 60, y, "Select Source Folder"): self.select_src()
        y += 60
        cvui.text(frame, 60, y, f"Target: ...{self.dst_dir[-30:]}"); y += 30
        if cvui.button(frame, 60, y, "Select Target Folder"): self.select_dst()
        y += 80
        cvui.text(frame, 50, y, "2. UI Display Scale", 0.6); y += 40
        cvui.counter(frame, 60, y, self.display_scale, 0.1, "%.1f"); y += 80
        if self.src_dir and self.dst_dir:
            if cvui.button(frame, 200, y, 200, 50, "START LABELLING"):
                self.start_labelling()

    def render_main(self):
        img_h, img_w = self.display_img.shape[:2]
        # 使用當前縮放後的高度
        current_h = max(850, img_h)
        frame = np.zeros((current_h, UI_WIDTH + img_w, 3), np.uint8)
        frame[0:img_h, UI_WIDTH:UI_WIDTH+img_w] = self.display_img.copy()

        m = cvui.mouse(WINDOW_NAME) # 獲取當前鼠標位置
        # 使用你提到的 LEFT_BUTTON 和 CLICK 常數來檢測鼠標點擊事件 m.x是鼠標的x座標，m.y是鼠標的y座標，這裡檢查是否在圖像區域內點擊
        if cvui.mouse(WINDOW_NAME, cvui.LEFT_BUTTON, cvui.CLICK) and m.x > UI_WIDTH:
            self.hit_x = int((m.x - UI_WIDTH) / self.scale_factor)
            self.hit_y = int(m.y / self.scale_factor)

        def draw_pt(ox, oy, color, label):# 畫標記點的函數
            if ox < 0: return
            sx, sy = int(ox * self.scale_factor) + UI_WIDTH, int(oy * self.scale_factor)
            cv2.drawMarker(frame, (sx, sy), color, cv2.MARKER_CROSS, 20, 2)
            cv2.putText(frame, f"{label}: {ox},{oy}", (sx+12, sy-12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        draw_pt(self.orig_x, self.orig_y, (0, 0, 255), "Orig")
        if self.hit_x != self.orig_x and self.hit_y != self.orig_y:
            draw_pt(self.hit_x, self.hit_y, (0, 255, 255), "Target") 
        if self.pred_x != -1: draw_pt(self.pred_x, self.pred_y, (255, 0, 255), "AI")

        # 左側控制面板
        panel = frame[:, 0:UI_WIDTH]
        panel[:] = (40, 40, 40)
        y = 30
        cvui.text(frame, 20, y, f"Status: {self.status_msg}", 0.4, 0x00FF00); y += 40
        cvui.text(frame, 20, y, f"Image: {self.current_idx+1}/{len(self.img_list)}"); y += 25
        cvui.text(frame, 20, y, f"Saved: {self.save_count}"); y += 40
        if cvui.button(frame, 20, y, 260, 40, "Save & Next (W)"): self.save_and_next()
        y += 60
        cvui.rect(frame, 15, y, 270, 120, 0x666666)
        cvui.text(frame, 25, y+10, "AI Model")
        if cvui.button(frame, 30, y+40, 240, 30, "Load .pth Model"): self.load_model_file()
        if self.model:
            st = [self.auto_predict]
            cvui.checkbox(frame, 35, y+80, "Auto Predict", st)
            self.auto_predict = st[0]
        y += 140
        if cvui.button(frame, 20, y, 125, 30, "Prev (A)"): # 上一張
            # Prev 按鈕邏輯
            if self.current_idx > 0:
                if cvui.button(frame, 20, y, 125, 30, "Prev (A)"): 
                    self.current_idx -= 1
                    self.load_image_data()
            else:
                # 當在第一張時，按鈕雖然顯示但點擊提示訊息
                if cvui.button(frame, 20, y, 125, 30, "Prev (A)"):
                    self.status_msg = "Notice: This is the FIRST image."
                    
        if cvui.button(frame, 155, y, 125, 30, "Next (D)"): # 下一張
            # Next 按鈕邏輯
            if self.current_idx < len(self.img_list) - 1:
                if cvui.button(frame, 155, y, 125, 30, "Next (D)"): 
                    self.current_idx += 1
                    self.load_image_data()
            else:
                # 當在最後一張時，按鈕雖然顯示但點擊提示訊息
                if cvui.button(frame, 155, y, 125, 30, "Next (D)"):
                    self.status_msg = "Notice: This is the LAST image."
        
        # 修正這裡：使用 current_h 替代缺失的常數
        if cvui.button(frame, 20, current_h-60, 260, 40, "EXIT"): sys.exit()

        cvui.update(WINDOW_NAME)
        cv2.imshow(WINDOW_NAME, frame)

    def run(self):
        setup_frame = np.zeros((600, 600, 3), np.uint8)
        while True:
            if not self.setup_complete:
                self.render_setup(setup_frame)
                cvui.update(WINDOW_NAME)
                cv2.imshow(WINDOW_NAME, setup_frame)
            else:
                self.render_main()
            key = cv2.waitKey(20)
            if key == 27: break
            elif key in [ord('w'), ord('W'), 32]: self.save_and_next()
            elif key in [ord('a'), ord('A')]: 
                self.current_idx = max(0, self.current_idx-1); self.load_image_data()
            elif key in [ord('d'), ord('D')]: 
                self.current_idx = min(len(self.img_list)-1, self.current_idx+1); self.load_image_data()

if __name__ == "__main__":
    PhotoLabeller().run()
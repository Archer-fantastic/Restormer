import sys
import os
import numpy as np
import argparse
import torch.nn as nn
import torch
import torch.nn.functional as F
from datetime import datetime
from skimage import img_as_ubyte
from natsort import natsorted
import yaml

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 现在导入 Denoising 模块
import Denoising.utils as utils

# 导入Restormer模型
from basicsr.models.archs.restormer_arch import Restormer

# 导入PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTreeView, QFileSystemModel, QLabel, QPushButton, QComboBox, 
    QSpinBox, QLineEdit, QTextEdit, QFileDialog, QSplitter, QFrame
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QDir

class DenoiseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Denoise Test GUI')
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化变量
        self.current_image = None
        self.current_image_path = None
        self.model = None
        self.sigma = 25
        self.tile_size = 512
        self.model_type = 'denoise_blind'
        
        # 创建主布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 使用垂直分割器
        self.main_splitter = QSplitter(Qt.Vertical)
        
        # 创建顶部参数设置区域
        self.setup_parameters()
        
        # 创建中间内容区域（文件树和图像显示）
        self.setup_content()
        
        # 创建底部控制台区域
        self.setup_console()
        
        # 设置主布局
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.addWidget(self.param_frame)
        main_layout.addWidget(self.main_splitter)
        
        # 初始化模型
        self.load_model()
        
    def setup_parameters(self):
        """设置参数选择区域"""
        self.param_frame = QFrame()
        param_layout = QHBoxLayout(self.param_frame)
        
        # 模型类型选择
        model_type_label = QLabel('Model Type:')
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(['denoise_blind', 'denoise_non_blind', 'derain', 'deblur'])
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        
        # Denoise 模式选择（仅在 denoise 模型时显示）
        self.denoise_mode_label = QLabel('Denoise Mode:')
        self.denoise_mode_combo = QComboBox()
        self.denoise_mode_combo.addItems(['gray', 'color'])
        self.denoise_mode_combo.currentTextChanged.connect(self.on_denoise_mode_changed)
        
        # Sigma值选择
        sigma_label = QLabel('Sigma:')
        self.sigma_spin = QSpinBox()
        self.sigma_spin.setRange(0, 100)
        self.sigma_spin.setValue(25)
        self.sigma_spin.valueChanged.connect(self.on_sigma_changed)
        
        # Tile大小选择
        tile_label = QLabel('Tile Size:')
        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(64, 1024)
        self.tile_spin.setSingleStep(64)
        self.tile_spin.setValue(512)
        self.tile_spin.valueChanged.connect(self.on_tile_changed)
        
        # 处理按钮
        process_button = QPushButton('Process Image')
        process_button.clicked.connect(self.process_image)
        
        # 保存按钮
        save_button = QPushButton('Save Image')
        save_button.clicked.connect(self.save_image)
        
        # 文件夹选择按钮
        folder_button = QPushButton('Select Folder')
        folder_button.clicked.connect(self.select_folder)
        
        # 添加到布局
        param_layout.addWidget(model_type_label)
        param_layout.addWidget(self.model_type_combo)
        param_layout.addWidget(self.denoise_mode_label)
        param_layout.addWidget(self.denoise_mode_combo)
        param_layout.addWidget(sigma_label)
        param_layout.addWidget(self.sigma_spin)
        param_layout.addWidget(tile_label)
        param_layout.addWidget(self.tile_spin)
        param_layout.addWidget(folder_button)
        param_layout.addWidget(process_button)
        param_layout.addWidget(save_button)
        param_layout.addStretch()
        
        self.main_splitter.addWidget(self.param_frame)
    
    def setup_content(self):
        """设置文件树和图像显示区域"""
        content_frame = QWidget()
        content_layout = QHBoxLayout(content_frame)
        
        # 创建文件系统模型和树视图
        self.file_model = QFileSystemModel()
        # 不设置默认路径，初始化为空
        self.file_tree = QTreeView()
        self.file_tree.setModel(self.file_model)
        self.file_tree.setColumnWidth(0, 250)
        self.file_tree.clicked.connect(self.on_file_selected)
        
        # 创建图像显示区域
        image_frame = QWidget()
        image_layout = QHBoxLayout(image_frame)
        # 设置图像布局的伸缩因子，使两个滚动区域能够平分空间
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(10)
        
        # 原图显示（带滚动和缩放）
        from PyQt5.QtWidgets import QScrollArea
        
        # 原图滚动区域
        self.original_scroll = QScrollArea()
        self.original_scroll.setWidgetResizable(True)  # 设置为True以自适应窗口大小
        self.original_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.original_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.original_scroll.setMinimumSize(400, 400)
        self.original_label = QLabel('Original Image')
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 400)
        self.original_label.setScaledContents(True)  # 自动缩放内容以适应标签大小
        self.original_label.wheelEvent = self.on_wheel_event
        self.original_label.mousePressEvent = self.on_mouse_press
        self.original_label.mouseMoveEvent = self.on_mouse_move
        self.original_label.mouseReleaseEvent = self.on_mouse_release
        self.original_scroll.setWidget(self.original_label)
        
        # 效果图滚动区域
        self.processed_scroll = QScrollArea()
        self.processed_scroll.setWidgetResizable(True)  # 设置为True以自适应窗口大小
        self.processed_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.processed_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.processed_scroll.setMinimumSize(400, 400)
        self.processed_label = QLabel('Processed Image')
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setMinimumSize(400, 400)
        self.processed_label.setScaledContents(True)  # 自动缩放内容以适应标签大小
        self.processed_label.wheelEvent = self.on_wheel_event
        self.processed_label.mousePressEvent = self.on_mouse_press
        self.processed_label.mouseMoveEvent = self.on_mouse_move
        self.processed_label.mouseReleaseEvent = self.on_mouse_release
        self.processed_scroll.setWidget(self.processed_label)
        
        # 拖动状态
        self.is_dragging = False
        self.last_mouse_pos = None
        
        # 缩放因子
        self.zoom_factor = 1.0
        
        # 添加到布局并设置伸缩因子
        image_layout.addWidget(self.original_scroll, 1)  # 伸缩因子为1
        image_layout.addWidget(self.processed_scroll, 1)  # 伸缩因子为1
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.file_tree)
        splitter.addWidget(image_frame)
        splitter.setSizes([300, 900])
        # 设置拉伸因子，使图像显示区域能够自适应窗口大小
        splitter.setStretchFactor(0, 0)  # 文件树不拉伸
        splitter.setStretchFactor(1, 1)  # 图像显示区域拉伸
        
        content_layout.addWidget(splitter)
        self.main_splitter.addWidget(content_frame)
    
    def on_wheel_event(self, event):
        """处理鼠标滚轮事件，实现图像缩放"""
        # 获取滚轮方向
        delta = event.angleDelta().y()
        if delta > 0:
            # 放大
            self.zoom_factor *= 1.1
        else:
            # 缩小
            self.zoom_factor /= 1.1
        
        # 限制缩放范围
        self.zoom_factor = max(0.1, min(self.zoom_factor, 5.0))
        
        # 应用缩放（基于原始图像）
        self.apply_zoom()
    
    def on_mouse_press(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.last_mouse_pos = event.pos()
    
    def on_mouse_move(self, event):
        """鼠标移动事件"""
        if self.is_dragging and self.last_mouse_pos:
            # 获取当前鼠标所在的滚动区域
            current_scroll = None
            other_scroll = None
            if self.original_label.underMouse():
                current_scroll = self.original_scroll
                other_scroll = self.processed_scroll
            elif self.processed_label.underMouse():
                current_scroll = self.processed_scroll
                other_scroll = self.original_scroll
            
            if current_scroll and other_scroll:
                # 计算拖动距离
                delta = event.pos() - self.last_mouse_pos
                
                # 获取当前的滚动位置
                h_bar = current_scroll.horizontalScrollBar()
                v_bar = current_scroll.verticalScrollBar()
                
                # 反向滚动以实现拖动效果
                h_bar.setValue(h_bar.value() - delta.x())
                v_bar.setValue(v_bar.value() - delta.y())
                
                # 同步另一张图像的滚动位置
                other_h_bar = other_scroll.horizontalScrollBar()
                other_v_bar = other_scroll.verticalScrollBar()
                other_h_bar.setValue(h_bar.value())
                other_v_bar.setValue(v_bar.value())
                
                self.last_mouse_pos = event.pos()
    
    def on_mouse_release(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            self.last_mouse_pos = None
    
    def apply_zoom(self):
        """应用缩放因子到两个图像"""
        # 缩放原图
        if hasattr(self, 'original_pixmap') and not self.original_pixmap.isNull():
            # 使用整数进行缩放
            scaled_width = int(self.original_pixmap.width() * self.zoom_factor)
            scaled_height = int(self.original_pixmap.height() * self.zoom_factor)
            scaled_pixmap = self.original_pixmap.scaled(
                scaled_width,
                scaled_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.original_label.setPixmap(scaled_pixmap)
            # 调整标签大小以适应pixmap
            self.original_label.setMinimumSize(scaled_pixmap.size())
        
        # 缩放学效果图
        if hasattr(self, 'processed_pixmap') and not self.processed_pixmap.isNull():
            # 使用整数进行缩放
            scaled_width = int(self.processed_pixmap.width() * self.zoom_factor)
            scaled_height = int(self.processed_pixmap.height() * self.zoom_factor)
            scaled_pixmap = self.processed_pixmap.scaled(
                scaled_width,
                scaled_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.processed_label.setPixmap(scaled_pixmap)
            # 调整标签大小以适应pixmap
            self.processed_label.setMinimumSize(scaled_pixmap.size())
    
    def display_image(self, img, label, is_original=False):
        """显示图像"""
        try:
            # 转换为QImage
            if len(img.shape) == 2:
                # 灰度图像
                height, width = img.shape
                bytes_per_line = width
                # 将 memoryview 转换为 bytes
                img_data = bytes(img.data)
                q_image = QImage(img_data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            elif len(img.shape) == 3:
                # 彩色图像或带通道的灰度图像
                height, width, channels = img.shape
                if channels == 1:
                    # 单通道灰度图像
                    bytes_per_line = width
                    # 将 memoryview 转换为 bytes
                    img_data = bytes(img.squeeze().data)
                    q_image = QImage(img_data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                else:
                    # 彩色图像
                    bytes_per_line = channels * width
                    # 将 memoryview 转换为 bytes
                    img_data = bytes(img.data)
                    q_image = QImage(img_data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                self.log(f"Error: Unsupported image shape: {img.shape}")
                return
            
            # 转换为QPixmap并显示
            pixmap = QPixmap.fromImage(q_image)
            if not pixmap.isNull():
                # 保存原始pixmap用于缩放
                if is_original:
                    self.original_pixmap = pixmap
                else:
                    self.processed_pixmap = pixmap
                
                # 应用当前缩放因子
                # 使用整数进行缩放
                scaled_width = int(pixmap.width() * self.zoom_factor)
                scaled_height = int(pixmap.height() * self.zoom_factor)
                scaled_pixmap = pixmap.scaled(
                    scaled_width,
                    scaled_height,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                label.setPixmap(scaled_pixmap)
                
                # 调整标签大小以适应pixmap
                label.setMinimumSize(scaled_pixmap.size())
            else:
                self.log("Error: Failed to create pixmap from image")
                # 显示错误信息
                label.setText("Error: Failed to display image")
        except Exception as e:
            self.log(f"Error displaying image: {e}")
            import traceback
            traceback.print_exc()
    
    def setup_console(self):
        """设置控制台输出区域"""
        console_frame = QFrame()
        console_layout = QVBoxLayout(console_frame)
        
        console_label = QLabel('Console Output:')
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setMinimumHeight(150)
        
        console_layout.addWidget(console_label)
        console_layout.addWidget(self.console_text)
        
        self.main_splitter.addWidget(console_frame)
    
    def log(self, message):
        """向控制台输出日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.console_text.append(f"[{timestamp}] {message}")
        self.console_text.verticalScrollBar().setValue(self.console_text.verticalScrollBar().maximum())
    
    def load_model(self):
        """加载模型"""
        try:
            self.log("Loading model...")
            
            # 选择配置文件和权重文件
            if self.model_type == 'denoise_blind':
                # 根据 denoise 模式选择配置文件
                if hasattr(self, 'denoise_mode') and self.denoise_mode == 'color':
                    yaml_file = 'GaussianColorDenoising_Restormer.yml'
                    weights_file = 'gaussian_color_denoising_blind.pth'
                else:
                    yaml_file = 'GaussianGrayDenoising_Restormer.yml'
                    weights_file = 'gaussian_gray_denoising_blind.pth'
                weights_dir = 'Denoising/pretrained_models'
            elif self.model_type == 'denoise_non_blind':
                # 根据 denoise 模式选择配置文件
                if hasattr(self, 'denoise_mode') and self.denoise_mode == 'color':
                    yaml_file = f'GaussianColorDenoising_RestormerSigma{self.sigma}.yml'
                    weights_file = f'gaussian_color_denoising_sigma{self.sigma}.pth'
                else:
                    yaml_file = f'GaussianGrayDenoising_RestormerSigma{self.sigma}.yml'
                    weights_file = f'gaussian_gray_denoising_sigma{self.sigma}.pth'
                weights_dir = 'Denoising/pretrained_models'
            elif self.model_type == 'derain':
                yaml_file = 'Deraining_Restormer.yml'
                weights_file = 'deraining.pth'
                weights_dir = 'Deraining/pretrained_models'
            elif self.model_type == 'deblur':
                yaml_file = 'Deblurring_Restormer.yml'
                weights_file = 'motion_deblurring.pth'
                weights_dir = 'Motion_Deblurring/pretrained_models'
            else:
                self.log(f"Error: Unknown model type: {self.model_type}")
                return
            
            # 构建绝对路径 - 使用项目根目录
            if self.model_type in ['denoise_blind', 'denoise_non_blind']:
                yaml_file = os.path.join(project_root, 'Denoising', 'Options', 'Origin', yaml_file)
            elif self.model_type == 'derain':
                yaml_file = os.path.join(project_root, 'Deraining', 'Options', yaml_file)
            elif self.model_type == 'deblur':
                yaml_file = os.path.join(project_root, 'Motion_Deblurring', 'Options', yaml_file)
            
            weights_path = os.path.join(project_root, weights_dir, weights_file)
            
            # 打印调试信息
            self.log(f"Looking for YAML file at: {yaml_file}")
            self.log(f"Looking for weights file at: {weights_path}")
            
            if not os.path.exists(yaml_file):
                self.log(f"Error: YAML file not found: {yaml_file}")
                return
            
            # 加载配置
            with open(yaml_file, mode='r', encoding='utf-8') as f:
                x = yaml.load(f, Loader=yaml.FullLoader)
            
            # 移除'type'字段，因为Restormer.__init__不接受这个参数
            network_g = x['network_g'].copy()
            if 'type' in network_g:
                del network_g['type']
            
            # 创建模型
            self.model = Restormer(**network_g)
            
            if not os.path.exists(weights_path):
                self.log(f"Error: Weights file not found: {weights_path}")
                return
            
            checkpoint = torch.load(weights_path)
            self.model.load_state_dict(checkpoint['params'])
            
            # 移动到GPU
            self.model.cuda()
            self.model = nn.DataParallel(self.model)
            self.model.eval()
            
            self.log(f"Model loaded successfully: {weights_path}")
        except Exception as e:
            self.log(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
    

    
    def on_model_type_changed(self, text):
        """模型类型改变时的处理"""
        self.model_type = text
        self.log(f"Model type changed to: {text}")
        
        # 根据模型类型显示或隐藏 denoise 模式选择控件
        if text in ['denoise_blind', 'denoise_non_blind']:
            # denoise 模型，显示模式选择
            self.denoise_mode_label.show()
            self.denoise_mode_combo.show()
        else:
            # 非 denoise 模型，隐藏模式选择
            self.denoise_mode_label.hide()
            self.denoise_mode_combo.hide()
        
        self.load_model()
    
    def on_denoise_mode_changed(self, text):
        """Denoise 模式改变时的处理"""
        self.denoise_mode = text
        self.log(f"Denoise mode changed to: {text}")
        # 重新加载模型以适应新模式
        if self.model_type in ['denoise_blind', 'denoise_non_blind']:
            self.load_model()
    
    def on_sigma_changed(self, value):
        """Sigma值改变时的处理"""
        self.sigma = value
        self.log(f"Sigma changed to: {value}")
        if self.model_type == 'denoise_non_blind':
            self.load_model()
    
    def on_tile_changed(self, value):
        """Tile大小改变时的处理"""
        self.tile_size = value
        self.log(f"Tile size changed to: {value}")
    
    def select_folder(self):
        """选择文件夹"""
        from PyQt5.QtWidgets import QFileDialog
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.log(f"Selected folder: {folder_path}")
            # 刷新文件树
            self.file_model.setRootPath(folder_path)
            self.file_tree.setRootIndex(self.file_model.index(folder_path))
    
    def on_file_selected(self, index):
        """选择文件时的处理"""
        file_path = self.file_model.filePath(index)
        if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']):
            self.current_image_path = file_path
            self.load_image(file_path)
    
    def setup_content(self):
        """设置文件树和图像显示区域"""
        content_frame = QWidget()
        content_layout = QHBoxLayout(content_frame)
        
        # 创建文件系统模型和树视图
        self.file_model = QFileSystemModel()
        # 不设置默认路径，初始化为空
        self.file_tree = QTreeView()
        self.file_tree.setModel(self.file_model)
        self.file_tree.setColumnWidth(0, 250)
        self.file_tree.clicked.connect(self.on_file_selected)
        
        # 创建图像显示区域
        image_frame = QWidget()
        image_layout = QHBoxLayout(image_frame)
        # 设置图像布局的伸缩因子，使两个滚动区域能够平分空间
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(10)
        
        # 原图显示（带滚动和缩放）
        from PyQt5.QtWidgets import QScrollArea
        
        # 原图滚动区域
        self.original_scroll = QScrollArea()
        self.original_scroll.setWidgetResizable(True)  # 设置为True以自适应窗口大小
        self.original_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.original_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.original_scroll.setMinimumSize(400, 400)
        self.original_label = QLabel('Original Image')
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 400)
        self.original_label.setScaledContents(True)  # 自动缩放内容以适应标签大小
        self.original_label.wheelEvent = self.on_wheel_event
        self.original_label.mousePressEvent = self.on_mouse_press
        self.original_label.mouseMoveEvent = self.on_mouse_move
        self.original_label.mouseReleaseEvent = self.on_mouse_release
        self.original_scroll.setWidget(self.original_label)
        
        # 效果图滚动区域
        self.processed_scroll = QScrollArea()
        self.processed_scroll.setWidgetResizable(True)  # 设置为True以自适应窗口大小
        self.processed_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.processed_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.processed_scroll.setMinimumSize(400, 400)
        self.processed_label = QLabel('Processed Image')
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setMinimumSize(400, 400)
        self.processed_label.setScaledContents(True)  # 自动缩放内容以适应标签大小
        self.processed_label.wheelEvent = self.on_wheel_event
        self.processed_label.mousePressEvent = self.on_mouse_press
        self.processed_label.mouseMoveEvent = self.on_mouse_move
        self.processed_label.mouseReleaseEvent = self.on_mouse_release
        self.processed_scroll.setWidget(self.processed_label)
        
        # 拖动状态
        self.is_dragging = False
        self.last_mouse_pos = None
        
        # 缩放因子
        self.zoom_factor = 1.0
        
        # 添加到布局并设置伸缩因子
        image_layout.addWidget(self.original_scroll, 1)  # 伸缩因子为1
        image_layout.addWidget(self.processed_scroll, 1)  # 伸缩因子为1
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.file_tree)
        splitter.addWidget(image_frame)
        splitter.setSizes([300, 900])
        # 设置拉伸因子，使图像显示区域能够自适应窗口大小
        splitter.setStretchFactor(0, 0)  # 文件树不拉伸
        splitter.setStretchFactor(1, 1)  # 图像显示区域拉伸
        
        content_layout.addWidget(splitter)
        
        self.main_splitter.addWidget(content_frame)
    
    def load_image(self, file_path):
        """加载图像"""
        try:
            self.log(f"Loading image: {file_path}")
            
            # 根据模型类型和模式选择图像加载函数
            if self.model_type in ['derain', 'deblur']:
                # derain 和 deblur 使用彩色图像
                img = utils.load_img(file_path)
                self.log(f"Loaded color image for {self.model_type}")
            else:
                # denoise 根据模式选择图像类型
                if hasattr(self, 'denoise_mode') and self.denoise_mode == 'color':
                    # 彩色降噪
                    img = utils.load_img(file_path)
                    self.log("Loaded color image for denoise")
                else:
                    # 灰度降噪
                    img = utils.load_gray_img(file_path)
                    self.log("Loaded gray image for denoise")
            
            self.current_image = img
            
            # 显示原图，传递is_original=True
            self.display_image(img, self.original_label, is_original=True)
            self.log("Image loaded successfully")
        except Exception as e:
            self.log(f"Error loading image: {e}")
    
    def save_image(self):
        """保存处理后的图像"""
        if not hasattr(self, 'processed_image'):
            self.log("No processed image to save")
            return
        
        try:
            # 弹出文件保存对话框
            from PyQt5.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Image", "", "PNG Files (*.png);;All Files (*)"
            )
            
            if file_path:
                # 确保文件扩展名为.png
                if not file_path.lower().endswith('.png'):
                    file_path += '.png'
                
                # 保存图像
                self.log(f"Saving image to: {file_path}")
                
                # 根据模型类型和模式选择图像保存函数
                if self.model_type in ['derain', 'deblur']:
                    # derain 和 deblur 使用彩色图像保存
                    utils.save_img(file_path, self.processed_image)
                    self.log(f"Saved color image for {self.model_type}")
                else:
                    # denoise 根据模式选择保存方式
                    if hasattr(self, 'denoise_mode') and self.denoise_mode == 'color':
                        # 彩色降噪，使用彩色图像保存
                        utils.save_img(file_path, self.processed_image)
                        self.log("Saved color image for denoise")
                    else:
                        # 灰度降噪，使用灰度图像保存
                        utils.save_gray_img(file_path, self.processed_image)
                        self.log("Saved gray image for denoise")
                
                self.log("Image saved successfully!")
        except Exception as e:
            self.log(f"Error saving image: {e}")
            import traceback
            traceback.print_exc()
    

    
    def process_image(self):
        """处理图像"""
        if self.current_image is None or not self.model:
            self.log("Please select an image and ensure model is loaded")
            return
        
        try:
            self.log("Processing image...")
            
            # 输出图像信息
            height, width, channels = self.current_image.shape
            self.log(f"Image size: {width}x{height}x{channels}")
            self.log(f"Tile size: {self.tile_size}")
            self.log(f"Sigma: {self.sigma}")
            self.log(f"Model type: {self.model_type}")
            
            # 准备图像
            self.log("Preparing image...")
            
            # 根据模型类型和模式决定是否添加噪声
            if self.model_type in ['derain', 'deblur']:
                # derain 和 deblur 不需要添加噪声，并且需要 3 通道彩色图像
                img = np.float32(self.current_image) / 255.
                
                # 确保图像是 3 通道彩色图像
                if len(img.shape) == 2 or img.shape[2] == 1:
                    # 将灰度图像转换为 3 通道
                    img = np.repeat(img[..., np.newaxis], 3, axis=2)
                    self.log("Converted grayscale image to 3-channel color image")
                
                self.log(f"Processing {self.model_type} image (no noise added)")
            else:
                # 降噪模型需要添加噪声
                img = np.float32(self.current_image) / 255.
                self.log("Adding noise...")
                np.random.seed(seed=0)  # for reproducibility
                img = img + np.random.normal(0, self.sigma/255., img.shape)
                
                # 确保图像通道数正确
                if hasattr(self, 'denoise_mode') and self.denoise_mode == 'color':
                    # 彩色降噪，确保 3 通道
                    if len(img.shape) == 2 or img.shape[2] == 1:
                        img = np.repeat(img[..., np.newaxis], 3, axis=2)
                        self.log("Converted grayscale image to 3-channel color image for denoise")
                    self.log(f"Processing color denoising image with sigma={self.sigma}")
                else:
                    # 灰度降噪，确保 1 通道
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        # 将彩色图像转换为灰度
                        img = np.mean(img, axis=2, keepdims=True)
                        self.log("Converted color image to grayscale for denoise")
                    self.log(f"Processing grayscale denoising image with sigma={self.sigma}")
            
            # 检查并处理图像维度
            self.log(f"Image shape before processing: {img.shape}")
            
            # 如果图像是 4 维的，移除批次维度
            if len(img.shape) == 4:
                img = img.squeeze(0)
                self.log(f"Removed batch dimension, new shape: {img.shape}")
            
            # 确保图像是 3 维的
            if len(img.shape) != 3:
                self.log(f"Error: Image shape is {img.shape}, expected 3 dimensions")
                return
            
            # 再次检查图像维度
            self.log(f"Image shape before tensor conversion: {img.shape}")
            
            # 确保图像是 3 维的
            if len(img.shape) != 3:
                # 尝试将图像转换为 3 维
                if len(img.shape) == 2:
                    # 2 维灰度图像，添加通道维度
                    img = img[..., np.newaxis]
                    self.log(f"Added channel dimension, new shape: {img.shape}")
                elif len(img.shape) == 4:
                    # 4 维图像，移除批次维度
                    img = img.squeeze(0)
                    self.log(f"Removed batch dimension, new shape: {img.shape}")
                
                # 再次检查
                if len(img.shape) != 3:
                    self.log(f"Error: Image shape is {img.shape}, expected 3 dimensions")
                    return
            
            # 转换为张量并确保类型为float32
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            input_ = img_tensor.unsqueeze(0).cuda()
            
            # 打印输入张量形状
            self.log(f"Input tensor shape: {input_.shape}")
            
            # 填充以确保是8的倍数
            factor = 8
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
            
            # 处理图像
            self.log("Processing with model...")
            with torch.no_grad():
                if self.tile_size is None:
                    # 整体处理
                    self.log("Using full image processing")
                    restored = self.model(input_)
                else:
                    # 分块处理
                    self.log(f"Using tile processing with size: {self.tile_size}")
                    b, c, h, w = input_.shape
                    tile = min(self.tile_size, h, w)
                    assert tile % 8 == 0, "tile size should be multiple of 8"
                    tile_overlap = 32
                    
                    stride = tile - tile_overlap
                    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
                    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
                    total_patches = len(h_idx_list) * len(w_idx_list)
                    processed_patches = 0
                    
                    self.log(f"Total patches: {total_patches}")
                    
                    E = torch.zeros(b, c, h, w).type_as(input_)
                    W = torch.zeros_like(E)
                    
                    for h_idx in h_idx_list:
                        for w_idx in w_idx_list:
                            in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                            out_patch = self.model(in_patch)
                            out_patch_mask = torch.ones_like(out_patch)
                            
                            E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                            W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
                            
                            processed_patches += 1
                            progress = (processed_patches / total_patches) * 100
                            if processed_patches % 5 == 0 or processed_patches == total_patches:
                                self.log(f"Processing patch {processed_patches}/{total_patches} ({progress:.1f}%)")
                    restored = E.div_(W)
            
            # 去除填充
            restored = restored[:, :, :h, :w]
            
            # 转换为numpy数组
            self.log("Converting result...")
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            restored = img_as_ubyte(restored)
            
            # 存储处理后的图像数据
            self.processed_image = restored
            
            # 显示处理后的图像
            self.display_image(restored, self.processed_label, is_original=False)
            
            self.log("Processing completed successfully!")
        except Exception as e:
            self.log(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DenoiseGUI()
    window.show()
    sys.exit(app.exec_())

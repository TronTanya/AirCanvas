"""AirCanvas main window (PySide6). All widget updates happen in the GUI thread."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from camera_worker import CameraWorker, FrameResult
from canvas import CANVAS_HEIGHT, CANVAS_WIDTH, Color, StrokeCanvas
from diagnostics import PerformanceMonitor, TrialRecorder, export_csv
from gestures import (
    DEFAULT_PRESS_THRESHOLD,
    DEFAULT_RELEASE_THRESHOLD,
    DEFAULT_SMOOTHING_ALPHA,
    GestureInterpreter,
    smoothing_time_constant,
)
from hand_tracker import MODEL_PATH, SETUP_HINT, model_available

BG = "#F7F8FA"
SURFACE = "#FFFFFF"
TEXT = "#202733"
TEXT_2 = "#667085"
BORDER = "#E4E7EC"
ACCENT = "#315EFB"

PALETTE: list[tuple[str, str]] = [
    ("Тёмно-серый", "#202733"),
    ("Синий", "#315EFB"),
    ("Красный", "#E5484D"),
    ("Зелёный", "#2E9E5B"),
    ("Оранжевый", "#F08C00"),
    ("Фиолетовый", "#7C3AED"),
]

RESOLUTIONS: list[tuple[int, int]] = [(640, 480), (960, 540), (1280, 720)]

INSTRUCTION = (
    "Сведите большой и указательный пальцы, чтобы рисовать. "
    "Разведите их, чтобы остановиться."
)

STYLE = f"""
QWidget {{ color: {TEXT}; font-size: 13px; }}
QMainWindow, #central {{ background: {BG}; }}
#header, #toolbar, #sidePanel {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px; }}
#title {{ font-size: 20px; font-weight: 600; }}
#subtitle, .secondary {{ color: {TEXT_2}; }}
QLabel[secondary="true"] {{ color: {TEXT_2}; }}
QLabel[warning="true"] {{ color: #B54708; }}
QLabel[error="true"] {{ color: #D92D20; }}
QPushButton {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 6px 12px; min-height: 18px;
}}
QPushButton:hover {{ border-color: #C7CCD5; background: #F2F4F7; }}
QPushButton:pressed {{ background: #E9ECF1; }}
QPushButton:disabled {{ color: #A8AFBD; background: #F9FAFB; }}
QPushButton#primary {{ background: {ACCENT}; color: white; border-color: {ACCENT}; }}
QPushButton#primary:hover {{ background: #2A51DB; }}
QPushButton#primary:disabled {{ background: #A9BBFD; border-color: #A9BBFD; }}
QPushButton:checked {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton[swatch="true"] {{ padding: 0; min-width: 22px; max-width: 22px; min-height: 22px; max-height: 22px; border-radius: 11px; border: 2px solid {SURFACE}; }}
QPushButton[swatch="true"]:checked {{ border: 2px solid {ACCENT}; }}
QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 4px; padding: 4px 6px;
}}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 6px; margin-top: 14px; padding: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {TEXT_2}; }}
QStatusBar {{ background: {SURFACE}; border-top: 1px solid {BORDER}; }}
QStatusBar QLabel {{ padding: 2px 10px; }}
#preview {{ background: #EEF0F3; border: 1px solid {BORDER}; border-radius: 4px; color: {TEXT_2}; }}
#modeBadge {{ border: 1px solid {BORDER}; border-radius: 4px; padding: 4px 8px; color: {TEXT_2}; }}
"""


def hex_to_rgb(value: str) -> Color:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def role_label(text: str, role: str) -> QLabel:
    """QLabel styled by a dynamic property: secondary, warning or error."""
    label = QLabel(text)
    label.setProperty(role, True)
    return label


def list_cameras() -> list[tuple[int, str]]:
    """Camera indices with human-readable names when Qt can provide them."""
    try:
        from PySide6.QtMultimedia import QMediaDevices

        devices = QMediaDevices.videoInputs()
        if devices:
            return [(i, f"Камера {i}: {d.description()}") for i, d in enumerate(devices)]
    except Exception:  # noqa: BLE001 - QtMultimedia is optional
        pass
    return [(i, f"Камера {i}") for i in range(3)]


class CanvasView(QWidget):
    """Displays the canvas scaled to fit (aspect preserved) plus a cursor overlay.

    The cursor overlay is painted only on screen; it is never part of the
    canvas image and is never saved.
    """

    mouse_pressed = Signal(QPointF)
    mouse_moved = Signal(QPointF)
    mouse_released = Signal()

    def __init__(self, canvas: StrokeCanvas, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.canvas = canvas
        self._image: QImage | None = None
        self._image_version = -1
        self.cursor_pos: QPointF | None = None
        self.cursor_active = False
        self.cursor_color = QColor(TEXT)
        self.cursor_radius = 4.0
        self.mouse_enabled = True
        self._mouse_down = False
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(480, 270)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def sizeHint(self) -> QSize:
        return QSize(960, 540)

    def image_rect(self) -> QRectF:
        w, h = self.width() - 2, self.height() - 2
        scale = min(w / self.canvas.width, h / self.canvas.height)
        dw, dh = self.canvas.width * scale, self.canvas.height * scale
        return QRectF((self.width() - dw) / 2, (self.height() - dh) / 2, dw, dh)

    def to_canvas(self, pos: QPointF) -> QPointF:
        r = self.image_rect()
        x = (pos.x() - r.x()) / r.width() * self.canvas.width
        y = (pos.y() - r.y()) / r.height() * self.canvas.height
        return QPointF(x, y)

    def to_widget(self, pos: QPointF) -> QPointF:
        r = self.image_rect()
        return QPointF(
            r.x() + pos.x() / self.canvas.width * r.width(),
            r.y() + pos.y() / self.canvas.height * r.height(),
        )

    def set_cursor_overlay(self, pos: QPointF | None, active: bool) -> None:
        self.cursor_pos = pos
        self.cursor_active = active
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._image_version != self.canvas.version or self._image is None:
            arr = self.canvas.render()
            h, w = arr.shape[:2]
            self._image = QImage(arr.data, w, h, arr.strides[0], QImage.Format.Format_RGB888).copy()
            self._image_version = self.canvas.version
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.image_rect()
        p.drawImage(r, self._image)
        p.setPen(QPen(QColor(BORDER), 1))
        p.drawRect(r.adjusted(-0.5, -0.5, 0.5, 0.5))
        if self.cursor_pos is not None:
            scale = r.width() / self.canvas.width
            c = self.to_widget(self.cursor_pos)
            radius = max(4.0, self.cursor_radius * scale)
            if self.cursor_active:
                p.setPen(QPen(QColor("white"), 1.5))
                p.setBrush(self.cursor_color)
            else:
                p.setPen(QPen(self.cursor_color, 1.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, radius, radius)
            p.setPen(QPen(QColor(ACCENT), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, radius + 3, radius + 3)
        p.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if not self.mouse_enabled or event.button() != Qt.MouseButton.LeftButton:
            return
        if not self.image_rect().contains(event.position()):
            return
        self._mouse_down = True
        self.mouse_pressed.emit(self.to_canvas(event.position()))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self.mouse_enabled:
            return
        pos = self.to_canvas(event.position())
        self.set_cursor_overlay(pos, self._mouse_down)
        if self._mouse_down:
            self.mouse_moved.emit(pos)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._mouse_down and event.button() == Qt.MouseButton.LeftButton:
            self._mouse_down = False
            self.mouse_released.emit()

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self.mouse_enabled and not self._mouse_down:
            self.set_cursor_overlay(None, False)

    def cancel_mouse(self) -> None:
        if self._mouse_down:
            self._mouse_down = False
            self.mouse_released.emit()


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("О проекте")
        self.resize(620, 560)
        text = QTextBrowser()
        text.setOpenExternalLinks(True)
        text.setHtml(ABOUT_HTML)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("Закрыть")
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(text)
        layout.addWidget(buttons)


ABOUT_HTML = f"""
<h2 style="margin-bottom:2px">AirCanvas</h2>
<p style="color:{TEXT_2};margin-top:0">Рисование жестами перед камерой</p>

<h3>Цель</h3>
<p>Показать, что бесконтактное рисование можно реализовать с обычной веб-камерой,
без специальных датчиков и без облачных сервисов.</p>
<p><b>Гипотеза:</b> «Отслеживание руки и сглаживание координат позволяют реализовать
бесконтактное рисование с помощью обычной веб-камеры.»</p>

<h3>Технологии</h3>
<p>Python 3.11, OpenCV (захват видео), MediaPipe Hand Landmarker (Tasks API),
PySide6 (интерфейс), NumPy. Всё работает локально на процессоре.</p>

<h3>Обнаружение ключевых точек руки</h3>
<p>Модель MediaPipe находит на каждом кадре 21 ключевую точку руки. Кончик
указательного пальца (точка 8) используется как курсор. Изображение и координаты
отражены по горизонтали, как в зеркале.</p>

<h3>Логика щипка</h3>
<p>Расстояние между кончиками большого (4) и указательного (8) пальцев делится на
расстояние от запястья (0) до основания среднего пальца (9). Так признак не зависит
от расстояния до камеры. Используется гистерезис: рисование начинается, когда
отношение меньше нижнего порога, и прекращается, когда оно больше верхнего.
Между порогами состояние сохраняется, поэтому линия не «мерцает».</p>

<h3>Сглаживание координат</h3>
<p>Экспоненциальное сглаживание: s = α·x + (1 − α)·s<sub>пред</sub>. Малое α
даёт устойчивый курсор, но с задержкой; α = 1 — без сглаживания. После потери руки
сглаживание сбрасывается, а линия не соединяет точки до и после потери.</p>

<h3>Практическое применение</h3>
<ul>
<li>Бесконтактное управление в медицине, лабораториях, на производстве;</li>
<li>Интерактивные презентации и обучение;</li>
<li>Доступные интерфейсы для людей, которым неудобна мышь.</li>
</ul>

<h3>Ограничения</h3>
<ul>
<li>Качество зависит от освещения, фона, перекрытия пальцев и положения руки;</li>
<li>Отслеживается одна рука;</li>
<li>Скорость обработки зависит от компьютера и камеры;</li>
<li>Время инференса — не полная задержка системы.</li>
</ul>
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AirCanvas — рисование жестами")
        self.resize(1360, 820)
        self.setMinimumSize(960, 600)

        self.canvas = StrokeCanvas(CANVAS_WIDTH, CANVAS_HEIGHT)
        self.interpreter = GestureInterpreter(CANVAS_WIDTH, CANVAS_HEIGHT)
        self.perf = PerformanceMonitor()
        self.trials = TrialRecorder(CANVAS_WIDTH, CANVAS_HEIGHT)
        self.worker: CameraWorker | None = None
        self._stalled_workers: list[CameraWorker] = []
        self.camera_running = False
        self.color: Color = hex_to_rgb(PALETTE[0][1])
        self.brush_width = 8

        self._build_ui()
        self.setStyleSheet(STYLE)
        self._apply_brush_to_cursor()
        self._set_input_mode_mouse()
        self._update_buttons()
        self._update_model_hint()

        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(250)
        self._stats_timer.timeout.connect(self._refresh_stats)
        self._stats_timer.start()

        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self.undo)
        QShortcut(QKeySequence.StandardKey.Save, self, activated=self.save_png)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget(objectName="central")
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(10)
        root.addWidget(self._build_header())
        root.addWidget(self._build_toolbar())

        body = QHBoxLayout()
        body.setSpacing(10)
        self.canvas_view = CanvasView(self.canvas)
        self.canvas_view.mouse_pressed.connect(self._on_mouse_pressed)
        self.canvas_view.mouse_moved.connect(self._on_mouse_moved)
        self.canvas_view.mouse_released.connect(self._on_mouse_released)
        canvas_frame = QFrame(objectName="header")
        cf = QVBoxLayout(canvas_frame)
        cf.setContentsMargins(8, 8, 8, 8)
        cf.addWidget(self.canvas_view)
        body.addWidget(canvas_frame, 1)
        side_scroll = QScrollArea()
        side_scroll.setWidget(self._build_side_panel())
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setFixedWidth(356)
        side_scroll.setStyleSheet(f"QScrollArea {{ background: {BG}; }}")
        body.addWidget(side_scroll)
        root.addLayout(body, 1)
        self.setCentralWidget(central)
        self._build_status_bar()

    def _build_header(self) -> QWidget:
        header = QFrame(objectName="header")
        lay = QHBoxLayout(header)
        lay.setContentsMargins(14, 8, 14, 8)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(QLabel("AirCanvas", objectName="title"))
        titles.addWidget(QLabel("Рисование жестами перед камерой", objectName="subtitle"))
        lay.addLayout(titles)
        lay.addStretch(1)
        self.diag_button = QPushButton("Диагностика")
        self.diag_button.setCheckable(True)
        self.diag_button.toggled.connect(self._toggle_diagnostics)
        about = QPushButton("О проекте")
        about.clicked.connect(lambda: AboutDialog(self).exec())
        lay.addWidget(self.diag_button)
        lay.addWidget(about)
        return header

    def _build_toolbar(self) -> QWidget:
        bar = QFrame(objectName="toolbar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(8)

        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(140)
        self.camera_combo.setMaximumWidth(220)
        self.camera_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.camera_combo.setMinimumContentsLength(12)
        self.camera_combo.setToolTip("Выбор веб-камеры")
        for index, name in list_cameras():
            self.camera_combo.addItem(name, index)
        self.start_button = QPushButton("Включить камеру", objectName="primary")
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button = QPushButton("Остановить")
        self.stop_button.clicked.connect(self.stop_camera)
        lay.addWidget(self.camera_combo)
        lay.addWidget(self.start_button)
        lay.addWidget(self.stop_button)
        lay.addWidget(self._separator())

        lay.addWidget(role_label("Цвет", "secondary"))
        self.color_group = QButtonGroup(self)
        self.color_group.setExclusive(True)
        for i, (name, hex_value) in enumerate(PALETTE):
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setProperty("swatch", True)
            btn.setToolTip(name)
            btn.setAccessibleName(f"Цвет: {name}")
            btn.setStyleSheet(f"background: {hex_value};")
            btn.setChecked(i == 0)
            btn.clicked.connect(lambda _=False, h=hex_value: self.set_color(h))
            self.color_group.addButton(btn, i)
            lay.addWidget(btn)
        lay.addWidget(self._separator())

        lay.addWidget(role_label("Кисть", "secondary"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(2, 40)
        self.width_spin.setValue(self.brush_width)
        self.width_spin.setSuffix(" px")
        self.width_spin.setToolTip("Толщина кисти в пикселях холста 1280×720")
        self.width_spin.valueChanged.connect(self.set_brush_width)
        lay.addWidget(self.width_spin)
        lay.addWidget(self._separator())

        self.undo_button = QPushButton("Отменить")
        self.undo_button.setToolTip("Отменить последний штрих (Ctrl+Z)")
        self.undo_button.clicked.connect(self.undo)
        self.clear_button = QPushButton("Очистить")
        self.clear_button.clicked.connect(self.clear_canvas)
        self.save_button = QPushButton("Сохранить PNG")
        self.save_button.setToolTip("Сохранить только рисунок (Ctrl+S)")
        self.save_button.clicked.connect(self.save_png)
        lay.addWidget(self.undo_button)
        lay.addWidget(self.clear_button)
        lay.addWidget(self.save_button)
        lay.addStretch(1)
        self.mode_badge = QLabel(objectName="modeBadge")
        lay.addWidget(self.mode_badge)
        return bar

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setStyleSheet(f"color: {BORDER};")
        return line

    def _build_side_panel(self) -> QWidget:
        panel = QFrame(objectName="sidePanel")
        panel.setFixedWidth(340)
        panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        lay.addWidget(role_label("Камера (зеркальный вид)", "secondary"))
        self.preview = QLabel("Камера выключена", objectName="preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setFixedSize(320, 240)
        lay.addWidget(self.preview)

        instr = QLabel(INSTRUCTION)
        instr.setWordWrap(True)
        lay.addWidget(instr)

        self.model_hint = role_label("", "warning")
        self.model_hint.setWordWrap(True)
        lay.addWidget(self.model_hint)

        settings = QGroupBox("Настройки отслеживания")
        grid = QGridLayout(settings)
        grid.setVerticalSpacing(6)

        self.res_combo = QComboBox()
        for w, h in RESOLUTIONS:
            self.res_combo.addItem(f"{w} × {h}", (w, h))
        self.res_combo.setToolTip("Запрашиваемое разрешение захвата; применяется при следующем запуске камеры")
        grid.addWidget(QLabel("Разрешение"), 0, 0)
        grid.addWidget(self.res_combo, 0, 1)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.05, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.setValue(DEFAULT_SMOOTHING_ALPHA)
        self.alpha_spin.setToolTip(
            "Коэффициент α экспоненциального сглаживания.\n"
            "Меньше — курсор стабильнее, но запаздывает; 1.0 — без сглаживания."
        )
        self.alpha_spin.valueChanged.connect(self._on_alpha_changed)
        grid.addWidget(QLabel("Сглаживание α"), 1, 0)
        grid.addWidget(self.alpha_spin, 1, 1)
        self.alpha_hint = role_label("", "secondary")
        self.alpha_hint.setWordWrap(True)
        grid.addWidget(self.alpha_hint, 2, 0, 1, 2)

        self.press_spin = QDoubleSpinBox()
        self.release_spin = QDoubleSpinBox()
        for spin, value in ((self.press_spin, DEFAULT_PRESS_THRESHOLD),
                            (self.release_spin, DEFAULT_RELEASE_THRESHOLD)):
            spin.setRange(0.05, 1.50)
            spin.setSingleStep(0.01)
            spin.setDecimals(2)
            spin.setValue(value)
            spin.valueChanged.connect(self._on_thresholds_changed)
        self.press_spin.setToolTip("Рисование начинается, когда отношение щипка меньше этого порога")
        self.release_spin.setToolTip("Рисование прекращается, когда отношение щипка больше этого порога")
        grid.addWidget(QLabel("Порог сведения"), 3, 0)
        grid.addWidget(self.press_spin, 3, 1)
        grid.addWidget(QLabel("Порог разведения"), 4, 0)
        grid.addWidget(self.release_spin, 4, 1)
        self.threshold_error = role_label("", "error")
        self.threshold_error.setWordWrap(True)
        self.threshold_error.hide()
        grid.addWidget(self.threshold_error, 5, 0, 1, 2)
        self.pinch_check = QCheckBox("Показывать отношение щипка")
        self.pinch_check.setChecked(True)
        grid.addWidget(self.pinch_check, 6, 0, 1, 2)
        self.pinch_label = role_label("Отношение щипка: —", "secondary")
        grid.addWidget(self.pinch_label, 7, 0, 1, 2)
        lay.addWidget(settings)
        self._on_alpha_changed(self.alpha_spin.value())

        self.diag_box = QGroupBox("Диагностика")
        dg = QGridLayout(self.diag_box)
        self.diag_fps = QLabel("—")
        self.diag_inf = QLabel("—")
        self.diag_det = QLabel("—")
        self.diag_trials = QLabel("Замеров: 0")
        dg.addWidget(QLabel("Обработано кадров/с"), 0, 0)
        dg.addWidget(self.diag_fps, 0, 1)
        dg.addWidget(QLabel("Среднее время модели"), 1, 0)
        dg.addWidget(self.diag_inf, 1, 1)
        dg.addWidget(QLabel("Кадры с рукой"), 2, 0)
        dg.addWidget(self.diag_det, 2, 1)
        note = role_label("Время модели — только вызов нейросети, а не полная задержка системы.", "secondary")
        note.setWordWrap(True)
        dg.addWidget(note, 3, 0, 1, 2)
        self.trial_button = QPushButton("Начать замер")
        self.trial_button.setToolTip("Держите руку неподвижно 10–20 секунд, затем завершите замер")
        self.trial_button.clicked.connect(self._toggle_trial)
        self.export_button = QPushButton("Экспорт CSV")
        self.export_button.clicked.connect(self.export_diagnostics)
        dg.addWidget(self.trial_button, 4, 0)
        dg.addWidget(self.export_button, 4, 1)
        dg.addWidget(self.diag_trials, 5, 0, 1, 2)
        self.diag_box.hide()
        lay.addWidget(self.diag_box)

        lay.addStretch(1)
        return panel

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        self.st_camera = QLabel()
        self.st_hand = QLabel()
        self.st_draw = QLabel()
        self.st_fps = QLabel()
        for w in (self.st_camera, self.st_hand, self.st_draw, self.st_fps):
            bar.addWidget(w)
        self.setStatusBar(bar)
        self._set_camera_status("выключена")
        self._set_hand_status("—")
        self._set_draw_status(False)
        self.st_fps.setText("FPS обработки: —")

    # --------------------------------------------------------------- status
    def _set_camera_status(self, text: str) -> None:
        self.st_camera.setText(f"Камера: {text}")

    def _set_hand_status(self, text: str) -> None:
        self.st_hand.setText(f"Рука: {text}")

    def _set_draw_status(self, drawing: bool) -> None:
        self.st_draw.setText("Рисование: идёт" if drawing else "Рисование: нет")

    def _set_input_mode_gesture(self) -> None:
        self.canvas_view.cancel_mouse()
        self.canvas_view.mouse_enabled = False
        self.canvas_view.setCursor(Qt.CursorShape.ArrowCursor)
        self.canvas_view.set_cursor_overlay(None, False)
        self.mode_badge.setText("Режим ввода: Жесты")

    def _set_input_mode_mouse(self) -> None:
        self.canvas_view.mouse_enabled = True
        self.canvas_view.setCursor(Qt.CursorShape.CrossCursor)
        self.canvas_view.set_cursor_overlay(None, False)
        self.mode_badge.setText("Режим ввода: Мышь")

    def _update_buttons(self) -> None:
        busy = self.worker is not None
        self.start_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        self.camera_combo.setEnabled(not busy)
        self.res_combo.setEnabled(not busy)
        self.undo_button.setEnabled(self.canvas.can_undo)
        self.trial_button.setEnabled(self.camera_running or self.trials.active)

    def _update_model_hint(self) -> None:
        if model_available():
            self.model_hint.hide()
        else:
            self.model_hint.setText(
                "Модель не найдена. Выполните «python download_model.py», "
                "а пока можно рисовать мышью."
            )
            self.model_hint.show()

    def _refresh_stats(self) -> None:
        if not self.camera_running:
            self.st_fps.setText("FPS обработки: —")
            return
        snap = self.perf.snapshot()
        self.st_fps.setText(f"FPS обработки: {snap.fps:.1f}")
        if self.diag_box.isVisible():
            self.diag_fps.setText(f"{snap.fps:.1f}")
            self.diag_inf.setText(f"{snap.avg_inference_ms:.1f} мс")
            self.diag_det.setText(f"{snap.detection_percent:.0f} %")

    def _apply_brush_to_cursor(self) -> None:
        self.canvas_view.cursor_color = QColor(*self.color)
        self.canvas_view.cursor_radius = self.brush_width / 2
        self.canvas_view.update()

    # --------------------------------------------------------------- camera
    def start_camera(self) -> None:
        if self.worker is not None:
            return
        if not model_available():
            QMessageBox.warning(self, "Модель не найдена", SETUP_HINT.format(path=MODEL_PATH))
            self._update_model_hint()
            return
        index = self.camera_combo.currentData()
        width, height = self.res_combo.currentData()
        self.canvas_view.cancel_mouse()
        self.canvas.end_stroke()
        self.interpreter.reset()
        self.perf.reset_session()

        worker = CameraWorker(int(index), width, height, preview_width=320, parent=self)
        worker.camera_started.connect(self._on_camera_started)
        worker.frame_processed.connect(self._on_frame)
        worker.error_occurred.connect(self._on_camera_error)
        worker.finished.connect(self._on_worker_finished)
        self.worker = worker
        self._set_camera_status("запуск…")
        self.preview.setText("Подключение к камере…")
        self._set_input_mode_gesture()
        self._update_buttons()
        worker.start()

    def stop_camera(self) -> None:
        worker = self.worker
        if worker is None:
            return
        worker.stop()
        if not worker.wait(5000):
            # Keep a reference: destroying a running QThread would crash.
            self._stalled_workers.append(worker)
            QMessageBox.warning(self, "Камера", "Поток камеры не остановился вовремя.")
        self._on_worker_finished(worker)

    def _on_camera_started(self, width: int, height: int) -> None:
        if self.sender() is not self.worker:
            return
        self.camera_running = True
        self._set_camera_status(f"включена ({width}×{height})")
        self._set_hand_status("Рука не обнаружена")
        self._update_buttons()

    def _on_camera_error(self, message: str) -> None:
        if self.sender() is not self.worker:
            return
        self._set_camera_status("ошибка")
        QMessageBox.warning(self, "Камера", message)

    def _on_worker_finished(self, worker: CameraWorker | None = None) -> None:
        worker = worker if isinstance(worker, CameraWorker) else self.sender()
        if worker is not self.worker or worker is None:
            return
        self.worker = None
        self.camera_running = False
        if self.trials.active:
            self._finish_trial()
        self.canvas.end_stroke()
        self.interpreter.reset()
        if worker not in self._stalled_workers:
            worker.deleteLater()
        self.preview.clear()
        self.preview.setText("Камера выключена")
        if not self.st_camera.text().endswith("ошибка"):
            self._set_camera_status("выключена")
        self._set_hand_status("—")
        self._set_draw_status(False)
        self.pinch_label.setText("Отношение щипка: —")
        self._set_input_mode_mouse()
        self._update_buttons()
        self.canvas_view.update()

    def _on_frame(self, result: FrameResult) -> None:
        worker = self.worker
        if self.sender() is not worker or worker is None:
            return
        try:
            self.preview.setPixmap(
                QPixmap.fromImage(result.preview).scaled(
                    self.preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            gf = self.interpreter.update(result.observation)
            self.perf.record(result.timestamp_s, result.inference_ms, gf.hand_detected)
            self.trials.record(result.timestamp_s, result.inference_ms, gf.raw_cursor,
                               gf.cursor if gf.hand_detected else None)

            if gf.stroke_break and self.canvas.is_drawing:
                self.canvas.end_stroke()
            if gf.drawing and gf.cursor is not None:
                if self.canvas.is_drawing:
                    self.canvas.add_point(gf.cursor)
                else:
                    self.canvas.begin_stroke(gf.cursor, self.color, self.brush_width)

            if gf.hand_detected:
                self._set_hand_status("обнаружена")
                self.canvas_view.set_cursor_overlay(QPointF(*gf.cursor), gf.drawing)
            else:
                self._set_hand_status("Рука не обнаружена")
                self.canvas_view.set_cursor_overlay(None, False)
            self._set_draw_status(gf.drawing)
            if self.pinch_check.isChecked() and gf.pinch_ratio is not None:
                self.pinch_label.setText(f"Отношение щипка: {gf.pinch_ratio:.2f}")
            elif not gf.hand_detected:
                self.pinch_label.setText("Отношение щипка: —")
            self.undo_button.setEnabled(self.canvas.can_undo)
        finally:
            worker.acknowledge()

    # ---------------------------------------------------------------- mouse
    def _on_mouse_pressed(self, pos: QPointF) -> None:
        if self.worker is not None:
            return
        self.canvas.begin_stroke((pos.x(), pos.y()), self.color, self.brush_width)
        self._set_draw_status(True)
        self.canvas_view.update()

    def _on_mouse_moved(self, pos: QPointF) -> None:
        if self.worker is not None or not self.canvas.is_drawing:
            return
        if self.canvas.add_point((pos.x(), pos.y())):
            self.canvas_view.update()

    def _on_mouse_released(self) -> None:
        if self.worker is not None:
            return
        self.canvas.end_stroke()
        self._set_draw_status(False)
        self._update_buttons()
        self.canvas_view.update()

    # ---------------------------------------------------------------- brush
    def set_color(self, hex_value: str) -> None:
        self.color = hex_to_rgb(hex_value)
        self.canvas.set_brush(self.color, self.brush_width)
        self._apply_brush_to_cursor()

    def set_brush_width(self, value: int) -> None:
        self.brush_width = int(value)
        self.canvas.set_brush(self.color, self.brush_width)
        self._apply_brush_to_cursor()

    def _on_alpha_changed(self, value: float) -> None:
        self.interpreter.smoother.alpha = float(value)
        tau = smoothing_time_constant(float(value))
        if tau <= 0:
            self.alpha_hint.setText("Без сглаживания: максимальная отзывчивость, больше дрожания.")
        else:
            self.alpha_hint.setText(
                f"Постоянная времени ≈ {tau:.1f} кадра. Меньше α — стабильнее, но с задержкой."
            )

    def _on_thresholds_changed(self) -> None:
        press, release = self.press_spin.value(), self.release_spin.value()
        try:
            self.interpreter.pinch.set_thresholds(press, release)
        except ValueError as exc:
            self.threshold_error.setText(f"{exc}. Действуют прежние значения.")
            self.threshold_error.show()
            return
        self.threshold_error.hide()

    # --------------------------------------------------------------- actions
    def undo(self) -> None:
        if self.canvas.undo():
            self.canvas_view.update()
        self._update_buttons()

    def clear_canvas(self) -> None:
        if not self.canvas.can_undo:
            return
        answer = QMessageBox.question(
            self, "Очистить холст", "Удалить весь рисунок? Это действие нельзя отменить.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.canvas.clear()
            self.canvas_view.update()
            self._update_buttons()

    def _default_dir(self) -> Path:
        pictures = Path.home() / "Pictures"
        return pictures if pictures.is_dir() else Path.home()

    def save_png(self) -> None:
        name = datetime.now().strftime("aircanvas_%Y%m%d_%H%M%S.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить рисунок", str(self._default_dir() / name), "Изображение PNG (*.png)"
        )
        if not path:
            self.statusBar().showMessage("Сохранение отменено", 3000)
            return
        try:
            saved = self.canvas.save_png(path)
        except OSError as exc:
            QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось сохранить файл:\n{exc}")
            return
        self.statusBar().showMessage(f"Сохранено: {saved}", 5000)

    # ----------------------------------------------------------- diagnostics
    def _toggle_diagnostics(self, visible: bool) -> None:
        self.diag_box.setVisible(visible)

    def _toggle_trial(self) -> None:
        if self.trials.active:
            self._finish_trial()
        else:
            self.trials.start(self.alpha_spin.value())
            self.trial_button.setText("Завершить замер")
            self.alpha_spin.setEnabled(False)

    def _finish_trial(self) -> None:
        result = self.trials.stop()
        self.trial_button.setText("Начать замер")
        self.alpha_spin.setEnabled(True)
        self.diag_trials.setText(f"Замеров: {len(self.trials.results)}")
        if result is not None:
            self.statusBar().showMessage(
                f"Замер {result.trial}: α={result.alpha:.2f}, кадров {result.frames}, "
                f"σx/W={result.smoothed_std_x_norm:.4f}, σy/H={result.smoothed_std_y_norm:.4f}",
                8000,
            )
        self._update_buttons()

    def export_diagnostics(self) -> None:
        name = datetime.now().strftime("aircanvas_diagnostics_%Y%m%d_%H%M%S.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт диагностики", str(self._default_dir() / name), "CSV (*.csv)"
        )
        if not path:
            return
        settings = {
            "canvas": f"{CANVAS_WIDTH}x{CANVAS_HEIGHT}",
            "capture_requested": "{}x{}".format(*self.res_combo.currentData()),
            "alpha": self.alpha_spin.value(),
            "press_threshold": self.press_spin.value(),
            "release_threshold": self.release_spin.value(),
        }
        try:
            saved = export_csv(path, self.perf.session_summary(), self.trials.results, settings)
        except OSError as exc:
            QMessageBox.critical(self, "Ошибка экспорта", f"Не удалось сохранить CSV:\n{exc}")
            return
        self.statusBar().showMessage(f"Диагностика сохранена: {saved}", 5000)

    # ---------------------------------------------------------------- close
    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_camera()
        super().closeEvent(event)

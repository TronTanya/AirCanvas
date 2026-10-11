"""AirCanvas entry point."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "Не найден PySide6. Активируйте виртуальное окружение и выполните:\n"
            "    pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("AirCanvas")
    app.setStyle("Fusion")
    font = QFont("Segoe UI" if sys.platform == "win32" else app.font().family())
    font.setPointSize(10)
    app.setFont(font)

    try:
        from window import MainWindow
    except ImportError as exc:
        print(
            f"Не удалось загрузить модули приложения: {exc}\n"
            "Проверьте установку зависимостей: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

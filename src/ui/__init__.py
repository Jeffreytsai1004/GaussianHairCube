"""
UI module for the application interface.
"""

# Lazy imports to avoid circular dependencies during PyInstaller bundling
__all__ = [
    'MainWindow',
    'InputPanel',
    'OutputPanel',
    'ViewerWidget'
]

def __getattr__(name):
    if name == 'MainWindow':
        from src.ui.main_window import MainWindow
        return MainWindow
    elif name == 'InputPanel':
        from src.ui.input_panel import InputPanel
        return InputPanel
    elif name == 'OutputPanel':
        from src.ui.output_panel import OutputPanel
        return OutputPanel
    elif name == 'ViewerWidget':
        from src.ui.viewer_widget import ViewerWidget
        return ViewerWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
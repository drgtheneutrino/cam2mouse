"""
System Tray Integration for AirCursor
======================================
Provides system tray functionality for background operation.
"""

from PyQt5 import QtWidgets, QtGui, QtCore


class SystemTray(QtWidgets.QSystemTrayIcon):
    """System tray icon with context menu"""
    
    def __init__(self, parent=None, icon_path=None):
        super().__init__(parent)
        self.parent = parent
        
        if icon_path:
            self.setIcon(QtGui.QIcon(icon_path))
        
        # Create context menu
        self.menu = QtWidgets.QMenu()
        self.menu.setStyleSheet("""
            QMenu {
                background-color: #21262D;
                border: 1px solid #30363D;
                border-radius: 8px;
                padding: 5px;
            }
            QMenu::item {
                padding: 8px 20px;
                color: #E6EDF3;
            }
            QMenu::item:selected {
                background-color: #1F6FEB;
                border-radius: 4px;
            }
            QMenu::separator {
                height: 1px;
                background: #30363D;
                margin: 5px 10px;
            }
        """)
        
        # Show/Hide action
        self.show_action = QtWidgets.QAction("👁️ Show Window")
        self.show_action.triggered.connect(self.restore_app)
        self.menu.addAction(self.show_action)
        
        # Separator
        self.menu.addSeparator()
        
        # Start/Stop tracking
        self.tracking_action = QtWidgets.QAction("▶️ Start Tracking")
        self.tracking_action.triggered.connect(self.toggle_tracking)
        self.menu.addAction(self.tracking_action)
        
        # Separator
        self.menu.addSeparator()
        
        # About action
        self.about_action = QtWidgets.QAction("ℹ️ About")
        self.about_action.triggered.connect(self.show_about)
        self.menu.addAction(self.about_action)
        
        # Exit action
        self.exit_action = QtWidgets.QAction("❌ Exit")
        self.exit_action.triggered.connect(self.exit_app)
        self.menu.addAction(self.exit_action)
        
        self.setContextMenu(self.menu)
        self.activated.connect(self.on_tray_activated)
        
        # Tooltip
        self.setToolTip("AirCursor 2.0 - Gesture Control")
        
    def on_tray_activated(self, reason):
        """Handle tray icon activation"""
        if reason == self.Trigger:  # Single click
            self.restore_app()
        elif reason == self.DoubleClick:
            self.restore_app()
            
    def restore_app(self):
        """Restore the main window"""
        if self.parent:
            self.parent.show()
            self.parent.activateWindow()
            self.parent.raise_()
            
    def toggle_tracking(self):
        """Toggle hand tracking"""
        if self.parent:
            if hasattr(self.parent, 'tracking_active'):
                if self.parent.tracking_active:
                    self.parent.stop_tracking()
                    self.tracking_action.setText("▶️ Start Tracking")
                else:
                    self.parent.start_tracking()
                    self.tracking_action.setText("⏹️ Stop Tracking")
                    
    def update_tracking_status(self, is_active: bool):
        """Update the tracking menu item"""
        if is_active:
            self.tracking_action.setText("⏹️ Stop Tracking")
        else:
            self.tracking_action.setText("▶️ Start Tracking")
            
    def show_about(self):
        """Show about dialog"""
        if self.parent and hasattr(self.parent, 'show_about_dialog'):
            self.parent.show_about_dialog()
            
    def exit_app(self):
        """Exit the application"""
        if self.parent:
            self.parent.close()

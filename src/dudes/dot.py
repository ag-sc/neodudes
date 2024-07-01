import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

if Gtk.init_check()[0] is True:
    from xdot import DotWindow  # type: ignore


class DotFile:
    def __init__(self, dotStr: str):
        self._dotStr = dotStr

    @staticmethod
    def runXDot(dotStr, run=True, title=None):
        width, height = 610, 610

        win = DotWindow(width=width, height=height)
        win.connect('delete-event', Gtk.main_quit)
        # print(dotStr)
        win.set_dotcode(dotStr.encode("utf-8"))
        if title is not None:
            win.set_title(title)
        # Reset KeyboardInterrupt SIGINT handler, so that glib loop can be stopped by it
        import signal
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        if run:
            Gtk.main()

    def view(self, run=True, title=None):
        DotFile.runXDot(self._dotStr, run=run, title=title)

    def __str__(self):
        return self._dotStr

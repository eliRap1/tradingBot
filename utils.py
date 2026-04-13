import logging
import os
import yaml
from datetime import datetime
from pathlib import Path


# ANSI color codes for terminal output
class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    # Colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    # Backgrounds
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


class ColorFormatter(logging.Formatter):
    """Colorized console formatter with visual hierarchy."""

    LEVEL_COLORS = {
        logging.DEBUG:    (_C.GRAY,    "DBG"),
        logging.INFO:     (_C.CYAN,    "INF"),
        logging.WARNING:  (_C.YELLOW,  "WRN"),
        logging.ERROR:    (_C.RED,     "ERR"),
        logging.CRITICAL: (_C.BG_RED + _C.WHITE, "CRT"),
    }

    # Module icons for quick visual scanning
    MODULE_ICONS = {
        "coordinator": ">>",
        "watcher":     "@@",
        "portfolio":   "$$",
        "risk":        "!!",
        "data":        "<>",
        "regime":      "~~",
        "broker":      "[]",
        "strategy":    "**",
        "alerts":      ">>",
        "live_trading": "##",
        "optimizer":   "%%",
    }

    def format(self, record):
        color, tag = self.LEVEL_COLORS.get(record.levelno, (_C.WHITE, "???"))

        # Find module icon
        icon = ".."
        name = record.name
        for key, ic in self.MODULE_ICONS.items():
            if key in name:
                icon = ic
                break

        # Shorten module name for display
        short_name = name.split(".")[-1] if "." in name else name
        if len(short_name) > 12:
            short_name = short_name[:12]

        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        msg = record.getMessage()

        # Highlight key trading events in the message
        msg = self._highlight(msg, record.levelno)

        line = (
            f"{_C.DIM}{timestamp}{_C.RESET} "
            f"{color}{_C.BOLD}{tag}{_C.RESET} "
            f"{_C.GRAY}{icon}{_C.RESET} "
            f"{_C.BLUE}{short_name:<12}{_C.RESET} "
            f"{msg}"
        )
        return line

    def _highlight(self, msg: str, level: int) -> str:
        """Highlight key trading terms in messages."""
        if level >= logging.ERROR:
            return f"{_C.RED}{msg}{_C.RESET}"
        if level >= logging.WARNING:
            return f"{_C.YELLOW}{msg}{_C.RESET}"

        # Highlight specific trading keywords
        highlights = {
            "CONFIRMED": _C.GREEN + _C.BOLD,
            "ORDER:": _C.GREEN + _C.BOLD,
            "TRAILING STOP": _C.YELLOW + _C.BOLD,
            "BREAKEVEN": _C.YELLOW + _C.BOLD,
            "PARTIAL EXIT": _C.MAGENTA + _C.BOLD,
            "TIME STOP": _C.YELLOW + _C.BOLD,
            "DRAWDOWN": _C.RED + _C.BOLD,
            "DAILY LOSS LIMIT": _C.RED + _C.BOLD,
            "RATE LIMIT": _C.RED + _C.BOLD,
            "BULL": _C.GREEN,
            "BEAR": _C.RED,
            "CHOP": _C.YELLOW,
            "SIGNAL": _C.CYAN + _C.BOLD,
            "BUY": _C.GREEN + _C.BOLD,
            "SELL": _C.RED + _C.BOLD,
            "LONG": _C.GREEN,
            "SHORT": _C.RED,
            "P&L=": _C.BOLD,
        }
        for keyword, color in highlights.items():
            if keyword in msg:
                msg = msg.replace(keyword, f"{color}{keyword}{_C.RESET}")
        return msg


def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Console handler — colorized, UTF-8 safe on Windows
        import sys, io
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") \
            if hasattr(sys.stdout, "buffer") else sys.stdout
        ch = logging.StreamHandler(stream)
        ch.setLevel(logging.INFO)
        ch.setFormatter(ColorFormatter())
        logger.addHandler(ch)

        # File handler — plain text (no colors)
        log_file = os.path.join(log_dir, f"bot_{datetime.now():%Y%m%d}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logger.addHandler(fh)

    return logger


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

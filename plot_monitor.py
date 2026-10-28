"""Display a live training graph while a JSONL log is being written."""

import argparse
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

from helper import plot_training_history
from plot_training import load_records


class TrainingGraphMonitor:
    def __init__(self, log_path, output_path, interval_ms):
        self.log_path = Path(log_path)
        self.output_path = Path(output_path)
        self.interval_ms = interval_ms
        self.root = tk.Tk()
        self.root.title('Snake With Brain - Live Training Analysis')
        self.root.geometry('1280x900')
        self.image_label = tk.Label(self.root, text='Waiting for episode data...')
        self.image_label.pack(expand=True, fill='both')
        self.root.after(0, self.refresh)

    def refresh(self):
        try:
            if self.log_path.exists():
                records = load_records(self.log_path)
                if plot_training_history(records, self.output_path) and self.output_path.exists():
                    with Image.open(self.output_path) as image:
                        image_copy = image.copy()
                    image_copy.thumbnail((1240, 840), Image.Resampling.LANCZOS)
                    self.photo = ImageTk.PhotoImage(image_copy)
                    self.image_label.configure(image=self.photo, text='')
                    episodes = sum(record.get('event') == 'episode' for record in records)
                    self.root.title(f'Snake With Brain - Live Training Analysis ({episodes} games)')
        except (OSError, ValueError, tk.TclError):
            # The trainer may be writing the log or replacing the PNG at this moment.
            pass
        self.root.after(self.interval_ms, self.refresh)

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description='Display a live graph from a JSONL training log.')
    parser.add_argument('--log-file', default='runs/shared/shared-training.jsonl')
    parser.add_argument('--output', default='runs/shared/shared-training.png')
    parser.add_argument('--interval-ms', type=int, default=2000)
    args = parser.parse_args()
    TrainingGraphMonitor(args.log_file, args.output, args.interval_ms).run()


if __name__ == '__main__':
    main()

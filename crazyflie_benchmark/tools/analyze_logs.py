import argparse
import os
import sys
from typing import List, Tuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import tkinter as tk
from tkinter import ttk

def analyze_log(log_file, save_plots=False, output_dir=None):
    if not os.path.exists(log_file):
        print(f"Error: File {log_file} not found")
        return

    print(f"Analyzing {log_file}...")

    # Load data
    df = pd.read_csv(log_file)

    # Sort by timestamp just in case
    df = df.sort_values('timestamp')

    # Keep raw data; no forward fill. Only fill state to aid segmentation.
    if 'state' in df.columns:
        df['state'] = df['state'].ffill().bfill()

    telem_df = df[df["type"] == "telemetry"].copy()
    cmd_df = df[df["type"] == "command"].copy()

    # Create output directory if saving
    if save_plots:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            # Create a folder named after the log file in the same directory
            base_name = os.path.splitext(os.path.basename(log_file))[0]
            output_dir = os.path.join(os.path.dirname(log_file), base_name)
            os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(log_file))[0]

    # Helpers
    def get_data(col, source="telem"):
        if source == "telem":
            return telem_df[col].to_numpy()
        return df[col].to_numpy()

    def state_segments() -> List[Tuple[float, float, str]]:
        if 'state' not in telem_df.columns or telem_df.empty:
            return []
        states = telem_df['state'].to_numpy()
        ts = telem_df['timestamp'].to_numpy()
        segments = []
        start_idx = 0
        for i in range(1, len(states)):
            if states[i] != states[start_idx]:
                segments.append((ts[start_idx], ts[i-1], states[start_idx]))
                start_idx = i
        # tail
        segments.append((ts[start_idx], ts[-1], states[start_idx]))
        return segments

    STATE_COLORS = [
        "#f94144", "#f3722c", "#f8961e", "#f9844a", "#f9c74f",
        "#90be6d", "#43aa8b", "#4d908e", "#577590", "#277da1"
    ]

    def shade_states(ax):
        segments = state_segments()
        if not segments:
            return
        ymin, ymax = ax.get_ylim()
        for idx, (t0, t1, state) in enumerate(segments):
            color = STATE_COLORS[idx % len(STATE_COLORS)]
            ax.axvspan(t0, t1, color=color, alpha=0.12, linewidth=0)
            ax.text((t0 + t1) / 2, ymax, state, ha='center', va='bottom',
                    fontsize=8, color=color, rotation=0, alpha=0.9)

    if save_plots:
        # Plot 1: Attitude (Roll/Pitch)
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(get_data('timestamp'), get_data('stabilizer.roll'), label='Roll (Actual)', linestyle='-')
        if 'cmd_roll' in cmd_df.columns and not cmd_df.empty:
            plt.scatter(cmd_df['timestamp'].to_numpy(), cmd_df['cmd_roll'].to_numpy(),
                        color='r', s=12, label='Roll (Cmd)', alpha=0.8)
        plt.ylabel('Roll (deg)')
        plt.legend()
        plt.grid(True)
        plt.title(f'Attitude Response - {base_name}')
        shade_states(plt.gca())

        plt.subplot(2, 1, 2)
        plt.plot(get_data('timestamp'), get_data('stabilizer.pitch'), label='Pitch (Actual)', linestyle='-')
        if 'cmd_pitch' in cmd_df.columns and not cmd_df.empty:
            plt.scatter(cmd_df['timestamp'].to_numpy(), cmd_df['cmd_pitch'].to_numpy(),
                        color='r', s=12, label='Pitch (Cmd)', alpha=0.8)
        plt.ylabel('Pitch (deg)')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.grid(True)
        shade_states(plt.gca())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_attitude.png"))
        print(f"Saved {base_name}_attitude.png")
        plt.close()

        # Plot 2: Position & Thrust
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        if 'position.z' in df.columns:
            plt.plot(get_data('timestamp'), get_data('position.z'), label='Z (Height)')
        plt.ylabel('Height (m)')
        plt.legend()
        plt.grid(True)
        plt.title(f'Position & Thrust - {base_name}')
        shade_states(plt.gca())

        plt.subplot(2, 1, 2)
        if 'cmd_thrust' in cmd_df.columns and not cmd_df.empty:
            plt.scatter(cmd_df['timestamp'].to_numpy(), cmd_df['cmd_thrust'].to_numpy(),
                        label='Thrust (Cmd)', color='g', s=12, alpha=0.8)
        plt.ylabel('Thrust')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.grid(True)
        shade_states(plt.gca())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_thrust.png"))
        print(f"Saved {base_name}_thrust.png")
        plt.close()

        # Plot 3: Trajectory (X-Y)
        if 'position.x' in df.columns and 'position.y' in df.columns:
            plt.figure(figsize=(8, 8))
            plt.plot(get_data('position.x'), get_data('position.y'), label='Trajectory')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title(f'XY Trajectory - {base_name}')
            plt.grid(True)
            plt.axis('equal')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"{base_name}_trajectory.png"))
            print(f"Saved {base_name}_trajectory.png")
            plt.close()

    else:
        # Interactive Mode with tabs (TkAgg). Built-in toolbar supports pan/zoom.
        class LogViewerTabs:
            def __init__(self, df, base_name):
                self.df = df
                self.base_name = base_name

                self.root = tk.Tk()
                self.root.title(f'Log Analysis - {base_name}')
                self.root.geometry("1200x850")

                notebook = ttk.Notebook(self.root)
                notebook.pack(fill=tk.BOTH, expand=True)

                self.tab_att = ttk.Frame(notebook)
                self.tab_pos = ttk.Frame(notebook)
                self.tab_traj = ttk.Frame(notebook)
                notebook.add(self.tab_att, text="Attitude")
                notebook.add(self.tab_pos, text="Position & Thrust")
                notebook.add(self.tab_traj, text="Trajectory")

                self.build_attitude_tab()
                self.build_position_tab()
                self.build_traj_tab()

                self.root.mainloop()

            def _add_toolbar(self, frame, fig):
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                toolbar = NavigationToolbar2Tk(canvas, frame)
                toolbar.update()
                return canvas

            def build_attitude_tab(self):
                fig = Figure(figsize=(9, 7), dpi=100)
                ax1 = fig.add_subplot(3, 1, 1)
                ax1.plot(get_data('timestamp'), get_data('stabilizer.roll'), label='Roll (Actual)', linestyle='-')
                if 'cmd_roll' in cmd_df.columns and not cmd_df.empty:
                    ax1.scatter(cmd_df['timestamp'].to_numpy(), cmd_df['cmd_roll'].to_numpy(),
                                color='r', s=12, label='Roll (Cmd)', alpha=0.8)
                ax1.set_ylabel('Roll (deg)')
                ax1.legend()
                ax1.grid(True)
                ax1.set_title(f'Attitude Response - {self.base_name}')
                shade_states(ax1)

                ax2 = fig.add_subplot(3, 1, 2)
                ax2.plot(get_data('timestamp'), get_data('stabilizer.pitch'), label='Pitch (Actual)', linestyle='-')
                if 'cmd_pitch' in cmd_df.columns and not cmd_df.empty:
                    ax2.scatter(cmd_df['timestamp'].to_numpy(), cmd_df['cmd_pitch'].to_numpy(),
                                color='r', s=12, label='Pitch (Cmd)', alpha=0.8)
                ax2.set_ylabel('Pitch (deg)')
                ax2.legend()
                ax2.grid(True)
                shade_states(ax2)

                ax3 = fig.add_subplot(3, 1, 3)
                # Prefer gyro z (deg/s) for yaw rate if available
                if 'gyro.z' in telem_df.columns:
                    ax3.plot(get_data('timestamp'), get_data('gyro.z'), label='Yaw Rate (Actual)', linestyle='-')
                elif 'stabilizer.yaw' in telem_df.columns:
                    ax3.plot(get_data('timestamp'), get_data('stabilizer.yaw'), label='Yaw (deg) (Actual)', linestyle='-')
                if 'cmd_yaw_rate' in cmd_df.columns and not cmd_df.empty:
                    ax3.scatter(cmd_df['timestamp'].to_numpy(), cmd_df['cmd_yaw_rate'].to_numpy(),
                                color='r', s=12, label='Yaw Rate (Cmd)', alpha=0.8)
                ax3.set_ylabel('Yaw Rate (deg/s)')
                ax3.set_xlabel('Time (s)')
                ax3.legend()
                ax3.grid(True)
                shade_states(ax3)

                fig.tight_layout()
                self._add_toolbar(self.tab_att, fig)

            def build_position_tab(self):
                fig = Figure(figsize=(8, 6), dpi=100)
                ax1 = fig.add_subplot(2, 1, 1)
                if 'position.z' in self.df.columns:
                    ax1.plot(get_data('timestamp'), get_data('position.z'), label='Z (Height)')
                ax1.set_ylabel('Height (m)')
                ax1.legend()
                ax1.grid(True)
                ax1.set_title(f'Position & Thrust - {self.base_name}')
                shade_states(ax1)

                ax2 = fig.add_subplot(2, 1, 2)
                if 'cmd_thrust' in cmd_df.columns and not cmd_df.empty:
                    ax2.scatter(cmd_df['timestamp'].to_numpy(), cmd_df['cmd_thrust'].to_numpy(),
                                label='Thrust (Cmd)', color='g', s=12, alpha=0.8)
                ax2.set_ylabel('Thrust')
                ax2.set_xlabel('Time (s)')
                ax2.legend()
                ax2.grid(True)
                shade_states(ax2)

                fig.tight_layout()
                self._add_toolbar(self.tab_pos, fig)

            def build_traj_tab(self):
                fig = Figure(figsize=(7, 6), dpi=100)
                ax = fig.add_subplot(1, 1, 1)
                if 'position.x' in self.df.columns and 'position.y' in self.df.columns:
                    ax.plot(get_data('position.x'), get_data('position.y'), label='Trajectory')
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                    ax.set_title(f'XY Trajectory - {self.base_name}')
                    ax.grid(True)
                    ax.axis('equal')
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, "No Position Data", ha='center', va='center')

                fig.tight_layout()
                self._add_toolbar(self.tab_traj, fig)

        LogViewerTabs(df, base_name)

def main():
    parser = argparse.ArgumentParser(description='Analyze Crazyflie Flight Logs')
    parser.add_argument('log_file', help='Path to CSV log file')
    parser.add_argument('--save', action='store_true', help='Save plots to file instead of showing them')
    parser.add_argument('--output', '-o', help='Output directory for plots (only used with --save)')

    args = parser.parse_args()

    analyze_log(args.log_file, args.save, args.output)

if __name__ == "__main__":
    main()

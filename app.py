from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from fuzzy_rules import FuzzyScales
from simulation_core import (
    PIDConfig,
    SimulationConfig,
    SimulationResult,
    ThermalParams,
    calculate_metrics,
    export_data_files,
    run_simulation,
)


matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

PARAMETER_FIELDS = (
    ("ambient_temp", "环境温度 (°C)", "37.0"),
    ("initial_temp", "初始温度 (°C)", "37.0"),
    ("target_temp", "目标温度 (°C)", "43.0"),
    ("duration", "仿真时长 (s)", "180.0"),
    ("dt", "时间步长 (s)", "0.1"),
    ("tau", "热时间常数 (s)", "45.0"),
    ("heat_gain", "激光加热增益 (°C/s)", "0.30"),
    ("kp", "基础 Kp", "0.18"),
    ("ki", "基础 Ki", "0.012"),
    ("kd", "基础 Kd", "0.20"),
    ("min_power_percent", "最小激光功率 (%)", "0.0"),
    ("max_power_percent", "最大激光功率 (%)", "100.0"),
    ("derivative_filter", "微分滤波系数", "0.85"),
    ("integral_limit", "积分上限 (°C·s)", "200.0"),
    ("noise_std", "测量噪声标准差 (°C)", "0.45"),
    ("kalman_q", "卡尔曼 Q（每步方差）", "0.0025"),
    ("kalman_r", "卡尔曼 R（°C²）", "0.2025"),
    ("error_scale", "模糊误差尺度", "6.0"),
    ("error_rate_scale", "模糊误差变化率尺度", "0.5"),
    ("delta_kp", "最大 ΔKp", "0.08"),
    ("delta_ki", "最大 ΔKi", "0.006"),
    ("delta_kd", "最大 ΔKd", "0.12"),
    ("seed", "随机种子", "20260817"),
)

SLIDER_RANGES = {
    "target_temp": (37.0, 55.0, 0.1),
    "kp": (0.0, 1.0, 0.005),
    "ki": (0.0, 0.2, 0.001),
    "kd": (0.0, 2.0, 0.01),
    "min_power_percent": (0.0, 90.0, 1.0),
    "max_power_percent": (10.0, 100.0, 1.0),
}


class ScrollableControls(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, width=450, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas, padding=10)
        self.window_id = self.canvas.create_window(
            (0, 0), window=self.content, anchor="nw"
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.content.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._fit_content_width)
        self.canvas.bind("<Enter>", self._enable_mousewheel)
        self.canvas.bind("<Leave>", self._disable_mousewheel)

    def _update_scroll_region(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _fit_content_width(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _enable_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _disable_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")


class SimulatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("组织升温与模糊 PID 激光功率控制仿真")
        self.root.geometry("1500x900")
        self.root.minsize(1180, 720)
        self.entries: dict[str, ttk.Entry] = {}
        self.scales: dict[str, tk.Scale] = {}
        self.controller_mode = tk.StringVar(value="fuzzy")
        self.use_kalman = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="调整参数后点击运行。默认值是教学仿真参数。")
        self.current_result: SimulationResult | None = None
        self.comparison_result: SimulationResult | None = None
        self._build_layout()
        self.run_current()

    def _build_layout(self) -> None:
        container = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        control_host = ScrollableControls(container)
        controls = control_host.content
        self.controls_canvas = control_host.canvas
        charts = ttk.Frame(container, padding=(0, 10, 10, 10))
        container.add(control_host, weight=0)
        container.add(charts, weight=1)

        ttk.Label(
            controls,
            text="控制与模型参数",
            font=("Microsoft YaHei UI", 12, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Label(controls, text="控制器").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.controller_mode,
            values=("classic", "fuzzy"),
            state="readonly",
            width=15,
        ).grid(row=1, column=1, columnspan=2, sticky="ew", pady=2)
        ttk.Checkbutton(
            controls,
            text="使用卡尔曼估计反馈",
            variable=self.use_kalman,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=2)

        for row, (name, label, default) in enumerate(PARAMETER_FIELDS, start=3):
            ttk.Label(controls, text=label).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=1
            )
            entry = ttk.Entry(controls, width=14)
            entry.insert(0, default)
            entry.grid(row=row, column=1, sticky="ew", pady=1)
            self.entries[name] = entry
            if name in SLIDER_RANGES:
                entry.bind("<FocusOut>", lambda _event, key=name: self._sync_scale(key))
                lower, upper, resolution = SLIDER_RANGES[name]
                scale = tk.Scale(
                    controls,
                    from_=lower,
                    to=upper,
                    resolution=resolution,
                    orient=tk.HORIZONTAL,
                    showvalue=False,
                    length=150,
                    highlightthickness=0,
                    command=lambda value, key=name: self._sync_entry(key, value),
                )
                scale.set(float(default))
                scale.grid(row=row, column=2, sticky="ew", padx=(6, 0), pady=1)
                self.scales[name] = scale

        button_row = 3 + len(PARAMETER_FIELDS)
        ttk.Button(controls, text="运行当前参数", command=self.run_current).grid(
            row=button_row, column=0, columnspan=3, sticky="ew", pady=(10, 2)
        )
        ttk.Button(
            controls,
            text="经典 PID 与模糊 PID 对比",
            command=self.run_comparison,
        ).grid(row=button_row + 1, column=0, columnspan=3, sticky="ew", pady=2)
        ttk.Button(controls, text="恢复默认参数", command=self.reset_defaults).grid(
            row=button_row + 2, column=0, columnspan=3, sticky="ew", pady=2
        )
        ttk.Button(controls, text="导出结果", command=self.export_current).grid(
            row=button_row + 3, column=0, columnspan=3, sticky="ew", pady=2
        )
        ttk.Label(
            controls,
            text="默认模型和控制参数用于教学，不是论文原始实验参数。",
            foreground="#8A4B08",
            wraplength=360,
        ).grid(row=button_row + 4, column=0, columnspan=3, sticky="w", pady=(8, 0))
        controls.columnconfigure(1, weight=1)

        self.figure = Figure(figsize=(10, 7), dpi=100, constrained_layout=True)
        self.axes = self.figure.subplots(2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=charts)
        toolbar = NavigationToolbar2Tk(self.canvas, charts, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.metrics_text = tk.Text(
            charts,
            height=5,
            wrap="word",
            font=("Consolas", 10),
            relief=tk.FLAT,
            background="#F5F5F5",
        )
        self.metrics_text.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(self.root, textvariable=self.status, anchor="w", padding=5).grid(
            row=1, column=0, sticky="ew"
        )

    def _sync_entry(self, name: str, value: str) -> None:
        entry = self.entries[name]
        entry.delete(0, tk.END)
        resolution = SLIDER_RANGES[name][2]
        digits = max(0, len(str(resolution).split(".")[-1].rstrip("0")))
        entry.insert(0, f"{float(value):.{digits}f}")

    def _sync_scale(self, name: str) -> None:
        try:
            value = self._number(name)
        except ValueError:
            return
        lower, upper, _ = SLIDER_RANGES[name]
        if lower <= value <= upper:
            self.scales[name].set(value)

    def _number(self, name: str) -> float:
        try:
            value = float(self.entries[name].get())
        except ValueError as exc:
            raise ValueError(f"{name} 必须是数字") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} 必须是有限数值")
        return value

    def read_config(self, mode: str | None = None) -> SimulationConfig:
        thermal = ThermalParams(
            ambient_temp=self._number("ambient_temp"),
            tau=self._number("tau"),
            heat_gain=self._number("heat_gain"),
        )
        pid = PIDConfig(
            kp=self._number("kp"),
            ki=self._number("ki"),
            kd=self._number("kd"),
            output_min=self._number("min_power_percent") / 100.0,
            output_max=self._number("max_power_percent") / 100.0,
            derivative_filter=self._number("derivative_filter"),
            integral_limit=self._number("integral_limit"),
        )
        fuzzy = FuzzyScales(
            error_scale=self._number("error_scale"),
            error_rate_scale=self._number("error_rate_scale"),
            delta_kp_max=self._number("delta_kp"),
            delta_ki_max=self._number("delta_ki"),
            delta_kd_max=self._number("delta_kd"),
        )
        return SimulationConfig(
            duration=self._number("duration"),
            dt=self._number("dt"),
            initial_temp=self._number("initial_temp"),
            target_temp=self._number("target_temp"),
            measurement_noise_std=self._number("noise_std"),
            kalman_q=self._number("kalman_q"),
            kalman_r=self._number("kalman_r"),
            use_kalman=self.use_kalman.get(),
            controller_mode=mode or self.controller_mode.get(),
            random_seed=int(self._number("seed")),
            thermal=thermal,
            pid=pid,
            fuzzy=fuzzy,
        )

    def run_current(self) -> None:
        try:
            self.current_result = run_simulation(self.read_config())
            self.comparison_result = None
            self.draw_results(self.current_result)
            self.status.set("仿真完成。默认模型用于教学，不代表真实治疗参数。")
        except Exception as exc:
            messagebox.showerror("参数或计算错误", str(exc), parent=self.root)

    def run_comparison(self) -> None:
        try:
            classic_config = self.read_config("classic")
            fuzzy_config = self.read_config("fuzzy")
            classic = run_simulation(classic_config)
            noise = classic.measurement_temp - classic.true_temp
            fuzzy = run_simulation(fuzzy_config, noise_sequence=noise)
            self.current_result = classic
            self.comparison_result = fuzzy
            self.draw_results(classic, fuzzy)
            self.status.set("已使用相同组织参数和噪声完成控制器对比。")
        except Exception as exc:
            messagebox.showerror("参数或计算错误", str(exc), parent=self.root)

    def draw_results(
        self,
        primary: SimulationResult,
        secondary: SimulationResult | None = None,
    ) -> None:
        for axis in self.axes.flat:
            axis.clear()
            axis.grid(True, alpha=0.25)
        temp_ax, power_ax, error_ax, gain_ax = self.axes.flat

        primary_label = primary.config.controller_mode
        temp_ax.plot(primary.time, primary.target_temp, "k--", label="目标温度")
        temp_ax.plot(
            primary.time,
            primary.true_temp,
            color="#0072B2",
            label=f"{primary_label} 真实温度" if secondary is not None else "真实温度",
        )
        temp_ax.scatter(
            primary.time[::10],
            primary.measurement_temp[::10],
            s=7,
            alpha=0.2,
            color="#777777",
            label="带噪测量",
        )
        if secondary is None:
            temp_ax.plot(
                primary.time,
                primary.estimated_temp,
                color="#009E73",
                label="卡尔曼估计",
            )
        power_ax.plot(
            primary.time,
            primary.power * 100.0,
            color="#D55E00",
            label=primary_label,
        )
        error_ax.plot(
            primary.time, primary.error, color="#CC79A7", label=primary_label
        )
        gain_ax.plot(primary.time, primary.kp, label=f"{primary_label} Kp")
        gain_ax.plot(primary.time, primary.ki, label=f"{primary_label} Ki")
        gain_ax.plot(primary.time, primary.kd, label=f"{primary_label} Kd")

        if secondary is not None:
            secondary_label = secondary.config.controller_mode
            temp_ax.plot(
                secondary.time,
                secondary.true_temp,
                color="#E69F00",
                label=f"{secondary_label} 真实温度",
            )
            power_ax.plot(
                secondary.time,
                secondary.power * 100.0,
                color="#56B4E9",
                label=secondary_label,
            )
            error_ax.plot(
                secondary.time,
                secondary.error,
                color="#009E73",
                label=secondary_label,
            )
            gain_ax.plot(
                secondary.time, secondary.kp, "--", label=f"{secondary_label} Kp"
            )
            gain_ax.plot(
                secondary.time, secondary.ki, "--", label=f"{secondary_label} Ki"
            )
            gain_ax.plot(
                secondary.time, secondary.kd, "--", label=f"{secondary_label} Kd"
            )

        temp_ax.set(title="温度响应", xlabel="时间 (s)", ylabel="温度 (°C)")
        power_ax.set(
            title="激光功率",
            xlabel="时间 (s)",
            ylabel="功率 (%)",
            ylim=(-2, 102),
        )
        error_ax.set(title="反馈温度误差", xlabel="时间 (s)", ylabel="误差 (°C)")
        gain_ax.set(title="PID 实时增益", xlabel="时间 (s)", ylabel="增益")
        for axis in (temp_ax, power_ax, error_ax):
            axis.legend(loc="best", fontsize=8)
        gain_ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=8
        )
        self.metrics_text.configure(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, self._format_metrics(primary, secondary))
        self.metrics_text.configure(state=tk.DISABLED)
        self.canvas.draw_idle()

    def _format_metrics(
        self,
        primary: SimulationResult,
        secondary: SimulationResult | None,
    ) -> str:
        def line(label: str, result: SimulationResult) -> str:
            metrics = calculate_metrics(result)
            rise_time = metrics["rise_time_s"]
            rise = "未达到" if rise_time is None else f"{rise_time:.1f} s"
            return (
                f'{label}: 超调 {metrics["overshoot_c"]:.3f} °C | '
                f'超调率 {metrics["overshoot_percent"]:.1f}% | '
                f"上升时间 {rise} | "
                f'稳态误差 {metrics["steady_state_error_c"]:.3f} °C | '
                f'控制 MAE {metrics["control_mae_c"]:.3f} °C | '
                f'测量 RMSE {metrics["measurement_rmse_c"]:.3f} °C | '
                f'卡尔曼 RMSE {metrics["kalman_rmse_c"]:.3f} °C | '
                f'归一化功率积分 {metrics["normalized_energy_s"]:.2f} s\n'
            )

        text = line(primary.config.controller_mode, primary)
        if secondary is not None:
            text += line(secondary.config.controller_mode, secondary)
        return text

    def reset_defaults(self) -> None:
        defaults = {name: default for name, _, default in PARAMETER_FIELDS}
        for name, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, defaults[name])
            if name in self.scales:
                self.scales[name].set(float(defaults[name]))
        self.controller_mode.set("fuzzy")
        self.use_kalman.set(True)
        self.run_current()

    def export_current(self) -> None:
        if self.current_result is None:
            messagebox.showinfo("没有结果", "请先运行仿真。", parent=self.root)
            return
        base = filedialog.askdirectory(
            initialdir=Path(__file__).parent / "outputs", parent=self.root
        )
        if not base:
            return
        directory = Path(base) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        try:
            export_data_files(self.current_result, directory)
            if self.comparison_result is not None:
                export_data_files(self.comparison_result, directory / "fuzzy_pid")
                (directory / "comparison.txt").write_text(
                    "根目录保存经典 PID 数据，fuzzy_pid 目录保存模糊 PID 数据。\n",
                    encoding="utf-8",
                )
            self.figure.savefig(directory / "curves.png", dpi=180, facecolor="white")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc), parent=self.root)
            return
        self.status.set(f"结果已导出到 {directory}")


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    SimulatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

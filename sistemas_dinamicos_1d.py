#!/usr/bin/env python3
"""
Simulador de Sistemas Dinámicos — Modelado y Simulación
Ing. Omar J. Cáceres
"""

import tkinter as tk
from tkinter import messagebox, ttk
import threading
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.lines import Line2D
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
#  PALETA
# ══════════════════════════════════════════════════════════════════
C = {
    "bg":       "#1a1a2e",
    "panel":    "#16213e",
    "accent":   "#7c3aed",
    "accent_h": "#6d28d9",
    "accent2":  "#06b6d4",
    "stable":   "#22c55e",
    "unstable": "#ef4444",
    "semi":     "#f59e0b",
    "text":     "#e2e8f0",
    "text2":    "#94a3b8",
    "border":   "#2d3748",
    "plot_bg":  "#0d0d1a",
    "grid":     "#1e2a3a",
    "entry_bg": "#0f172a",
}

# ══════════════════════════════════════════════════════════════════
#  ANÁLISIS NUMÉRICO
# ══════════════════════════════════════════════════════════════════

def _make_f(expr):
    """Convierte una expresión string en función f(x) vectorizada."""
    ns = {
        "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
        "tanh": np.tanh, "sinh": np.sinh, "cosh": np.cosh,
        "pi": np.pi, "e": np.e,
    }
    def f(x):
        local = dict(ns)
        local["x"] = x
        return eval(expr, local)
    return f


def _scalar(f):
    """Versión escalar segura de f para brentq."""
    def sf(xi):
        try:
            v = f(np.array([float(xi)]))
            return float(v[0]) if hasattr(v, "__len__") else float(v)
        except Exception:
            return np.nan
    return sf


def find_equilibria(f, xmin, xmax, n=3000):
    xs = np.linspace(xmin, xmax, n)
    try:
        fx = np.asarray(f(xs), dtype=float)
    except Exception:
        sf = _scalar(f)
        fx = np.array([sf(xi) for xi in xs])

    sf    = _scalar(f)
    roots = []
    for i in range(len(xs) - 1):
        fi, fi1 = fx[i], fx[i + 1]
        if not (np.isfinite(fi) and np.isfinite(fi1)):
            continue
        if abs(fi) < 1e-9:
            roots.append(xs[i])
        elif fi * fi1 < 0:
            try:
                roots.append(brentq(sf, xs[i], xs[i + 1], xtol=1e-11, maxiter=300))
            except Exception:
                pass

    if not roots:
        return []
    roots = sorted(roots)
    unique = [roots[0]]
    for r in roots[1:]:
        if abs(r - unique[-1]) > 1e-6:
            unique.append(r)
    return unique


def classify(f, xstar, h=1e-5):
    sf = _scalar(f)
    try:
        fp = (sf(xstar + h) - sf(xstar - h)) / (2 * h)
    except Exception:
        fp = 0.0
    if   fp < -1e-6: return "Estable",     fp, "stable"
    elif fp >  1e-6: return "Inestable",   fp, "unstable"
    else:            return "Semiestable", fp, "semi"


# ══════════════════════════════════════════════════════════════════
#  GRÁFICAS
# ══════════════════════════════════════════════════════════════════

TC = {"stable": C["stable"], "unstable": C["unstable"], "semi": C["semi"]}


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(C["plot_bg"])
    ax.set_title(title, color=C["text"], fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, color=C["text2"], fontsize=9)
    ax.set_ylabel(ylabel, color=C["text2"], fontsize=9)
    ax.tick_params(colors=C["text2"], labelsize=8)
    ax.grid(True, color=C["grid"], alpha=0.6, linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor(C["border"])


def plot_fx(ax, f, xmin, xmax, equil, stabs):
    xs = np.linspace(xmin, xmax, 900)
    try:
        fx = np.asarray(f(xs), dtype=float)
    except Exception:
        sf = _scalar(f)
        fx = np.array([sf(xi) for xi in xs])

    ax.plot(xs, fx, color=C["accent2"], linewidth=2.2, zorder=3, label="f(x)")
    ax.axhline(0, color=C["text2"], linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color=C["text2"], linewidth=0.4, alpha=0.3)

    ax.fill_between(xs, fx, 0, where=(fx > 0), alpha=0.14,
                    color=C["stable"],   label="f(x) > 0  →  x crece ↑")
    ax.fill_between(xs, fx, 0, where=(fx < 0), alpha=0.14,
                    color=C["unstable"], label="f(x) < 0  →  x decrece ↓")

    fy_max = max(np.nanmax(np.abs(fx[np.isfinite(fx)])) if np.any(np.isfinite(fx)) else 1, 0.5)
    for eq, (lbl, fp, stype) in zip(equil, stabs):
        col = TC[stype]
        ax.plot(eq, 0, "o", color=col, markersize=11, zorder=6)
        ax.annotate(f"x*={eq:.3f}", xy=(eq, 0),
                    xytext=(eq, fy_max * 0.2 + 0.1),
                    color=col, fontsize=8, ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.9), zorder=7)

    _style(ax, "f(x)  —  Campo de pendientes", "x", "f(x)")
    ax.legend(fontsize=7.5, facecolor=C["panel"], labelcolor=C["text"],
              framealpha=0.85, loc="upper right")


def plot_phase(ax, f, xmin, xmax, equil, stabs):
    """Diagrama de fase VERTICAL: eje y = variable x, flechas hacia arriba/abajo."""
    margin = abs(xmax - xmin) * 0.08
    ax.set_ylim(xmin - margin, xmax + margin)
    ax.set_xlim(-1.6, 1.6)

    # Línea vertical central (la "recta de fase")
    ax.axvline(0, color=C["text"], linewidth=3, alpha=0.85, zorder=2)

    # Flechas de flujo entre equilibrios
    sf      = _scalar(f)
    regions = [xmin] + list(equil) + [xmax]
    for i in range(len(regions) - 1):
        mid  = (regions[i] + regions[i + 1]) / 2.0
        fmid = sf(mid)
        if not np.isfinite(fmid):
            continue
        dy  = 0.22 * (regions[i + 1] - regions[i])
        col = C["stable"] if fmid > 0 else C["unstable"]
        # f > 0 → x crece → flecha SUBE; f < 0 → x decrece → flecha BAJA
        if fmid > 0:
            ax.annotate("", xy=(0, mid + dy), xytext=(0, mid - dy),
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=2.5, mutation_scale=20), zorder=3)
        else:
            ax.annotate("", xy=(0, mid - dy), xytext=(0, mid + dy),
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=2.5, mutation_scale=20), zorder=3)

    # Puntos de equilibrio sobre la línea vertical
    for eq, (lbl, fp, stype) in zip(equil, stabs):
        col = TC[stype]
        if stype == "stable":
            ax.plot(0, eq, "o", color=col, markersize=16, zorder=5)
        elif stype == "unstable":
            ax.plot(0, eq, "o", color=C["plot_bg"], markersize=16,
                    markeredgecolor=col, markeredgewidth=3, zorder=5)
        else:
            ax.plot(0, eq, "D", color=col, markersize=12, zorder=5)
        # Etiqueta a la derecha
        ax.text(0.25, eq, f"x* = {eq:.3f}", color=col, fontsize=8.5,
                ha="left", va="center", fontweight="bold")
        # Estabilidad a la izquierda
        ax.text(-0.25, eq, lbl, color=col, fontsize=8,
                ha="right", va="center", style="italic")

    handles = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C["stable"],
               markersize=9, label="Estable ●"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C["plot_bg"],
               markeredgecolor=C["unstable"], markeredgewidth=2.5,
               markersize=9, label="Inestable ○"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor=C["semi"],
               markersize=8, label="Semiestable ◑"),
    ]
    ax.legend(handles=handles, fontsize=7, facecolor=C["panel"],
              labelcolor=C["text"], framealpha=0.85, loc="lower right")

    _style(ax, "Diagrama de Fase", "", "x")
    ax.set_xticks([])
    ax.grid(False)
    # Solo grilla horizontal (líneas de nivel para y)
    ax.yaxis.grid(True, color=C["grid"], alpha=0.4, linewidth=0.5)
    ax.set_axisbelow(True)


def plot_time(ax, f, xmin, xmax, tmax, equil, stabs):
    sf     = _scalar(f)
    t_eval = np.linspace(0, tmax, 600)
    x0s    = list(np.linspace(xmin * 0.9, xmax * 0.9, 9))
    palette = plt.cm.plasma(np.linspace(0.08, 0.92, len(x0s)))

    for ic, color in zip(x0s, palette):
        try:
            sol = solve_ivp(lambda t, y: [sf(y[0])], (0, tmax), [ic],
                            t_eval=t_eval, method="RK45",
                            rtol=1e-7, atol=1e-9,
                            max_step=tmax / 200)
            if sol.success:
                y = np.clip(sol.y[0], xmin - 1.5, xmax + 1.5)
                ax.plot(sol.t, y, color=color, linewidth=1.7,
                        alpha=0.88, label=f"x₀={ic:.2f}")
        except Exception:
            pass

    for eq, (lbl, fp, stype) in zip(equil, stabs):
        ax.axhline(eq, color=TC[stype], linewidth=1.3,
                   linestyle="--", alpha=0.8, label=f"x*={eq:.3f} ({lbl})")

    margin = abs(xmax - xmin) * 0.1
    ax.set_ylim(xmin - margin, xmax + margin)
    _style(ax, "Evolución Temporal  x(t)", "t", "x(t)")
    ax.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["text"],
              framealpha=0.85, ncol=2, loc="upper right")


def run_analysis(expr, xmin, xmax, tmax, fig):
    f      = _make_f(expr)
    equil  = find_equilibria(f, xmin, xmax)
    stabs  = [classify(f, e) for e in equil]

    fig.clear()
    fig.patch.set_facecolor(C["bg"])

    # Layout:  f(x) arriba (ancho completo)
    #          diagrama de fase vertical (izq) | evolución temporal (der)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1, 3],
        height_ratios=[1, 1.4],
        hspace=0.42, wspace=0.28,
        left=0.08, right=0.97, top=0.96, bottom=0.07,
    )
    ax1 = fig.add_subplot(gs[0, :])   # f(x) — fila superior, ancho completo
    ax2 = fig.add_subplot(gs[1, 0])   # diagrama de fase vertical — izquierda
    ax3 = fig.add_subplot(gs[1, 1])   # evolución temporal — derecha

    plot_fx(ax1, f, xmin, xmax, equil, stabs)
    plot_phase(ax2, f, xmin, xmax, equil, stabs)
    plot_time(ax3, f, xmin, xmax, tmax, equil, stabs)

    return equil, stabs


# ══════════════════════════════════════════════════════════════════
#  VENTANA DE ANÁLISIS
# ══════════════════════════════════════════════════════════════════

class AnalysisWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Sistemas Dinámicos 1D Autónomos — Análisis")
        self.configure(bg=C["bg"])
        self.geometry("1300x860")
        self.minsize(1000, 700)
        self._build()

    def _build(self):
        _s = ttk.Style(self)
        _s.configure("Load.Horizontal.TProgressbar",
                      troughcolor=C["entry_bg"], background=C["accent"],
                      bordercolor=C["border"], thickness=10)

        # ── Header ────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=C["accent"], height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  📊  Sistemas Dinámicos Unidimensionales Autónomos",
                 bg=C["accent"], fg="white",
                 font=("Segoe UI", 12, "bold")).pack(side="left", padx=14, pady=10)
        tk.Label(hdr, text="dx/dt = f(x)  ",
                 bg=C["accent"], fg="#c4b5fd",
                 font=("Segoe UI", 10, "italic")).pack(side="right", padx=14)

        # ── Panel de entrada ──────────────────────────────────────
        inp = tk.Frame(self, bg=C["panel"])
        inp.pack(fill="x", padx=10, pady=(10, 4))

        # Fila 1: f(x)
        row1 = tk.Frame(inp, bg=C["panel"])
        row1.pack(fill="x", padx=14, pady=(12, 6))

        tk.Label(row1, text="f(x)  =",
                 bg=C["panel"], fg=C["accent2"],
                 font=("Segoe UI", 13, "bold")).pack(side="left", padx=(0, 10))

        self._expr_var = tk.StringVar(value="x**2 - 1")
        expr_entry = tk.Entry(row1, textvariable=self._expr_var,
                              font=("Consolas", 13), width=42,
                              bg=C["entry_bg"], fg=C["text"],
                              insertbackground=C["text"],
                              relief="flat", bd=6)
        expr_entry.pack(side="left", ipady=4)
        expr_entry.bind("<Return>", lambda e: self._run())

        tk.Label(row1,
                 text="  Funciones: sin, cos, tan, exp, log, sqrt, abs, tanh, pi, e",
                 bg=C["panel"], fg=C["text2"],
                 font=("Segoe UI", 8)).pack(side="left", padx=12)

        # Fila 2: parámetros numéricos + botón
        row2 = tk.Frame(inp, bg=C["panel"])
        row2.pack(fill="x", padx=14, pady=(0, 12))

        def lbl_entry(parent, label, var, w=8):
            tk.Label(parent, text=label, bg=C["panel"], fg=C["text2"],
                     font=("Segoe UI", 9)).pack(side="left", padx=(14, 4))
            e = tk.Entry(parent, textvariable=var, width=w,
                         bg=C["entry_bg"], fg=C["accent2"],
                         insertbackground=C["text"],
                         font=("Consolas", 9), relief="flat", bd=4)
            e.pack(side="left")
            e.bind("<Return>", lambda ev: self._run())

        self._xmin = tk.StringVar(value="-4")
        self._xmax = tk.StringVar(value="4")
        self._tmax = tk.StringVar(value="10")

        lbl_entry(row2, "x mínimo", self._xmin)
        lbl_entry(row2, "x máximo", self._xmax)
        lbl_entry(row2, "t máximo", self._tmax)

        # Botón analizar
        self._btn = tk.Button(
            row2, text="  ▶  ANALIZAR  ",
            bg=C["accent"], fg="white",
            font=("Segoe UI", 10, "bold"), relief="flat",
            cursor="hand2", pady=5,
            activebackground=C["accent_h"], activeforeground="white",
            command=self._run,
        )
        self._btn.pack(side="left", padx=20)

        self._pb = ttk.Progressbar(row2, mode="indeterminate", length=110,
                                    style="Load.Horizontal.TProgressbar")

        # Resultado texto
        self._rlbl = tk.Label(row2, text="", bg=C["panel"], fg=C["text"],
                               font=("Consolas", 9), justify="left", anchor="w")
        self._rlbl.pack(side="left", fill="x", expand=True)

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x", padx=10)

        # ── Gráficas ──────────────────────────────────────────────
        pf = tk.Frame(self, bg=C["bg"])
        pf.pack(fill="both", expand=True, padx=10, pady=(4, 8))

        self._fig    = plt.Figure(facecolor=C["bg"])
        self._canvas = FigureCanvasTkAgg(self._fig, master=pf)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        tbf = tk.Frame(pf, bg=C["panel"])
        tbf.pack(fill="x")
        tb = NavigationToolbar2Tk(self._canvas, tbf)
        tb.config(bg=C["panel"])
        tb.update()

        # Análisis inicial
        self._run()

    def _run(self):
        expr = self._expr_var.get().strip()
        if not expr:
            return
        try:
            xmin = float(self._xmin.get())
            xmax = float(self._xmax.get())
            tmax = float(self._tmax.get())
        except ValueError:
            messagebox.showerror("Error", "x mínimo, x máximo y t máximo deben ser números.",
                                 parent=self)
            return

        if xmin >= xmax:
            messagebox.showerror("Error", "x mínimo debe ser menor que x máximo.", parent=self)
            return

        self._btn.config(state="disabled", text="  ⏳  Analizando…")
        self._rlbl.configure(text="")
        self._pb.pack(side="left", padx=(0, 14))
        self._pb.start(10)

        def _work():
            try:
                equil, stabs = run_analysis(expr, xmin, xmax, tmax, self._fig)
                self.after(0, lambda: self._finish(equil, stabs, None))
            except Exception as exc:
                self.after(0, lambda: self._finish(None, None, exc))

        threading.Thread(target=_work, daemon=True).start()

    def _finish(self, equil, stabs, error):
        self._pb.stop()
        self._pb.pack_forget()
        self._btn.config(state="normal", text="  ▶  ANALIZAR  ")

        if error is not None:
            messagebox.showerror("Error en el análisis",
                                 f"No se pudo evaluar f(x):\n\n{error}", parent=self)
            return

        self._canvas.draw()

        if not equil:
            txt = "  No se encontraron puntos de equilibrio en el rango."
        else:
            lines = ["  PUNTOS DE EQUILIBRIO   "]
            for eq, (lbl, fp, stype) in zip(equil, stabs):
                sym = {"stable": "●", "unstable": "○", "semi": "◑"}[stype]
                lines.append(f"  {sym}  x* = {eq:+.5f}    f '(x*) = {fp:+.5f}    →  {lbl}")
            txt = "\n".join(lines)
        self._rlbl.configure(text=txt)


# ══════════════════════════════════════════════════════════════════
#  MENÚ PRINCIPAL
# ══════════════════════════════════════════════════════════════════

class MainMenu(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulador — Modelado y Simulación")
        self.configure(bg=C["bg"])
        self.resizable(False, False)
        self._build()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=C["accent"], height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  Simulador de Sistemas Dinámicos",
                 bg=C["accent"], fg="white",
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=16, pady=10)
        tk.Label(hdr, text="Modelado y Simulación  ·  Ing. Omar J. Cáceres  ",
                 bg=C["accent"], fg="#c4b5fd",
                 font=("Segoe UI", 8)).pack(side="right", padx=14)

        # Cuerpo
        body = tk.Frame(self, bg=C["bg"], padx=60, pady=50)
        body.pack()

        tk.Label(body, text="Seleccioná el tipo de sistema a analizar:",
                 bg=C["bg"], fg=C["text2"],
                 font=("Segoe UI", 10)).pack(pady=(0, 24))

        # ── Botón principal ───────────────────────────────────────
        btn = tk.Button(
            body,
            text="📈   Sistemas Unidimensionales Autónomos",
            bg=C["panel"], fg=C["text"],
            font=("Segoe UI", 12, "bold"),
            relief="flat", cursor="hand2",
            padx=40, pady=22, width=34,
            activebackground=C["accent"], activeforeground="white",
            command=self._open_1d,
        )
        btn.pack(fill="x")

        # Línea decorativa bajo el botón
        tk.Frame(body, bg=C["accent"], height=3, width=480).pack(pady=(0, 6))

        tk.Label(body, text="dx/dt = f(x)   —   análisis de equilibrios, estabilidad y diagramas",
                 bg=C["bg"], fg=C["text2"],
                 font=("Segoe UI", 8, "italic")).pack()

        # ── Espaciado para futuros botones ────────────────────────
        spacer = tk.Frame(body, bg=C["bg"], height=30)
        spacer.pack()

        tk.Label(body,
                 text="·  ·  ·     Más tipos de sistemas próximamente     ·  ·  ·",
                 bg=C["bg"], fg="#3a4a6a",
                 font=("Segoe UI", 8)).pack(pady=10)

        # Footer
        tk.Label(self,
                 text="Modelado y Simulación 3.1.025  ·  Segunda Edición 2026",
                 bg=C["bg"], fg="#2d3748",
                 font=("Segoe UI", 7)).pack(pady=(0, 10))

    def _open_1d(self):
        win = AnalysisWindow(self)
        win.grab_set()
        win.focus_force()


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = MainMenu()
    app.mainloop()

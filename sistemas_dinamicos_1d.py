#!/usr/bin/env python3
"""
Simulador de Sistemas Dinamicos -- Modelado y Simulacion
Ing. Omar J. Caceres
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
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings("ignore")

# =====================================================================
#  PALETA
# =====================================================================
C = {
    "bg":       "#1a1a2e",
    "panel":    "#16213e",
    "sidebar":  "#0f3460",
    "accent":   "#7c3aed",
    "accent_h": "#6d28d9",
    "accent2":  "#06b6d4",
    "accent3":  "#0e7490",
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

# =====================================================================
#  FUNCIONES DE EVALUACION
# =====================================================================

def _ns_base():
    return {
        "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
        "tanh": np.tanh, "sinh": np.sinh, "cosh": np.cosh,
        "pi": np.pi, "e": np.e,
    }

def _make_f(expr):
    ns = _ns_base()
    def f(x):
        local = dict(ns); local["x"] = x
        return eval(expr, local)
    return f

def _make_f_r(expr, r_val):
    ns = _ns_base(); ns["r"] = float(r_val)
    def f(x):
        local = dict(ns); local["x"] = x
        return eval(expr, local)
    return f

def _scalar(f):
    def sf(xi):
        try:
            v = f(np.array([float(xi)]))
            return float(v[0]) if hasattr(v, "__len__") else float(v)
        except Exception:
            return np.nan
    return sf

# =====================================================================
#  ANALISIS NUMERICO
# =====================================================================

def find_equilibria(f, xmin, xmax, n=5000):
    xs = np.linspace(xmin, xmax, n)
    try:
        fx = np.asarray(f(xs), dtype=float)
    except Exception:
        sf = _scalar(f)
        fx = np.array([sf(xi) for xi in xs])
    sf = _scalar(f)
    roots = []

    # ── Detección por cambio de signo ─────────────────────────────────
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

    # ── Detección de raíces dobles / silla-nodo (sin cambio de signo) ─
    # Busca mínimos locales de |f| que estén cerca de cero.
    # Solo agrega el candidato si NO está ya cubierto por brentq (evita duplicados).
    finite_mask = np.isfinite(fx)
    if np.any(finite_mask):
        f_scale = np.nanmax(np.abs(fx[finite_mask]))
        tol = max(f_scale * 0.015, 1e-4)
        absfx = np.abs(fx)
        for i in range(1, len(xs) - 1):
            if not (np.isfinite(fx[i-1]) and np.isfinite(fx[i]) and np.isfinite(fx[i+1])):
                continue
            if absfx[i] < absfx[i-1] and absfx[i] < absfx[i+1] and absfx[i] < tol:
                try:
                    lo = xs[max(0, i-3)]
                    hi = xs[min(len(xs)-1, i+3)]
                    res = minimize_scalar(lambda x: abs(sf(x)), bounds=(lo, hi),
                                         method='bounded', options={'xatol': 1e-10})
                    candidate = res.x
                    if abs(sf(candidate)) < tol:
                        # Solo agregar si no está ya cubierto por la detección de signo
                        if not any(abs(candidate - r) < 1e-4 for r in roots):
                            roots.append(candidate)
                except Exception:
                    if absfx[i] < tol * 0.5:
                        candidate = xs[i]
                        if not any(abs(candidate - r) < 1e-4 for r in roots):
                            roots.append(candidate)

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

# =====================================================================
#  BIFURCACIONES
# =====================================================================

def _refine_bif_r(expr, r_a, r_b, xmin, xmax, sig_a, n_iter=40):
    """Bisección para localizar con precisión el r de bifurcación."""
    for _ in range(n_iter):
        r_mid = (r_a + r_b) / 2.0
        f_mid = _make_f_r(expr, r_mid)
        eq_mid = find_equilibria(f_mid, xmin, xmax)
        st_mid = [classify(f_mid, e) for e in eq_mid]
        sig_mid = (len(eq_mid), tuple(sorted(s[2] for s in st_mid)))
        if sig_mid == sig_a:
            r_a = r_mid
        else:
            r_b = r_mid
    return (r_a + r_b) / 2.0

def compute_bifurcation(expr, r_min, r_max, xmin, xmax, n_r=500):
    r_vals   = np.linspace(r_min, r_max, n_r)
    data     = []
    bif_pts  = []
    prev_sig = None
    prev_r   = r_min
    for r in r_vals:
        f     = _make_f_r(expr, r)
        equil = find_equilibria(f, xmin, xmax)
        stabs = [classify(f, e) for e in equil]
        types = tuple(sorted(s[2] for s in stabs))
        sig   = (len(equil), types)
        for eq, (lbl, fp, stype) in zip(equil, stabs):
            data.append((r, eq, stype))
        if prev_sig is not None and sig != prev_sig:
            # Refinar el punto exacto de bifurcación con bisección
            r_bif = _refine_bif_r(expr, prev_r, r, xmin, xmax, prev_sig)
            n_prev, _ = prev_sig
            n_curr, _ = sig
            if n_prev != n_curr:
                desc = "Silla-Nodo ({} -> {} equilibrios)".format(n_prev, n_curr)
            else:
                desc = "Cambio de estabilidad (Transcritica / Pitchfork)"
            bif_pts.append((r_bif, desc))
        prev_sig = sig
        prev_r   = r
    if len(bif_pts) > 1:
        tol    = (r_max - r_min) * 0.015
        unique = [bif_pts[0]]
        for bp in bif_pts[1:]:
            if abs(bp[0] - unique[-1][0]) > tol:
                unique.append(bp)
        bif_pts = unique
    return data, bif_pts

# =====================================================================
#  GRAFICAS
# =====================================================================

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
        sf = _scalar(f); fx = np.array([sf(xi) for xi in xs])
    ax.plot(xs, fx, color=C["accent2"], linewidth=2.2, zorder=3, label="f(x)")
    ax.axhline(0, color=C["text2"], linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color=C["text2"], linewidth=0.4, alpha=0.3)
    ax.fill_between(xs, fx, 0, where=(fx > 0), alpha=0.14,
                    color=C["stable"],   label="f(x)>0 -> x crece")
    ax.fill_between(xs, fx, 0, where=(fx < 0), alpha=0.14,
                    color=C["unstable"], label="f(x)<0 -> x decrece")
    fy_max = max(np.nanmax(np.abs(fx[np.isfinite(fx)])) if np.any(np.isfinite(fx)) else 1, 0.5)
    for eq, (lbl, fp, stype) in zip(equil, stabs):
        col = TC[stype]
        ax.plot(eq, 0, "o", color=col, markersize=11, zorder=6)
        ax.annotate("x*={:.3f}".format(eq), xy=(eq, 0),
                    xytext=(eq, fy_max * 0.2 + 0.1),
                    color=col, fontsize=8, ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.9), zorder=7)
    _style(ax, "f(x) -- Campo de pendientes", "x", "f(x)")
    ax.legend(fontsize=7.5, facecolor=C["panel"], labelcolor=C["text"],
              framealpha=0.85, loc="upper right")

def plot_phase(ax, f, xmin, xmax, equil, stabs):
    margin = abs(xmax - xmin) * 0.08
    ax.set_ylim(xmin - margin, xmax + margin)
    ax.set_xlim(-1.6, 1.6)
    ax.axvline(0, color=C["text"], linewidth=3, alpha=0.85, zorder=2)
    sf      = _scalar(f)
    regions = [xmin] + list(equil) + [xmax]
    for i in range(len(regions) - 1):
        mid  = (regions[i] + regions[i + 1]) / 2.0
        fmid = sf(mid)
        if not np.isfinite(fmid):
            continue
        dy  = 0.22 * (regions[i + 1] - regions[i])
        col = C["stable"] if fmid > 0 else C["unstable"]
        if fmid > 0:
            ax.annotate("", xy=(0, mid + dy), xytext=(0, mid - dy),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5, mutation_scale=20), zorder=3)
        else:
            ax.annotate("", xy=(0, mid - dy), xytext=(0, mid + dy),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5, mutation_scale=20), zorder=3)
    for eq, (lbl, fp, stype) in zip(equil, stabs):
        col = TC[stype]
        if stype == "stable":
            ax.plot(0, eq, "o", color=col, markersize=16, zorder=5)
        elif stype == "unstable":
            ax.plot(0, eq, "o", color=C["plot_bg"], markersize=16,
                    markeredgecolor=col, markeredgewidth=3, zorder=5)
        else:
            ax.plot(0, eq, "D", color=col, markersize=12, zorder=5)
        ax.text( 0.25, eq, "x*={:.3f}".format(eq), color=col, fontsize=8.5,
                ha="left", va="center", fontweight="bold")
        ax.text(-0.25, eq, lbl, color=col, fontsize=8,
                ha="right", va="center", style="italic")
    handles = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C["stable"], markersize=9, label="Estable"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C["plot_bg"],
               markeredgecolor=C["unstable"], markeredgewidth=2.5, markersize=9, label="Inestable"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor=C["semi"], markersize=8, label="Semiestable"),
    ]
    ax.legend(handles=handles, fontsize=7, facecolor=C["panel"], labelcolor=C["text"],
              framealpha=0.85, loc="lower right")
    _style(ax, "Diagrama de Fase", "", "x")
    ax.set_xticks([])
    ax.grid(False)
    ax.yaxis.grid(True, color=C["grid"], alpha=0.4, linewidth=0.5)
    ax.set_axisbelow(True)

def plot_time(ax, f, xmin, xmax, tmax, equil, stabs):
    sf      = _scalar(f)
    t_eval  = np.linspace(0, tmax, 600)
    x0s     = list(np.linspace(xmin * 0.9, xmax * 0.9, 9))
    palette = plt.cm.plasma(np.linspace(0.08, 0.92, len(x0s)))
    for ic, color in zip(x0s, palette):
        try:
            sol = solve_ivp(lambda t, y: [sf(y[0])], (0, tmax), [ic],
                            t_eval=t_eval, method="RK45",
                            rtol=1e-7, atol=1e-9, max_step=tmax / 200)
            if sol.success:
                y = np.clip(sol.y[0], xmin - 1.5, xmax + 1.5)
                ax.plot(sol.t, y, color=color, linewidth=1.7, alpha=0.88,
                        label="x0={:.2f}".format(ic))
        except Exception:
            pass
    for eq, (lbl, fp, stype) in zip(equil, stabs):
        ax.axhline(eq, color=TC[stype], linewidth=1.3, linestyle="--", alpha=0.8,
                   label="x*={:.3f} ({})".format(eq, lbl))
    margin = abs(xmax - xmin) * 0.1
    ax.set_ylim(xmin - margin, xmax + margin)
    _style(ax, "Evolucion Temporal x(t)", "t", "x(t)")
    ax.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["text"],
              framealpha=0.85, ncol=2, loc="upper right")

def _extract_branches(points, max_gap):
    """Agrupa puntos (r, x) en ramas continuas usando greedy matching."""
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    # Agrupar por r
    r_groups = {}
    for r, x in sorted_pts:
        r_groups.setdefault(r, []).append(x)
    r_vals = sorted(r_groups)
    # Inicializar ramas con el primer r
    branches = [[(r_vals[0], x)] for x in r_groups[r_vals[0]]]
    for r in r_vals[1:]:
        xs_new = sorted(r_groups[r])
        used = set()
        unmatched = []
        for x_new in xs_new:
            best_b, best_d = None, max_gap
            for bi, branch in enumerate(branches):
                if bi in used:
                    continue
                d = abs(x_new - branch[-1][1])
                if d < best_d:
                    best_d, best_b = d, bi
            if best_b is not None:
                branches[best_b].append((r, x_new))
                used.add(best_b)
            else:
                unmatched.append(x_new)
        for x_new in unmatched:
            branches.append([(r, x_new)])
    return branches


def plot_bifurcation_diagram(ax, data, bif_pts, r_min, r_max, r_sel=None):
    ax.set_facecolor(C["plot_bg"])
    if not data:
        ax.text(0.5, 0.5, "No se encontraron equilibrios en el rango.",
                transform=ax.transAxes, color=C["text2"], ha="center", va="center")
        _style(ax, "Diagrama de Bifurcacion", "r", "x*")
        return

    # Threshold adaptativo al rango de x*
    all_xs = [d[1] for d in data]
    x_span = max(all_xs) - min(all_xs) if len(all_xs) > 1 else 1.0
    max_gap = max(x_span * 0.12, 0.08)

    # Estilo por tipo: (stype, color, label, linestyle)
    styles = [
        ("stable",   C["stable"],   "Estable",     "-",  2.5),
        ("unstable", C["unstable"], "Inestable",   "--", 2.0),
        ("semi",     C["semi"],     "Semiestable", ":",  2.0),
    ]
    for stype, col, lbl, ls, lw in styles:
        points = [(d[0], d[1]) for d in data if d[2] == stype]
        branches = _extract_branches(points, max_gap)
        first = True
        for branch in branches:
            if len(branch) < 2:
                ax.plot([branch[0][0]], [branch[0][1]], "o", color=col,
                        markersize=5, zorder=4)
                continue
            rs = [p[0] for p in branch]
            xs = [p[1] for p in branch]
            ax.plot(rs, xs, color=col, linewidth=lw, linestyle=ls, alpha=0.92,
                    label=lbl if first else "_nolegend_", zorder=3)
            first = False

    ax.axhline(0, color=C["text2"], linewidth=0.5, linestyle="--", alpha=0.35)
    y_lo, y_hi = ax.get_ylim()
    y_span = y_hi - y_lo if y_hi != y_lo else 1.0
    for r_b, desc in bif_pts:
        ax.axvline(r_b, color=C["accent"], linewidth=1.5, linestyle="--", alpha=0.88, zorder=4)
        ax.text(r_b, y_hi - y_span * 0.02, " r={:.3f}".format(r_b),
                color=C["accent"], fontsize=7, ha="left", va="top", rotation=90, zorder=6)
    if r_sel is not None:
        ax.axvline(r_sel, color="white", linewidth=2, linestyle="-", alpha=0.9, zorder=5,
                   label="r actual={:.2f}".format(r_sel))
    _style(ax, "Diagrama de Bifurcacion -- x* vs r", "r (parametro)", "x* (equilibrios)")
    ax.legend(fontsize=7.5, facecolor=C["panel"], labelcolor=C["text"],
              framealpha=0.85, loc="upper right")

# =====================================================================
#  ORCHESTADOR 1D
# =====================================================================

def run_analysis(expr, xmin, xmax, tmax, fig):
    f     = _make_f(expr)
    equil = find_equilibria(f, xmin, xmax)
    stabs = [classify(f, e) for e in equil]
    fig.clear()
    fig.patch.set_facecolor(C["bg"])
    gs = fig.add_gridspec(2, 2,
        width_ratios=[1, 3], height_ratios=[1, 1.4],
        hspace=0.42, wspace=0.28,
        left=0.08, right=0.97, top=0.96, bottom=0.07)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    plot_fx(ax1, f, xmin, xmax, equil, stabs)
    plot_phase(ax2, f, xmin, xmax, equil, stabs)
    plot_time(ax3, f, xmin, xmax, tmax, equil, stabs)
    return equil, stabs

# =====================================================================
#  VENTANA: SISTEMAS 1D AUTONOMOS
# =====================================================================

class AnalysisWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Sistemas Dinamicos 1D Autonomos")
        self.configure(bg=C["bg"])
        self.geometry("1300x860")
        self.minsize(1000, 700)
        self._build()

    def _build(self):
        _s = ttk.Style(self)
        _s.configure("Load.Horizontal.TProgressbar",
                      troughcolor=C["entry_bg"], background=C["accent"],
                      bordercolor=C["border"], thickness=10)

        hdr = tk.Frame(self, bg=C["accent"], height=48)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="  Sistemas Dinamicos Unidimensionales Autonomos",
                 bg=C["accent"], fg="white",
                 font=("Segoe UI", 12, "bold")).pack(side="left", padx=14, pady=10)
        tk.Label(hdr, text="dx/dt = f(x)  ",
                 bg=C["accent"], fg="#c4b5fd",
                 font=("Segoe UI", 10, "italic")).pack(side="right", padx=14)

        inp = tk.Frame(self, bg=C["panel"])
        inp.pack(fill="x", padx=10, pady=(10, 4))

        row1 = tk.Frame(inp, bg=C["panel"])
        row1.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(row1, text="f(x)  =", bg=C["panel"], fg=C["accent2"],
                 font=("Segoe UI", 13, "bold")).pack(side="left", padx=(0, 10))
        self._expr_var = tk.StringVar(value="x**2 - 1")
        e = tk.Entry(row1, textvariable=self._expr_var, font=("Consolas", 13), width=42,
                     bg=C["entry_bg"], fg=C["text"], insertbackground=C["text"],
                     relief="flat", bd=6)
        e.pack(side="left", ipady=4)
        e.bind("<Return>", lambda ev: self._run())
        tk.Label(row1, text="  sin, cos, tan, exp, log, sqrt, abs, tanh, pi, e",
                 bg=C["panel"], fg=C["text2"], font=("Segoe UI", 8)).pack(side="left", padx=12)

        row2 = tk.Frame(inp, bg=C["panel"])
        row2.pack(fill="x", padx=14, pady=(0, 12))

        def lbl_e(p, lbl, var, w=8):
            tk.Label(p, text=lbl, bg=C["panel"], fg=C["text2"],
                     font=("Segoe UI", 9)).pack(side="left", padx=(14, 4))
            en = tk.Entry(p, textvariable=var, width=w, bg=C["entry_bg"], fg=C["accent2"],
                          insertbackground=C["text"], font=("Consolas", 9), relief="flat", bd=4)
            en.pack(side="left"); en.bind("<Return>", lambda ev: self._run())

        self._xmin = tk.StringVar(value="-4")
        self._xmax = tk.StringVar(value="4")
        self._tmax = tk.StringVar(value="10")
        lbl_e(row2, "x minimo", self._xmin)
        lbl_e(row2, "x maximo", self._xmax)
        lbl_e(row2, "t maximo", self._tmax)

        self._btn = tk.Button(row2, text="  ANALIZAR  ", bg=C["accent"], fg="white",
                               font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2",
                               pady=5, activebackground=C["accent_h"], activeforeground="white",
                               command=self._run)
        self._btn.pack(side="left", padx=20)
        self._pb = ttk.Progressbar(row2, mode="indeterminate", length=110,
                                    style="Load.Horizontal.TProgressbar")
        self._rlbl = tk.Label(row2, text="", bg=C["panel"], fg=C["text"],
                               font=("Consolas", 9), justify="left", anchor="w")
        self._rlbl.pack(side="left", fill="x", expand=True)

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x", padx=10)

        pf = tk.Frame(self, bg=C["bg"])
        pf.pack(fill="both", expand=True, padx=10, pady=(4, 8))
        self._fig    = plt.Figure(facecolor=C["bg"])
        self._canvas = FigureCanvasTkAgg(self._fig, master=pf)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        tbf = tk.Frame(pf, bg=C["panel"]); tbf.pack(fill="x")
        tb = NavigationToolbar2Tk(self._canvas, tbf)
        tb.config(bg=C["panel"]); tb.update()
        self._run()

    def _run(self):
        expr = self._expr_var.get().strip()
        if not expr: return
        try:
            xmin = float(self._xmin.get())
            xmax = float(self._xmax.get())
            tmax = float(self._tmax.get())
        except ValueError:
            messagebox.showerror("Error", "Los rangos deben ser numeros.", parent=self); return
        if xmin >= xmax:
            messagebox.showerror("Error", "x minimo debe ser menor que x maximo.", parent=self); return

        self._btn.config(state="disabled", text="  Analizando...")
        self._rlbl.configure(text="")
        self._pb.pack(side="left", padx=(0, 14)); self._pb.start(10)

        def _work():
            try:
                eq, st = run_analysis(expr, xmin, xmax, tmax, self._fig)
                self.after(0, lambda: self._finish(eq, st, None))
            except Exception as exc:
                self.after(0, lambda: self._finish(None, None, exc))
        threading.Thread(target=_work, daemon=True).start()

    def _finish(self, equil, stabs, error):
        self._pb.stop(); self._pb.pack_forget()
        self._btn.config(state="normal", text="  ANALIZAR  ")
        if error is not None:
            messagebox.showerror("Error", "No se pudo evaluar f(x):\n\n{}".format(error), parent=self); return
        self._canvas.draw()
        if not equil:
            txt = "  No se encontraron puntos de equilibrio."
        else:
            sym_map = {"stable": "●", "unstable": "○", "semi": "◑"}
            lines = ["  PUNTOS DE EQUILIBRIO"]
            for eq, (lbl, fp, stype) in zip(equil, stabs):
                lines.append("  {}  x*={:+.5f}   f'(x*)={:+.5f}   -> {}".format(
                    sym_map[stype], eq, fp, lbl))
            txt = "\n".join(lines)
        self._rlbl.configure(text=txt)

# =====================================================================
#  VENTANA: BIFURCACIONES
# =====================================================================

class BifurcationWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Bifurcaciones -- Analisis")
        self.configure(bg=C["bg"])
        self.geometry("1340x900")
        self.minsize(1050, 720)
        self._bif_data   = []
        self._bif_pts    = []
        self._expr_cache = "r*x - x**3"
        self._xmin_cache = -3.0
        self._xmax_cache =  3.0
        self._tmax_cache = 10.0
        self._rmin_cache = -2.0
        self._rmax_cache =  2.0
        self._build()

    def _build(self):
        _s = ttk.Style(self)
        _s.configure("Bif.Horizontal.TProgressbar",
                      troughcolor=C["entry_bg"], background=C["sidebar"],
                      bordercolor=C["border"], thickness=10)

        hdr = tk.Frame(self, bg=C["sidebar"], height=48)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="  Analisis de Bifurcaciones",
                 bg=C["sidebar"], fg="white",
                 font=("Segoe UI", 12, "bold")).pack(side="left", padx=14, pady=10)
        tk.Label(hdr, text="x_dot = f(x, r)  ",
                 bg=C["sidebar"], fg="#c4b5fd",
                 font=("Segoe UI", 10, "italic")).pack(side="right", padx=14)

        inp = tk.Frame(self, bg=C["panel"])
        inp.pack(fill="x", padx=10, pady=(10, 4))

        row1 = tk.Frame(inp, bg=C["panel"])
        row1.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(row1, text="f(x, r)  =", bg=C["panel"], fg=C["accent2"],
                 font=("Segoe UI", 13, "bold")).pack(side="left", padx=(0, 10))
        self._expr_var = tk.StringVar(value="r*x - x**3")
        ent = tk.Entry(row1, textvariable=self._expr_var, font=("Consolas", 13), width=36,
                       bg=C["entry_bg"], fg=C["text"], insertbackground=C["text"],
                       relief="flat", bd=6)
        ent.pack(side="left", ipady=4)
        ent.bind("<Return>", lambda ev: self._run())
        tk.Label(row1, text="  Variables: x, r   |   sin, cos, exp, log, sqrt, abs, pi, e",
                 bg=C["panel"], fg=C["text2"], font=("Segoe UI", 8)).pack(side="left", padx=10)

        row2 = tk.Frame(inp, bg=C["panel"])
        row2.pack(fill="x", padx=14, pady=(0, 6))

        def field(p, lbl, var, w=7):
            tk.Label(p, text=lbl, bg=C["panel"], fg=C["text2"],
                     font=("Segoe UI", 9)).pack(side="left", padx=(12, 3))
            en = tk.Entry(p, textvariable=var, width=w, bg=C["entry_bg"], fg=C["accent2"],
                          insertbackground=C["text"], font=("Consolas", 9), relief="flat", bd=4)
            en.pack(side="left"); en.bind("<Return>", lambda ev: self._run())

        self._rmin_v = tk.StringVar(value="-2")
        self._rmax_v = tk.StringVar(value="2")
        self._xmin_v = tk.StringVar(value="-3")
        self._xmax_v = tk.StringVar(value="3")
        self._tmax_v = tk.StringVar(value="10")
        field(row2, "r minimo", self._rmin_v)
        field(row2, "r maximo", self._rmax_v)
        field(row2, "x minimo", self._xmin_v)
        field(row2, "x maximo", self._xmax_v)
        field(row2, "t maximo", self._tmax_v)

        self._btn = tk.Button(row2, text="  ANALIZAR  ", bg=C["sidebar"], fg="white",
                               font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2",
                               pady=5, activebackground=C["accent"], activeforeground="white",
                               command=self._run)
        self._btn.pack(side="left", padx=16)
        self._pb = ttk.Progressbar(row2, mode="indeterminate", length=100,
                                    style="Bif.Horizontal.TProgressbar")
        self._rlbl = tk.Label(row2, text="", bg=C["panel"], fg=C["text"],
                               font=("Consolas", 8), justify="left", anchor="w")
        self._rlbl.pack(side="left", fill="x", expand=True)

        row3 = tk.Frame(inp, bg=C["panel"])
        row3.pack(fill="x", padx=14, pady=(2, 10))
        tk.Label(row3, text="r  (fase / tiempo):",
                 bg=C["panel"], fg=C["text2"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 8))
        self._r_var = tk.DoubleVar(value=1.0)
        self._slider = tk.Scale(row3, variable=self._r_var,
            from_=-2, to=2, resolution=0.01, orient="horizontal", length=320,
            bg=C["panel"], fg=C["text"], troughcolor=C["entry_bg"],
            highlightthickness=0, activebackground=C["accent"],
            showvalue=False, command=lambda v: self._on_slider())
        self._slider.pack(side="left")
        self._r_lbl = tk.Label(row3, text="r = 1.00", bg=C["panel"], fg=C["accent2"],
                                font=("Consolas", 11, "bold"))
        self._r_lbl.pack(side="left", padx=10)

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x", padx=10)

        pf = tk.Frame(self, bg=C["bg"])
        pf.pack(fill="both", expand=True, padx=10, pady=(4, 8))
        self._fig    = plt.Figure(facecolor=C["bg"])
        self._canvas = FigureCanvasTkAgg(self._fig, master=pf)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        tbf = tk.Frame(pf, bg=C["panel"]); tbf.pack(fill="x")
        tb = NavigationToolbar2Tk(self._canvas, tbf)
        tb.config(bg=C["panel"]); tb.update()
        self._run()

    def _get_params(self):
        return (self._expr_var.get().strip(),
                float(self._rmin_v.get()), float(self._rmax_v.get()),
                float(self._xmin_v.get()), float(self._xmax_v.get()),
                float(self._tmax_v.get()))

    def _on_slider(self):
        r_sel = float(self._r_var.get())
        self._r_lbl.configure(text="r = {:.2f}".format(r_sel))
        if self._bif_data:
            self._draw_all(r_sel)

    def _run(self):
        try:
            expr, r_min, r_max, xmin, xmax, tmax = self._get_params()
        except ValueError:
            messagebox.showerror("Error", "Todos los rangos deben ser numeros.", parent=self); return
        if r_min >= r_max:
            messagebox.showerror("Error", "r minimo debe ser menor que r maximo.", parent=self); return

        self._slider.config(from_=r_min, to=r_max)
        r_sel = max(r_min, min(r_max, float(self._r_var.get())))
        self._r_var.set(r_sel)

        self._btn.config(state="disabled", text="  Calculando...")
        self._rlbl.configure(text="")
        self._pb.pack(side="left", padx=(0, 10)); self._pb.start(10)

        def _work():
            try:
                data, bif_pts = compute_bifurcation(expr, r_min, r_max, xmin, xmax)
                self.after(0, lambda: self._finish(
                    data, bif_pts, expr, r_min, r_max, xmin, xmax, tmax, r_sel, None))
            except Exception as exc:
                self.after(0, lambda: self._finish(
                    None, None, None, None, None, None, None, None, None, exc))
        threading.Thread(target=_work, daemon=True).start()

    def _finish(self, data, bif_pts, expr, r_min, r_max, xmin, xmax, tmax, r_sel, error):
        self._pb.stop(); self._pb.pack_forget()
        self._btn.config(state="normal", text="  ANALIZAR  ")
        if error is not None:
            messagebox.showerror("Error", "Error al analizar:\n\n{}".format(error), parent=self); return
        self._bif_data   = data
        self._bif_pts    = bif_pts
        self._expr_cache = expr
        self._rmin_cache = r_min; self._rmax_cache = r_max
        self._xmin_cache = xmin; self._xmax_cache = xmax
        self._tmax_cache = tmax
        self._draw_all(r_sel)

    def _draw_all(self, r_sel):
        expr = self._expr_cache
        xmin = self._xmin_cache; xmax = self._xmax_cache
        tmax = self._tmax_cache
        r_min = self._rmin_cache; r_max = self._rmax_cache

        self._fig.clear()
        self._fig.patch.set_facecolor(C["bg"])
        gs = self._fig.add_gridspec(2, 2,
            width_ratios=[1, 3], height_ratios=[1, 1.35],
            hspace=0.44, wspace=0.28,
            left=0.08, right=0.97, top=0.96, bottom=0.07)
        ax_bif  = self._fig.add_subplot(gs[0, :])
        ax_fase = self._fig.add_subplot(gs[1, 0])
        ax_time = self._fig.add_subplot(gs[1, 1])

        plot_bifurcation_diagram(ax_bif, self._bif_data, self._bif_pts, r_min, r_max, r_sel)

        f_r   = _make_f_r(expr, r_sel)
        equil = find_equilibria(f_r, xmin, xmax)
        stabs = [classify(f_r, e) for e in equil]

        plot_phase(ax_fase, f_r, xmin, xmax, equil, stabs)
        ax_fase.set_title("Diagrama de Fase  (r={:.2f})".format(r_sel),
                          color=C["text"], fontsize=9.5, fontweight="bold", pad=5)

        plot_time(ax_time, f_r, xmin, xmax, tmax, equil, stabs)
        ax_time.set_title("Evolucion Temporal  (r={:.2f})".format(r_sel),
                          color=C["text"], fontsize=9.5, fontweight="bold", pad=5)

        self._canvas.draw()

        # ── Actualizar label con equilibrios y bifurcaciones ──────────
        sym = {"stable": "●", "unstable": "○", "semi": "◑"}
        parts = []
        if self._bif_pts:
            blines = ["  BIFURCACIONES:"]
            for r_b, desc in self._bif_pts:
                blines.append("  * r = {:+.4f}  —  {}".format(r_b, desc))
            parts.append("\n".join(blines))
        if equil:
            elines = ["  EQUILIBRIOS  (r = {:.3f}):".format(r_sel)]
            for eq, (lbl, fp, stype) in zip(equil, stabs):
                elines.append("  {}  x* = {:+.5f}   f'(x*) = {:+.5f}   →  {}".format(
                    sym[stype], eq, fp, lbl))
            parts.append("\n".join(elines))
        else:
            parts.append("  Sin equilibrios en el rango para r = {:.3f}".format(r_sel))
        self._rlbl.configure(text="\n".join(parts))

# =====================================================================
#  ANALISIS 2D LINEAL -- funciones auxiliares
# =====================================================================

def _fmt_complex(z):
    z = complex(z)
    if abs(z.imag) < 1e-9:
        return "{:.5f}".format(z.real)
    return "{:.4f}{:+.4f}i".format(z.real, z.imag)


def _classify_2d(vals, A):
    lam1, lam2 = complex(vals[0]), complex(vals[1])
    scale = max(abs(lam1), abs(lam2), 1.0)
    tol   = 1e-8 * scale
    if abs(lam1.imag) > tol:               # Valores complejos conjugados
        alpha = lam1.real
        if   alpha < -tol: return "Espiral Estable",    "Asintoticamente Estable"
        elif alpha >  tol: return "Espiral Inestable",  "Inestable"
        else:              return "Centro",              "Estable"
    r1, r2 = lam1.real, lam2.real
    teq = 1e-7 * max(abs(r1), abs(r2), 1.0)
    if abs(r1 - r2) < teq:                 # Valores reales repetidos
        lam  = (r1 + r2) / 2.0
        rank = np.linalg.matrix_rank(A - lam * np.eye(2), tol=1e-8)
        tipo = "Nodo Estrella" if rank == 0 else "Nodo Degenerado"
        if   lam < -teq: estab = "Asintoticamente Estable"
        elif lam >  teq: estab = "Inestable"
        else:            estab = "Estable"
        return tipo, estab
    if r1 * r2 < 0:              return "Punto de Silla", "Inestable"
    if r1 < -tol and r2 < -tol:  return "Nodo Estable",   "Asintoticamente Estable"
    if r1 >  tol and r2 >  tol:  return "Nodo Inestable",  "Inestable"
    return "No Hiperbolico", "Indeterminado"


def _plot_phase_2d(ax, A, xmin, xmax, ymin, ymax, tmax, tipo):
    ax.set_facecolor(C["plot_bg"])
    nx, ny = 22, 22
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts  = np.stack([XX.ravel(), YY.ravel()])
    dpts = A @ pts
    UU   = dpts[0].reshape(ny, nx)
    VV   = dpts[1].reshape(ny, nx)
    speed = np.sqrt(UU**2 + VV**2) + 1e-12
    ax.streamplot(xs, ys, UU, VV,
                  color=speed / speed.max(),
                  cmap="plasma", linewidth=0.9,
                  arrowsize=1.0, density=1.2,
                  broken_streamlines=False)
    r0     = min(abs(xmax - xmin), abs(ymax - ymin)) * 0.35
    t_ev   = np.linspace(0, tmax, 800)
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    cols   = plt.cm.cool(np.linspace(0.1, 0.9, 8))
    for th, col in zip(angles, cols):
        x0, y0 = r0 * np.cos(th), r0 * np.sin(th)
        try:
            sol = solve_ivp(lambda t, y: A @ y, (0, tmax), [x0, y0],
                            t_eval=t_ev, method="RK45", rtol=1e-7, atol=1e-9)
            if sol.success:
                xc = np.clip(sol.y[0], xmin * 3, xmax * 3)
                yc = np.clip(sol.y[1], ymin * 3, ymax * 3)
                ax.plot(xc, yc, color=col, linewidth=1.4, alpha=0.85)
        except Exception:
            pass
    ax.plot(0, 0, "*", color="white", markersize=12, zorder=8, markeredgecolor="black")
    _style(ax, "Plano de Fase -- " + tipo, "x", "y")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def _plot_time_2d(ax, A, tmax, xmin, xmax, ymin, ymax):
    ax.set_facecolor(C["plot_bg"])
    r0     = min(abs(xmax - xmin), abs(ymax - ymin)) * 0.35
    t_ev   = np.linspace(0, tmax, 800)
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    cols   = plt.cm.cool(np.linspace(0.1, 0.9, 8))
    for th, col in zip(angles, cols):
        x0, y0 = r0 * np.cos(th), r0 * np.sin(th)
        try:
            sol = solve_ivp(lambda t, y: A @ y, (0, tmax), [x0, y0],
                            t_eval=t_ev, method="RK45", rtol=1e-7, atol=1e-9)
            if sol.success:
                xc = np.clip(sol.y[0], xmin * 5, xmax * 5)
                yc = np.clip(sol.y[1], ymin * 5, ymax * 5)
                ax.plot(sol.t, xc, color=col, linewidth=1.4, alpha=0.82, linestyle="-")
                ax.plot(sol.t, yc, color=col, linewidth=1.1, alpha=0.55, linestyle="--")
        except Exception:
            pass
    leg = [Line2D([0], [0], color="white", lw=1.5, ls="-",  label="x(t)"),
           Line2D([0], [0], color="white", lw=1.1, ls="--", label="y(t)")]
    ax.legend(handles=leg, fontsize=7.5, facecolor=C["panel"], labelcolor=C["text"],
              framealpha=0.85, loc="upper right")
    _style(ax, "Evolucion Temporal", "t", "x(t), y(t)")


def _analizar_2d(a, b, c, d, tmax, xmin, xmax, ymin, ymax, fig):
    A     = np.array([[a, b], [c, d]], dtype=float)
    tau   = a + d
    delta = a * d - b * c
    vals, vecs = np.linalg.eig(A)
    tipo, estab = _classify_2d(vals, A)
    fig.clear()
    fig.patch.set_facecolor(C["bg"])
    ax_p = fig.add_subplot(1, 2, 1)
    ax_t = fig.add_subplot(1, 2, 2)
    _plot_phase_2d(ax_p, A, xmin, xmax, ymin, ymax, tmax, tipo)
    _plot_time_2d(ax_t, A, tmax, xmin, xmax, ymin, ymax)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.09, wspace=0.32)
    return A, tau, delta, vals, vecs, tipo, estab


# =====================================================================
#  VENTANA: SISTEMAS LINEALES 2D
# =====================================================================

class SistemaLineal2DWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Sistemas Lineales 2D")
        self.configure(bg=C["bg"])
        self.geometry("1360x880")
        self.minsize(1050, 700)
        self._build()

    def _build(self):
        _s = ttk.Style(self)
        _s.configure("Lin2D.Horizontal.TProgressbar",
                      troughcolor=C["entry_bg"], background=C["accent3"],
                      bordercolor=C["border"], thickness=10)

        # Header
        hdr = tk.Frame(self, bg=C["accent3"], height=48)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="  Sistemas Lineales 2D",
                 bg=C["accent3"], fg="white",
                 font=("Segoe UI", 12, "bold")).pack(side="left", padx=14, pady=10)
        tk.Label(hdr, text="dx/dt = ax + by   |   dy/dt = cx + dy  ",
                 bg=C["accent3"], fg="#bae6fd",
                 font=("Segoe UI", 10, "italic")).pack(side="right", padx=14)

        # Panel de entrada
        inp = tk.Frame(self, bg=C["panel"])
        inp.pack(fill="x", padx=10, pady=(10, 4))

        # Matriz 2x2
        mat_frame = tk.Frame(inp, bg=C["panel"])
        mat_frame.pack(side="left", padx=(14, 24), pady=(10, 10))
        tk.Label(mat_frame, text="Matriz A", bg=C["panel"], fg=C["accent3"],
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=0, columnspan=6, pady=(0, 4))

        def mat_entry(r, c, var, lbl):
            tk.Label(mat_frame, text=lbl, bg=C["panel"], fg=C["text2"],
                     font=("Segoe UI", 9)).grid(row=r + 1, column=c * 2,
                                                padx=(6, 2), sticky="e")
            en = tk.Entry(mat_frame, textvariable=var, width=7,
                          bg=C["entry_bg"], fg=C["accent2"],
                          insertbackground=C["text"], font=("Consolas", 11),
                          relief="flat", bd=4)
            en.grid(row=r + 1, column=c * 2 + 1, padx=(0, 10), ipady=3)
            en.bind("<Return>", lambda ev: self._run())

        self._a = tk.StringVar(value="1")
        self._b = tk.StringVar(value="-2")
        self._c = tk.StringVar(value="1")
        self._d = tk.StringVar(value="-1")
        mat_entry(0, 0, self._a, "a =")
        mat_entry(0, 1, self._b, "b =")
        mat_entry(1, 0, self._c, "c =")
        mat_entry(1, 1, self._d, "d =")

        # Campos de rango
        rng_frame = tk.Frame(inp, bg=C["panel"])
        rng_frame.pack(side="left", padx=(0, 14), pady=(10, 10))

        def field(p, lbl, var, w=7):
            frm = tk.Frame(p, bg=C["panel"])
            frm.pack(anchor="w", pady=2)
            tk.Label(frm, text=lbl, bg=C["panel"], fg=C["text2"],
                     font=("Segoe UI", 9), width=10, anchor="e").pack(side="left", padx=(0, 4))
            en = tk.Entry(frm, textvariable=var, width=w,
                          bg=C["entry_bg"], fg=C["accent2"],
                          insertbackground=C["text"], font=("Consolas", 9),
                          relief="flat", bd=4)
            en.pack(side="left")
            en.bind("<Return>", lambda ev: self._run())

        self._tmax = tk.StringVar(value="10")
        self._xmin = tk.StringVar(value="-4")
        self._xmax = tk.StringVar(value="4")
        self._ymin = tk.StringVar(value="-4")
        self._ymax = tk.StringVar(value="4")
        field(rng_frame, "t maximo:", self._tmax)
        field(rng_frame, "x minimo:", self._xmin)
        field(rng_frame, "x maximo:", self._xmax)
        field(rng_frame, "y minimo:", self._ymin)
        field(rng_frame, "y maximo:", self._ymax)

        # Boton + progressbar
        btn_frame = tk.Frame(inp, bg=C["panel"])
        btn_frame.pack(side="left", padx=(10, 14), pady=10)
        self._btn = tk.Button(btn_frame, text="  ANALIZAR  ",
                               bg=C["accent3"], fg="white",
                               font=("Segoe UI", 11, "bold"), relief="flat",
                               cursor="hand2", pady=8,
                               activebackground="#0891b2", activeforeground="white",
                               command=self._run)
        self._btn.pack(pady=(0, 8))
        self._pb = ttk.Progressbar(btn_frame, mode="indeterminate", length=120,
                                    style="Lin2D.Horizontal.TProgressbar")

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x", padx=10)

        # Panel principal: sidebar resultados + canvas
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=10, pady=(4, 8))

        # Sidebar de resultados (izquierda)
        sb = tk.Frame(main, bg=C["panel"], width=310)
        sb.pack(side="left", fill="y", padx=(0, 6))
        sb.pack_propagate(False)
        tk.Label(sb, text="  RESULTADOS", bg=C["sidebar"], fg="white",
                 font=("Segoe UI", 9, "bold")).pack(fill="x", ipady=5)
        self._rtxt = tk.Text(sb, bg=C["entry_bg"], fg=C["text"],
                              font=("Consolas", 9), relief="flat",
                              wrap="word", state="disabled",
                              selectbackground=C["accent"], padx=8, pady=8)
        self._rtxt.pack(fill="both", expand=True)

        # Canvas matplotlib (derecha)
        pf = tk.Frame(main, bg=C["bg"])
        pf.pack(side="left", fill="both", expand=True)
        self._fig    = plt.Figure(facecolor=C["bg"])
        self._canvas = FigureCanvasTkAgg(self._fig, master=pf)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        tbf = tk.Frame(pf, bg=C["panel"]); tbf.pack(fill="x")
        tb = NavigationToolbar2Tk(self._canvas, tbf)
        tb.config(bg=C["panel"]); tb.update()
        self._run()

    def _run(self):
        try:
            a = float(self._a.get()); b = float(self._b.get())
            c = float(self._c.get()); d = float(self._d.get())
            tmax = float(self._tmax.get())
            xmin = float(self._xmin.get()); xmax = float(self._xmax.get())
            ymin = float(self._ymin.get()); ymax = float(self._ymax.get())
        except ValueError:
            messagebox.showerror("Error", "Todos los valores deben ser numeros.", parent=self)
            return
        self._btn.config(state="disabled", text="  Analizando...")
        self._pb.pack(pady=(0, 4)); self._pb.start(10)

        def _work():
            try:
                result = _analizar_2d(a, b, c, d, tmax, xmin, xmax, ymin, ymax, self._fig)
                self.after(0, lambda: self._finish(result, None))
            except Exception as exc:
                self.after(0, lambda: self._finish(None, exc))
        threading.Thread(target=_work, daemon=True).start()

    def _finish(self, result, error):
        self._pb.stop(); self._pb.pack_forget()
        self._btn.config(state="normal", text="  ANALIZAR  ")
        if error is not None:
            messagebox.showerror("Error", "Error al analizar:\n\n{}".format(error), parent=self)
            return
        self._canvas.draw()
        A, tau, delta, vals, vecs, tipo, estab = result
        lines = []
        lines.append("CLASIFICACION")
        lines.append("=" * 32)
        lines.append("  Tipo:        {}".format(tipo))
        lines.append("  Estabilidad: {}".format(estab))
        lines.append("")
        lines.append("TRAZAS Y DETERMINANTE")
        lines.append("-" * 32)
        lines.append("  Traza  (τ)  = {:+.6f}".format(tau))
        lines.append("  Det    (Δ)  = {:+.6f}".format(delta))
        lines.append("  τ² - 4Δ    = {:+.6f}".format(tau**2 - 4 * delta))
        lines.append("")
        lines.append("VALORES PROPIOS")
        lines.append("-" * 32)
        for i, v in enumerate(vals):
            lines.append("  λ{} = {}".format(i + 1, _fmt_complex(v)))
        lines.append("")
        lines.append("VECTORES PROPIOS")
        lines.append("-" * 32)
        for i, col in enumerate(vecs.T):
            lines.append("  v{}: [ {}".format(i + 1, _fmt_complex(col[0])))
            lines.append("       {} ]".format(_fmt_complex(col[1])))
        lines.append("")
        lines.append("MATRIZ A")
        lines.append("-" * 32)
        lines.append("  [ {:+.4f}  {:+.4f} ]".format(A[0, 0], A[0, 1]))
        lines.append("  [ {:+.4f}  {:+.4f} ]".format(A[1, 0], A[1, 1]))
        txt = "\n".join(lines)
        self._rtxt.config(state="normal")
        self._rtxt.delete("1.0", "end")
        self._rtxt.insert("end", txt)
        self._rtxt.config(state="disabled")


# =====================================================================
#  MENU PRINCIPAL
# =====================================================================

class MainMenu(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulador -- Modelado y Simulacion")
        self.configure(bg=C["bg"])
        self.resizable(False, False)
        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=C["accent"], height=52)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="  Simulador de Sistemas Dinamicos",
                 bg=C["accent"], fg="white",
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=16, pady=10)
        tk.Label(hdr, text="Modelado y Simulacion  *  Ing. Omar J. Caceres  ",
                 bg=C["accent"], fg="#c4b5fd",
                 font=("Segoe UI", 8)).pack(side="right", padx=14)

        body = tk.Frame(self, bg=C["bg"], padx=60, pady=40)
        body.pack()

        tk.Label(body, text="Selecciona el tipo de sistema a analizar:",
                 bg=C["bg"], fg=C["text2"],
                 font=("Segoe UI", 10)).pack(pady=(0, 20))

        # Boton 1: Sistemas 1D Autonomos
        tk.Button(body,
            text="   Sistemas Unidimensionales Autonomos",
            bg=C["panel"], fg=C["text"],
            font=("Segoe UI", 12, "bold"),
            relief="flat", cursor="hand2",
            padx=40, pady=22, width=36,
            activebackground=C["accent"], activeforeground="white",
            command=self._open_1d,
        ).pack(fill="x")
        tk.Frame(body, bg=C["accent"], height=3).pack(fill="x", pady=(0, 4))
        tk.Label(body, text="dx/dt = f(x)   --   equilibrios, estabilidad y diagramas",
                 bg=C["bg"], fg=C["text2"], font=("Segoe UI", 8, "italic")).pack()

        tk.Frame(body, bg=C["bg"], height=18).pack()

        # Boton 2: Bifurcaciones
        tk.Button(body,
            text="   Bifurcaciones",
            bg=C["panel"], fg=C["text"],
            font=("Segoe UI", 12, "bold"),
            relief="flat", cursor="hand2",
            padx=40, pady=22, width=36,
            activebackground=C["sidebar"], activeforeground="white",
            command=self._open_bif,
        ).pack(fill="x")
        tk.Frame(body, bg=C["sidebar"], height=3).pack(fill="x", pady=(0, 4))
        tk.Label(body, text="x_dot = f(x, r)   --   diagrama de bifurcacion, fase y temporal",
                 bg=C["bg"], fg=C["text2"], font=("Segoe UI", 8, "italic")).pack()

        tk.Frame(body, bg=C["bg"], height=18).pack()

        # Boton 3: Sistemas Lineales 2D
        tk.Button(body,
            text="   Sistemas Lineales 2D",
            bg=C["panel"], fg=C["text"],
            font=("Segoe UI", 12, "bold"),
            relief="flat", cursor="hand2",
            padx=40, pady=22, width=36,
            activebackground=C["accent3"], activeforeground="white",
            command=self._open_2d,
        ).pack(fill="x")
        tk.Frame(body, bg=C["accent3"], height=3).pack(fill="x", pady=(0, 4))
        tk.Label(body,
                 text="dx/dt = ax+by, dy/dt = cx+dy   --   eigenvalores, clasificacion y diagramas",
                 bg=C["bg"], fg=C["text2"], font=("Segoe UI", 8, "italic")).pack()

        tk.Frame(body, bg=C["bg"], height=20).pack()
        tk.Label(body, text="*  *  *     Mas tipos de sistemas proximamente     *  *  *",
                 bg=C["bg"], fg="#2d3a55", font=("Segoe UI", 8)).pack(pady=6)

        tk.Label(self, text="Modelado y Simulacion 3.1.025  *  Segunda Edicion 2026",
                 bg=C["bg"], fg="#2d3748", font=("Segoe UI", 7)).pack(pady=(0, 10))

    def _open_1d(self):
        win = AnalysisWindow(self); win.grab_set(); win.focus_force()

    def _open_bif(self):
        win = BifurcationWindow(self); win.grab_set(); win.focus_force()

    def _open_2d(self):
        win = SistemaLineal2DWindow(self); win.grab_set(); win.focus_force()

# =====================================================================
if __name__ == "__main__":
    app = MainMenu()
    app.mainloop()

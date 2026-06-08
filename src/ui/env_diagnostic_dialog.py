"""
Environment Diagnostic Dialog
=============================

A self-help dialog that shows which Python interpreter is running and
which AI dependencies are present.  Diagnoses the very common
"I pip-installed transformers but the app still says it's missing"
situation, which usually means pip and the running Python belong to
different installations.
"""

import customtkinter as ctk

from src.core.model_manager import diagnose_env


class EnvDiagnosticDialog(ctk.CTkToplevel):
    """Read-only diagnostic dialog with a copy-pasteable install command."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("🔧 环境诊断")
        self.geometry("720x500")
        self.minsize(560, 380)
        self.grab_set()

        info = diagnose_env()
        self._info = info
        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        ctk.CTkLabel(
            self, text="🔧 Python 环境诊断",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, padx=20, pady=(18, 10), sticky="w")

        # Body — read-only textbox
        body = ctk.CTkTextbox(self, font=("Consolas", 11), wrap="word")
        body.grid(row=1, column=0, padx=18, pady=(0, 8), sticky="nsew")

        py = self._info["python"]
        packages = self._info["packages"]
        pip_hint = self._info["pip_hint"]

        lines = [
            "═══ 当前运行的 Python ═══",
            f"  路径：{py['executable']}",
            f"  版本：{py['version']}",
            f"  Prefix：{py['prefix']}",
            "",
            "═══ AI 依赖检查 ═══",
        ]
        any_missing = False
        for p in packages:
            name = p["name"]
            if not p.get("installed"):
                lines.append(f"  ❌ {name:18s}  未安装")
                any_missing = True
            elif p.get("error"):
                lines.append(f"  ⚠️  {name:18s}  已安装但导入失败：{p['error']}")
                any_missing = True
            else:
                lines.append(f"  ✅ {name:18s}  v{p['version']}")
                lines.append(f"     来源：{p['origin']}")

        lines.append("")
        if any_missing:
            lines.extend([
                "═══ 修复建议 ═══",
                "",
                "如果你已经用 `pip install transformers` 安装过，但这里仍显示「未安装」，",
                "几乎可以确定是 pip 和当前 Python 不属于同一安装。",
                "请用 **当前 Python** 直接调用 pip，复制下面的命令到命令行运行：",
                "",
                "  " + (pip_hint or '"<python>" -m pip install transformers accelerate huggingface_hub safetensors'),
                "",
                "安装完成后，关闭并重新启动 GaussianHairCube。",
            ])
        else:
            lines.extend([
                "═══ 状态 ═══",
                "",
                "✅ 所有 AI 依赖已正确安装在当前 Python 中。",
                "如果模型下载仍失败，可能是网络问题。请在 Settings → AI 模型 中配置镜像，",
                "或查看日志窗口获取详细错误（Header → 📋 日志）。",
            ])

        body.insert("1.0", "\n".join(lines))
        body.configure(state="disabled")

        # Bottom buttons
        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.grid(row=2, column=0, pady=(4, 16))

        if pip_hint:
            ctk.CTkButton(
                btns, text="📋 复制安装命令",
                command=lambda: self._copy_to_clipboard(pip_hint),
                width=140, height=32,
                fg_color="#1565c0", hover_color="#0d47a1",
            ).pack(side="left", padx=6)

        ctk.CTkButton(
            btns, text="关闭", command=self.destroy,
            width=100, height=32, fg_color="gray40", hover_color="gray30",
        ).pack(side="left", padx=6)

    def _copy_to_clipboard(self, text: str):
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update()
        except Exception:
            pass

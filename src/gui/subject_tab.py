import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
from src.utils.data_utils import load_csv_file
from src.services.subject_analyzer import SubjectAnalyzer
from config.spark_config import spark
from src.utils.async_task import AsyncTaskRunner
from src.gui.components.loading_overlay import LoadingOverlay
class SubjectAnalysisTab:
    """
    Class quản lý giao diện và logic Tab phân tích học phần
    """
    
    def __init__(self, parent_frame):
        self.parent = parent_frame
        
        # Dữ liệu
        # chỉ giữ Spark DataFrame, dùng pandas chỉ khi cần hiển thị
        self.spark_df = None
        self.selected_subjects_codes = []
        self.analysis_results = {}  # key = MaMH
        self.listbox_subjects = None
        self.all_subjects_list = []   # lưu toàn bộ môn để filter
        self.model_path = tk.StringVar(value="models/subject_quality_kmeans_model")
        
        # Tiện ích
        self.task_runner = AsyncTaskRunner(self.parent.winfo_toplevel())
        self.loading = LoadingOverlay(self.parent.winfo_toplevel())

        # Tạo giao diện
        self.create_layout()
    
    def create_layout(self):
        """Tạo bố cục chính"""
        self.paned_window = tk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill="both", expand=True)
        
        self.left_frame = tk.Frame(self.paned_window, bg="#f1f5f9", width=280)
        self.right_frame = tk.Frame(self.paned_window, bg="#f1f5f9")
        
        self.paned_window.add(self.left_frame, minsize=250, width=280)
        self.paned_window.add(self.right_frame)
        
        self.create_left_panel()
        self.create_right_panel()
    
    def create_left_panel(self):
        """Tạo panel điều khiển bên trái"""
        # Container cho nội dung có thể scroll
        content_frame = tk.Frame(self.left_frame, bg="#f1f5f9")
        content_frame.pack(fill="both", expand=True)
        
        # Canvas và Scrollbar
        canvas = tk.Canvas(content_frame, bg="#f1f5f9", highlightthickness=0)
        scrollbar = tk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f1f5f9")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.configure(yscrollcommand=scrollbar.set)

        # Cho phép scroll bằng chuột - Chỉ khi chuột nằm trên panel này
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_mouse(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        def _unbind_mouse(event):
            canvas.unbind_all("<MouseWheel>")
            
        canvas.bind("<Enter>", _bind_mouse)
        canvas.bind("<Leave>", _unbind_mouse)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Đồng bộ chiều rộng của frame với canvas
        def _configure_canvas(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", _configure_canvas)
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Header
        tk.Label(
            scrollable_frame, 
            text="ĐIỀU KHIỂN", 
            bg="#1e293b", fg="white", 
            font=("Segoe UI", 12, "bold"), pady=10
        ).pack(fill="x")
        
        # 1. Load File
        frame_file = tk.LabelFrame(
            scrollable_frame, 
            text="1. Dữ liệu nguồn", 
            font=("Segoe UI", 10, "bold"), 
            bg="#f1f5f9",
            fg="#475569",
            padx=3,
            pady=3
        )
        frame_file.pack(fill="x", padx=5, pady=(5, 3))
        
        tk.Button(
            frame_file, 
            text="Tải CSV", 
            command=self.load_csv, 
            bg="#3b82f6", 
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            cursor="hand2"
        ).pack(fill="x", padx=5, pady=3)
        
        self.lbl_file_status = tk.Label(
            frame_file, 
            text="Chưa có dữ liệu", 
            fg="#94a3b8", 
            bg="#f1f5f9",
            font=("Segoe UI", 9)
        )
        self.lbl_file_status.pack(pady=2)

        # 1.1 Chọn Model (Đưa lên trên để dễ thấy)
        tk.Label(
            frame_file, 
            text="Mô hình huấn luyện:", 
            bg="#f1f5f9",
            fg="#475569",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", padx=5, pady=(5, 0))
        
        model_f = tk.Frame(frame_file, bg="#f1f5f9")
        model_f.pack(fill="x", padx=5, pady=2)
        
        tk.Entry(model_f, textvariable=self.model_path, font=("Segoe UI", 8)).pack(side="left", fill="x", expand=True)
        tk.Button(model_f, text="...", command=self.browse_model, width=3, font=("Segoe UI", 8)).pack(side="left", padx=2)

        
        # 2. Tìm kiếm & Thêm học phần
        frame_search = tk.LabelFrame(
            scrollable_frame, 
            text="2. Chọn học phần", 
            font=("Segoe UI", 10, "bold"), 
            bg="#f1f5f9",
            fg="#475569",
            padx=3,
            pady=3
        )
        frame_search.pack(fill="x", padx=5, pady=3)
        
        tk.Label(
            frame_search, 
            text="Tìm kiếm (Tên/Mã):", 
            bg="#f1f5f9",
            fg="#475569",
            font=("Segoe UI", 9)
        ).pack(anchor="w", padx=5, pady=(3, 0))
        
        self.entry_search = tk.Entry(frame_search)
        self.entry_search.pack(fill="x", padx=5, pady=2)
        self.entry_search.bind("<Return>", lambda e: self.search_subject())
        
        tk.Button(
            frame_search, 
            text="Tìm", 
            command=self.search_subject, 
            bg="#64748b", 
            fg="white",
            font=("Segoe UI", 9),
            relief="flat",
            cursor="hand2"
        ).pack(pady=3)
        
        self.combo_suggestions = ttk.Combobox(frame_search, state="readonly")
        self.combo_suggestions.pack(fill="x", padx=5, pady=3)
        
        tk.Button(
            frame_search, 
            text="⬇ Thêm vào danh sách phân tích", 
            command=self.add_subject_to_list,
            bg="#10b981", 
            fg="white", 
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            cursor="hand2"
        ).pack(fill="x", padx=5, pady=3)
        
        tk.Label(
            frame_search, 
            text="Danh sách các môn sẽ phân tích:", 
            bg="#f1f5f9", 
            fg="#475569",
            font=("Segoe UI", 9, "italic")
        ).pack(anchor="w", padx=5, pady=(3, 0))
        
        list_frame = tk.Frame(frame_search)
        list_frame.pack(fill="x", padx=5, pady=2)
        
        self.listbox_selected = tk.Listbox(list_frame, height=6)
        scrollbar_list = tk.Scrollbar(
            list_frame, 
            orient="vertical", 
            command=self.listbox_selected.yview
        )
        self.listbox_selected.config(yscrollcommand=scrollbar_list.set)
        
        self.listbox_selected.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar_list.pack(side=tk.RIGHT, fill="y")
        
        tk.Button(
            frame_search, 
            text="Xóa danh sách chọn", 
            command=self.clear_selection, 
            bg="#ef4444", 
            fg="white",
            font=("Segoe UI", 9),
            relief="flat",
            cursor="hand2"
        ).pack(pady=3)
        
        # 3. Tùy chọn Phân tích
        frame_opt = tk.LabelFrame(
            scrollable_frame, 
            text="3. Cấu hình báo cáo", 
            font=("Segoe UI", 10, "bold"), 
            bg="#f1f5f9",
            fg="#475569",
            padx=3,
            pady=3
        )
        frame_opt.pack(fill="x", padx=5, pady=3)
        
        self.ck_dokho = tk.BooleanVar(value=True)
        self.ck_chatluong = tk.BooleanVar(value=True)
        self.ck_xuhuong = tk.BooleanVar(value=True)
        
        tk.Checkbutton(
            frame_opt, 
            text="Độ khó học phần", 
            variable=self.ck_dokho, 
            bg="#f1f5f9",
            fg="#1e293b",
            font=("Segoe UI", 9),
            activebackground="#f1f5f9"
        ).pack(anchor="w", pady=1)
        tk.Checkbutton(
            frame_opt, 
            text="Kết quả lớp học phần", 
            variable=self.ck_chatluong, 
            bg="#f1f5f9",
            fg="#1e293b",
            font=("Segoe UI", 9),
            activebackground="#f1f5f9"
        ).pack(anchor="w", pady=1)
        tk.Checkbutton(
            frame_opt, 
            text="Xu hướng học tập", 
            variable=self.ck_xuhuong, 
            bg="#f1f5f9",
            fg="#1e293b",
            font=("Segoe UI", 9),
            activebackground="#f1f5f9"
        ).pack(anchor="w", pady=1)

        # 4. Nút Chạy Phân Tích
        tk.Button(
            scrollable_frame,
            text="TẠO BÁO CÁO PHÂN TÍCH",
            command=self.generate_report,
            bg="#8b5cf6", 
            fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            cursor="hand2",
            pady=12
        ).pack(fill="x", padx=5, pady=15)


        

    def browse_model(self):
        dirname = filedialog.askdirectory(title="Chọn thư mục model",
                                        initialdir="models")
        if dirname:
            # Luôn ưu tiên dùng đường dẫn tuyệt đối để tránh nhầm lẫn
            abs_path = os.path.abspath(dirname)
            self.model_path.set(abs_path)
    
    def create_right_panel(self):
        """Tạo panel hiển thị bên phải"""
        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.pack(fill="both", expand=True)
        
        # TAB 1: Dữ liệu
        self.tab_data = tk.Frame(self.notebook, bg="#f1f5f9")
        self.notebook.add(self.tab_data, text=" Dữ liệu Học phần ")
        
        search_df_frame = tk.Frame(self.tab_data, bg="#e2e8f0", height=40)
        search_df_frame.pack(fill="x")
        
        tk.Label(
            search_df_frame, 
            text="Tìm kiếm trong bảng dữ liệu:", 
            bg="#e2e8f0",
            fg="#475569",
            font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, padx=10, pady=5)
        
        self.entry_filter_df = tk.Entry(search_df_frame, width=40)
        self.entry_filter_df.pack(side=tk.LEFT, padx=5, pady=5)
        self.entry_filter_df.bind("<KeyRelease>", self.filter_dataframe_view)
        
        tree_frame = tk.Frame(self.tab_data)
        tree_frame.pack(fill="both", expand=True)
        
        scroll_y = ttk.Scrollbar(tree_frame)
        scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        self.tree = ttk.Treeview(
            tree_frame, 
            yscrollcommand=scroll_y.set, 
            xscrollcommand=scroll_x.set,
            selectmode="extended"
        )
        
        scroll_y.config(command=self.tree.yview)
        scroll_x.config(command=self.tree.xview)
        
        scroll_y.pack(side=tk.RIGHT, fill="y")
        scroll_x.pack(side=tk.BOTTOM, fill="x")
        self.tree.pack(side=tk.LEFT, fill="both", expand=True)
        
        # TAB 2: Báo cáo
        self.tab_report = tk.Frame(self.notebook, bg="#f1f5f9")
        self.notebook.add(self.tab_report, text="Báo cáo Phân tích ")

        # === Layout chia trái / phải ===
        report_paned = tk.PanedWindow(self.tab_report, orient=tk.HORIZONTAL)
        report_paned.pack(fill="both", expand=True)

        # --- Danh sách môn (bên trái) ---
        left_report = tk.Frame(report_paned, width=260, bg="#f1f5f9")
        report_paned.add(left_report, minsize=220)

        # Header danh sách
        tk.Label(
            left_report, text="DANH SÁCH HỌC PHẦN",
            bg="#1e293b", fg="white",
            font=("Segoe UI", 11, "bold"), pady=8
        ).pack(fill="x")

        # Legend màu
        legend_f = tk.Frame(left_report, bg="#e2e8f0", pady=4)
        legend_f.pack(fill="x")
        for color, label in [
            ("#fd7e7e", "Tiêu cực"),
            ("#fcd143", "Không ổn định"),
            ("#c4b5fd", "Khá"),
            ("#5eecb3", "Xuất sắc"),
        ]:
            row_f = tk.Frame(legend_f, bg="#e2e8f0")
            row_f.pack(side="left", padx=6)
            tk.Label(row_f, text="●", font=("Segoe UI", 12),
                     bg="#e2e8f0", fg=color).pack(side="left")
            tk.Label(row_f, text=label, font=("Segoe UI", 8),
                     bg="#e2e8f0", fg="#475569").pack(side="left")

        # Ô tìm kiếm
        self.entry_search_subject = tk.Entry(
            left_report, font=("Segoe UI", 9),
            relief="flat", bg="#e2e8f0", fg="#1e293b"
        )
        self.entry_search_subject.pack(fill="x", padx=6, pady=(6, 3), ipady=4)
        self.entry_search_subject.bind("<KeyRelease>", self.filter_subject_list)

        # Listbox môn
        lb_frame = tk.Frame(left_report, bg="#f1f5f9")
        lb_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        lb_vsb = ttk.Scrollbar(lb_frame, orient="vertical")
        self.listbox_subjects = tk.Listbox(
            lb_frame,
            font=("Segoe UI", 9),
            bg="#f1f5f9", fg="#1e293b",
            selectbackground="#3b82f6", selectforeground="#ffffff",
            activestyle="none",
            yscrollcommand=lb_vsb.set,
            borderwidth=0, highlightthickness=0
        )
        lb_vsb.config(command=self.listbox_subjects.yview)
        lb_vsb.pack(side="right", fill="y")
        self.listbox_subjects.pack(side="left", fill="both", expand=True)
        self.listbox_subjects.bind("<<ListboxSelect>>", self.on_subject_click)

        # --- Nội dung báo cáo (bên phải) - Canvas scroll giống popup ---
        right_report = tk.Frame(report_paned, bg="#f1f5f9")
        report_paned.add(right_report)

        # Header placeholder (sẽ update khi chọn môn)
        self.report_header_frame = tk.Frame(right_report, bg="#64748b", padx=20, pady=14)
        self.report_header_frame.pack(fill="x")

        self._report_header_icon = tk.Label(
            self.report_header_frame, text="📋",
            font=("Segoe UI Emoji", 28),
            bg="#64748b", fg="#ffffff"
        )
        self._report_header_icon.pack(side="left", padx=(0, 12))

        hdr_text_f = tk.Frame(self.report_header_frame, bg="#64748b")
        hdr_text_f.pack(side="left")
        self._report_header_title = tk.Label(
            hdr_text_f, text="Chọn học phần để xem phân tích",
            font=("Segoe UI", 13, "bold"),
            bg="#64748b", fg="#ffffff"
        )
        self._report_header_title.pack(anchor="w")
        self._report_header_sub = tk.Label(
            hdr_text_f, text="Nhấn vào môn học bên trái",
            font=("Segoe UI", 9),
            bg="#64748b", fg="#e2e8f0"
        )
        self._report_header_sub.pack(anchor="w")

        # Badge bên phải header
        self._report_badge_frame = tk.Frame(self.report_header_frame, bg="#64748b")
        self._report_badge_frame.pack(side="right")
        self._report_badge_label = tk.Label(
            self._report_badge_frame, text="",
            font=("Segoe UI", 10, "bold"),
            bg="#64748b", fg="#ffffff"
        )
        self._report_badge_label.pack(anchor="e")
        self._report_badge_tb = tk.Label(
            self._report_badge_frame, text="",
            font=("Segoe UI", 9),
            bg="#64748b", fg="#e2e8f0"
        )
        self._report_badge_tb.pack(anchor="e")

        # Canvas scroll body
        canvas_outer = tk.Frame(right_report, bg="#f1f5f9")
        canvas_outer.pack(fill="both", expand=True)

        self._report_canvas = tk.Canvas(canvas_outer, bg="#f1f5f9", highlightthickness=0)
        _vsb = ttk.Scrollbar(canvas_outer, orient="vertical", command=self._report_canvas.yview)
        self._report_canvas.configure(yscrollcommand=_vsb.set)
        _vsb.pack(side="right", fill="y")
        self._report_canvas.pack(side="left", fill="both", expand=True)

        self._report_body = tk.Frame(self._report_canvas, bg="#f1f5f9")
        self._report_body_win = self._report_canvas.create_window(
            (0, 0), window=self._report_body, anchor="nw"
        )

        def _resize_body(e):
            self._report_canvas.configure(scrollregion=self._report_canvas.bbox("all"))
            self._report_canvas.itemconfig(self._report_body_win, width=self._report_canvas.winfo_width())
        self._report_body.bind("<Configure>", _resize_body)
        self._report_canvas.bind(
            "<Configure>",
            lambda e: self._report_canvas.itemconfig(self._report_body_win, width=e.width)
        )
        self._report_canvas.bind_all(
            "<MouseWheel>",
            lambda e: self._report_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        )
    
    # ================= LOGIC XỬ LÝ =================
    
    def load_csv(self):
        """Tải file CSV vào Spark DataFrame và hiển thị mẫu lên UI (Chạy ngầm)"""
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        
        self.loading.show()
        self.task_runner.run_task(load_csv_file, args=(spark, path), callback=lambda df: self._on_csv_loaded(df, path))

    def _on_csv_loaded(self, df, path):
        self.spark_df = df.cache()
        # Đếm số dòng chạy ngầm
        self.task_runner.run_task(df.count, callback=lambda count: self._finish_load_csv(count, path, df))

    def _finish_load_csv(self, count, path, df):
        self.loading.hide()
        # reset lựa chọn
        self.selected_subjects_codes = []
        self.listbox_selected.delete(0, tk.END)

        # tạo bảng mẫu
        sample_df = df.limit(2000).toPandas()
        self.show_table(sample_df)
        
        self.lbl_file_status.config(
            text=f"✓ {os.path.basename(path)} ({count} dòng)", 
            fg="#10b981"
        )
        messagebox.showinfo("Thành công", f"Tải dữ liệu thành công! ({count} dòng)")
    
    def show_table(self, df):
        """Hiển thị dữ liệu lên Treeview"""
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(df.columns)
        self.tree["show"] = "headings"
        
        for col_name in df.columns:
            self.tree.heading(col_name, text=col_name)
            self.tree.column(col_name, width=100, anchor="center")
        
        for i, (_, row) in enumerate(df.iterrows()):
            if i > 1000: 
                break 
            self.tree.insert("", "end", values=list(row))
    
    def filter_dataframe_view(self, event=None):
        """Tìm kiếm trong Spark DataFrame và cập nhật bảng hiển thị"""
        # work directly with spark dataframe
        if self.spark_df is None:
            return

        keyword = self.entry_filter_df.get().strip().lower()
        if not keyword:
            df = self.spark_df.limit(2000).toPandas()
            self.show_table(df)
            return

        from pyspark.sql.functions import col

        expr = None
        for c in self.spark_df.columns:
            cond = col(c).cast("string").rlike(f"(?i).*{keyword}.*")
            expr = cond if expr is None else expr | cond

        df_filtered = self.spark_df.filter(expr).limit(2000).toPandas()
        self.show_table(df_filtered)
    
    def search_subject(self):
        """Tìm kiếm học phần trong dữ liệu Spark"""
        if self.spark_df is None:
            return

        keyword = self.entry_search.get().strip().lower()
        if not keyword:
            return

        from pyspark.sql.functions import col

        sdf = (
            self.spark_df.select("MaMH", "TenMH")
            .where(
                (col("TenMH").cast("string").rlike(f"(?i).*{keyword}.*")) |
                (col("MaMH").cast("string").rlike(f"(?i).*{keyword}.*"))
            )
            .distinct()
        )

        pandas_filtered = sdf.limit(1000).toPandas()
        suggestions = [f"{row['MaMH']} - {row['TenMH']}" for _, row in pandas_filtered.iterrows()]
        self.combo_suggestions['values'] = suggestions
        if suggestions:
            self.combo_suggestions.current(0)
    
    def add_subject_to_list(self):
        """Thêm học phần vào danh sách"""
        ma_mh = None
        ten_mh = None
        
        # Thử lấy từ dòng chọn trong Treeview
        if hasattr(self, "tree"):
            selected_items = self.tree.selection()
            if selected_items:
                item_id = selected_items[0]
                values = self.tree.item(item_id, "values")
                
                if values:
                    columns = list(self.tree["columns"])
                    try:
                        idx_ma = columns.index("MaMH")
                        idx_ten = columns.index("TenMH")
                        ma_mh = str(values[idx_ma]).strip()
                        ten_mh = str(values[idx_ten]).strip()
                    except ValueError:
                        pass
        
        # Fallback về Combobox
        if ma_mh is None:
            selected_text = self.combo_suggestions.get()
            if not selected_text:
                messagebox.showwarning(
                    "Chưa chọn môn",
                    "Hãy chọn một học phần trước."
                )
                return
            parts = selected_text.split(" - ", 1)
            ma_mh = parts[0].strip()
            ten_mh = parts[1].strip() if len(parts) > 1 else ""
        
        # Kiểm tra trùng
        if ma_mh in self.selected_subjects_codes:
            messagebox.showwarning("Trùng lặp", "Môn học này đã có trong danh sách!")
            return
        
        # Thêm vào danh sách
        self.selected_subjects_codes.append(ma_mh)
        display_text = f"{ma_mh} - {ten_mh}" if ten_mh else ma_mh
        self.listbox_selected.insert(tk.END, display_text)
    
    def clear_selection(self):
        """Xóa danh sách chọn"""
        self.selected_subjects_codes = []
        self.listbox_selected.delete(0, tk.END)
        if self.spark_df is not None:
            df = self.spark_df.limit(2000).toPandas()
            self.show_table(df)

    # ── Helpers render report (popup-style) ─────────────────────────────────

    def _get_group_info(self, ma_mh):
        """Trả về (priority 0=xấu..3=tốt, badge_bg, badge_fg, icon, label)."""
        result = self.analysis_results.get(ma_mh)
        if not result:
            return (2, "#c4b5fd", "#1e1b4b", "📋", "Ổn định")
        
        # Đã đổi tên cột từ ChatLuongCluster sang ChatLuong ở bước trước
        cl = str(result.get("ChatLuong") or "").strip()
        
        if "Tiêu cực" in cl:
            return (0, "#ef4444", "#ffffff", "⚡", "Tiêu cực")
        if "Không ổn định" in cl:
            return (1, "#f59e0b", "#1c1917", "⚠️", "Không ổn định")
        if "Khá" in cl:
            return (2, "#8b5cf6", "#ffffff", "📈", "Khá")
        if "Xuất sắc" in cl:
            return (3, "#10b981", "#ffffff", "✅", "Xuất sắc")
        
        return (2, "#8b5cf6", "#ffffff", "📈", "Khá")

    def get_subject_color(self, ma_mh):
        p, bg, fg, icon, label = self._get_group_info(ma_mh)
        mapping = {0: "#fd7e7e", 1: "#fcd143", 2: "#c4b5fd", 3: "#5eecb3"}
        return mapping.get(p, "#f1f5f9")

    def _sec_title_card(self, parent, text):
        tk.Label(
            parent, text=text,
            font=("Segoe UI", 10, "bold"),
            bg="#f1f5f9", fg="#94a3b8", anchor="w"
        ).pack(fill="x", padx=16, pady=(12, 4))

    def _info_row(self, parent, label, value, accent_color):
        row = tk.Frame(parent, bg="#e2e8f0")
        row.pack(fill="x", pady=2)
        tk.Label(row, text="▸", font=("Segoe UI", 11, "bold"),
                 bg="#e2e8f0", fg=accent_color
                 ).pack(side="left", padx=(0, 6))
        tk.Label(row, text=label, font=("Segoe UI", 10, "bold"),
                 bg="#e2e8f0", fg="#475569", width=16, anchor="w"
                 ).pack(side="left")
        tk.Label(row, text=str(value), font=("Segoe UI", 10),
                 bg="#e2e8f0", fg="#1e293b", anchor="w"
                 ).pack(side="left", fill="x", expand=True)

    def _render_report_body(self, ma_mh):
        """Vẽ lại nội dung card trong canvas body giống popup."""
        # Xóa body cũ
        for w in self._report_body.winfo_children():
            w.destroy()
        self._report_canvas.yview_moveto(0)

        row = self.analysis_results.get(ma_mh)
        if not row:
            return

        priority, badge_bg, badge_fg, icon, group_label = self._get_group_info(ma_mh)
        ten_mh = str(row.get("TenMH") or ma_mh)
        tb     = row.get("TB", "-")
        fp     = row.get("F%", "-")
        dokho  = str(row.get("DoKho") or "-")
        chat   = str(row.get("ChatLuong") or "-")
        xu     = str(row.get("XuHuong") or "-")

        # Update header
        self.report_header_frame.config(bg=badge_bg)
        self._report_header_icon.config(bg=badge_bg, fg=badge_fg, text=icon)
        for w in self.report_header_frame.winfo_children():
            if isinstance(w, tk.Frame):
                w.config(bg=badge_bg)
                for c in w.winfo_children():
                    if isinstance(c, tk.Label):
                        c.config(bg=badge_bg)
        self._report_header_title.config(text=ten_mh, bg=badge_bg, fg=badge_fg)
        self._report_header_sub.config(text=f"Mã môn: {ma_mh}", bg=badge_bg, fg=badge_fg)
        self._report_badge_label.config(text=group_label, bg=badge_bg, fg=badge_fg)
        self._report_badge_frame.config(bg=badge_bg)
        self._report_badge_tb.config(
            text=f"TB: {tb}  |  Rớt: {fp}%",
            bg=badge_bg, fg=badge_fg
        )

        P = {"padx": 16, "pady": (0, 8)}

        # ── 1. Chỉ số học tập ────────────────────────────────────────────────
        self._sec_title_card(self._report_body, "📊  Chỉ Số Học Tập")
        cards_f = tk.Frame(self._report_body, bg="#f1f5f9")
        cards_f.pack(fill="x", **P)

        # Tính toán đánh giá dựa trên giá trị thực tế (không dùng nhãn cụm)
        try:
            tb_val = float(tb)
            if tb_val >= 8.0: tb_eval = "Tốt / Giỏi"
            elif tb_val >= 6.5: tb_eval = "Khá"
            elif tb_val >= 5.0: tb_eval = "Trung bình"
            else: tb_eval = "Thấp / Kém"
            
            tb_color = "#10b981" if tb_val >= 7.5 else ("#f59e0b" if tb_val >= 6.0 else "#ef4444")
            tb_pct = tb_val / 10.0
        except:
            tb_eval, tb_color, tb_pct = "-", "#64748b", 0.0

        try:
            fp_val = float(fp)
            if fp_val == 0: fp_eval = "Rất thấp"
            elif fp_val <= 10: fp_eval = "Thấp"
            elif fp_val <= 20: fp_eval = "Trung bình"
            else: fp_eval = "Cao / Cảnh báo"
            
            fp_color = "#10b981" if fp_val <= 5 else ("#f59e0b" if fp_val <= 15 else "#ef4444")
            fp_pct = max(0.0, 1.0 - fp_val / 50.0)
        except:
            fp_eval, fp_color, fp_pct = "-", "#64748b", 0.0

        for title, val, subtitle, color, pct in [
            ("Điểm TB",    f"{tb}",  tb_eval,   tb_color, tb_pct),
            ("Tỷ lệ rớt",  f"{fp}%", fp_eval,   fp_color, fp_pct),
        ]:
            card = tk.Frame(cards_f, bg="#e2e8f0", padx=10, pady=10)
            card.pack(side="left", expand=True, fill="x", padx=4)
            tk.Label(card, text=title, font=("Segoe UI", 8, "bold"),
                     bg="#e2e8f0", fg="#64748b").pack(anchor="w")
            tk.Label(card, text=val, font=("Segoe UI", 20, "bold"),
                     bg="#e2e8f0", fg=color).pack(anchor="w")
            tk.Label(card, text=subtitle, font=("Segoe UI", 8),
                     bg="#e2e8f0", fg=color).pack(anchor="w")
            bar_bg = tk.Frame(card, bg="#374151", height=5)
            bar_bg.pack(fill="x", pady=(6, 2))
            bar_bg.pack_propagate(False)
            pct = max(0.0, min(1.0, pct))
            if pct > 0:
                tk.Frame(bar_bg, bg=color, height=5).place(
                    relx=0, rely=0, relwidth=pct, relheight=1
                )

        # ── 2. Đánh giá độ khó ──────────────────────────────────────────────
        self._sec_title_card(self._report_body, "🔍  Đánh Giá Độ Khó")
        sec2 = tk.Frame(self._report_body, bg="#e2e8f0", padx=14, pady=12)
        sec2.pack(fill="x", padx=16, pady=(0, 8))
        self._info_row(sec2, "Mức độ điểm:", tb_eval, badge_bg)
        self._info_row(sec2, "Độ khó học phần:", dokho, badge_bg)
        self._info_row(sec2, "Tỷ lệ rớt thực tế:", fp_eval, badge_bg)

        # ── 3. Chất lượng giảng dạy ─────────────────────────────────────────
        self._sec_title_card(self._report_body, "📋  Kết quả lớp Học phần")
        sec3 = tk.Frame(self._report_body, bg="#e2e8f0", padx=14, pady=12)
        sec3.pack(fill="x", padx=16, pady=(0, 8))
        self._info_row(sec3, "Phân loại cụm:", group_label, badge_bg)
        self._info_row(sec3, "Đánh giá chung:", chat, badge_bg)

        # ── 4. Xu hướng học tập ─────────────────────────────────────────────
        self._sec_title_card(self._report_body, "📈  Xu Hướng Học Tập")
        sec4 = tk.Frame(self._report_body, bg="#e2e8f0", padx=14, pady=12)
        sec4.pack(fill="x", padx=16, pady=(0, 16))
        self._info_row(sec4, "Nhận định:", xu, badge_bg)

    def generate_report(self):
        """Tạo báo cáo phân tích – Chạy ngầm tránh treo UI"""
        if self.spark_df is None:
            messagebox.showwarning("Cảnh báo", "Chưa có dữ liệu nguồn!")
            return

        model_p = self.model_path.get()
        self.loading.show()
        
        # Chạy phân tích trong background
        self.task_runner.run_task(
            SubjectAnalyzer.analyze_all_subjects, 
            args=(self.spark_df, model_p),
            callback=self._on_analysis_background_done
        )

    def _on_analysis_background_done(self, result_sdf):
        # Sau khi có SDF, collect kết quả cũng chạy ngầm
        self.task_runner.run_task(result_sdf.collect, callback=self._on_collect_done)

    def _on_collect_done(self, results):
        self.loading.hide()
        try:
            if not results:
                messagebox.showwarning("Cảnh báo", "Không có dữ liệu học phần để phân tích!")
                return

            # Lưu kết quả
            self.analysis_results = {row["MaMH"]: row.asDict() for row in results}

            # Convert results sang list dict luôn
            results = [self.analysis_results[k] for k in self.analysis_results]

            # Chuyển sang tab báo cáo
            self.notebook.select(self.tab_report)

            # Xóa body report
            for w in self._report_body.winfo_children():
                w.destroy()

            # Reset header về trạng thái mặc định
            self.report_header_frame.config(bg="#64748b")
            self._report_header_icon.config(bg="#64748b", fg="#ffffff", text="📋")
            self._report_header_title.config(
                text="Chọn học phần để xem phân tích",
                bg="#64748b", fg="#ffffff"
            )
            self._report_header_sub.config(
                text=f"Tổng: {len(results)} học phần – Nhấn vào môn bên trái",
                bg="#64748b", fg="#e2e8f0"
            )
            self._report_badge_label.config(text="", bg="#64748b")
            self._report_badge_frame.config(bg="#64748b")
            self._report_badge_tb.config(text="", bg="#64748b")

            # Sort theo priority: 0(đỏ) → 1(vàng) → 2(tím) → 3(xanh)
            def _sort_key(row):
                p, *_ = self._get_group_info(row["MaMH"])
                return (p, str(row.get("TenMH") or ""))

            sorted_results = sorted(results, key=_sort_key)

            # Group headers
            GROUP_LABELS = {
                0: ("🔴  Tiêu Cực – Cần Lưu Ý",    "#ef4444", "#ffffff"),
                1: ("🟡  Không Ổn Định",              "#f59e0b", "#1c1917"),
                2: ("🟣  Ổn Định",                    "#8b5cf6", "#ffffff"),
                3: ("🟢  Xuất Sắc – Kết Quả Tốt",     "#10b981", "#ffffff"),
            }
            current_group = None

            self.all_subjects_list = []
            self.listbox_subjects.delete(0, tk.END)

            for row in sorted_results:
                ma_mh = row["MaMH"]
                p, *_ = self._get_group_info(ma_mh)

                # Vẽ group header nếu sang nhóm mới
                if p != current_group:
                    current_group = p
                    glabel, gbg, gfg = GROUP_LABELS[p]
                    self.listbox_subjects.insert(tk.END, f"  {glabel}")
                    idx = self.listbox_subjects.size() - 1
                    self.listbox_subjects.itemconfigure(
                        idx, bg=gbg, fg=gfg,
                        selectbackground=gbg, selectforeground=gfg
                    )
                    self.all_subjects_list.append(None)

                ten_mh = str(row.get("TenMH") or "")
                item_text = f"  {ma_mh} – {ten_mh}"
                color = self.get_subject_color(ma_mh)
                self.listbox_subjects.insert(tk.END, item_text)
                idx = self.listbox_subjects.size() - 1
                self.listbox_subjects.itemconfigure(idx, bg=color, fg="#1e293b")
                self.all_subjects_list.append(ma_mh)

            messagebox.showinfo("Hoàn tất", "Đã tạo báo cáo. Chọn môn để xem chi tiết!")
        except Exception as e:
            messagebox.showerror("Lỗi phân tích", str(e))

    def on_subject_click(self, event):
        if not self.analysis_results:
            return

        selection = self.listbox_subjects.curselection()
        if not selection:
            return

        idx = selection[0]
        # Bỏ qua nếu click vào group header (all_subjects_list[idx] == None)
        if idx >= len(self.all_subjects_list) or self.all_subjects_list[idx] is None:
            return

        ma_mh = self.all_subjects_list[idx]
        if not ma_mh or ma_mh not in self.analysis_results:
            return

        self._render_report_body(ma_mh)

    def filter_subject_list(self, event=None):
        if not self.analysis_results:
            return

        keyword = self.entry_search_subject.get().strip().lower()
        self.listbox_subjects.delete(0, tk.END)

        # Khi filter: hiển thị phẳng không có group header
        GROUP_LABELS = {
            0: ("🔴  Tiêu Cực – Cần Lưu Ý",    "#ef4444", "#ffffff"),
            1: ("🟡  Không Ổn Định",              "#f59e0b", "#1c1917"),
            2: ("🟣  Bình Thường – Ổn Định",      "#8b5cf6", "#ffffff"),
            3: ("🟢  Tích Cực – Kết Quả Tốt",     "#10b981", "#ffffff"),
        }

        # Rebuild từ analysis_results đã sort
        def _sort_key(ma):
            p, *_ = self._get_group_info(ma)
            return (p, ma)

        sorted_codes = sorted(self.analysis_results.keys(), key=_sort_key)
        new_list = []
        current_group = None

        for ma_mh in sorted_codes:
            row = self.analysis_results[ma_mh]
            ten = str(row.get("TenMH") or "")
            item_text = f"  {ma_mh} – {ten}"

            if keyword and keyword not in item_text.lower():
                continue

            p, *_ = self._get_group_info(ma_mh)
            if not keyword and p != current_group:
                current_group = p
                glabel, gbg, gfg = GROUP_LABELS[p]
                self.listbox_subjects.insert(tk.END, f"  {glabel}")
                idx = self.listbox_subjects.size() - 1
                self.listbox_subjects.itemconfigure(idx, bg=gbg, fg=gfg,
                    selectbackground=gbg, selectforeground=gfg)
                new_list.append(None)

            color = self.get_subject_color(ma_mh)
            self.listbox_subjects.insert(tk.END, item_text)
            idx = self.listbox_subjects.size() - 1
            self.listbox_subjects.itemconfigure(idx, bg=color, fg="#1e293b")
            new_list.append(ma_mh)

        self.all_subjects_list = new_list
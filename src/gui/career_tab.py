"""
Tab đánh giá mức độ sẵn sàng nghề nghiệp
Thiết kế cao cấp, đồng bộ với Phân tích học phần và Dự đoán học tập.
Hỗ trợ xử lý chạy ngầm và cập nhật dữ liệu trực tuyến.
"""
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import pandas as pd

from src.utils.data_utils import load_csv_file
from src.services.career_analyzer import CareerAnalyzerSpark as CareerAnalyzer
from src.utils.async_task import AsyncTaskRunner
from src.gui.components.loading_overlay import LoadingOverlay
from config.spark_config import spark

class CareerAnalysisTab:
    """
    Tab quản lý Đánh giá tính sẵn sàng nghề nghiệp
    """

    def __init__(self, parent_frame):
        self.parent = parent_frame
        
        # Dữ liệu
        self.spark_df = None
        self.df_career_result = None
        self.market_data = []
        self.important_subjects = []
        
        # Tiện ích
        self.task_runner = AsyncTaskRunner(self.parent.winfo_toplevel())
        self.loading = LoadingOverlay(self.parent.winfo_toplevel())
        self.model_path = tk.StringVar(value="models/career_readiness_kmeans_model")
        
        # UI
        self.create_layout()
        
        # Tự động tải dữ liệu thị trường khi khởi tạo
        self.initial_load_market()

    def create_layout(self):
        """Tạo bố cục chính theo phong cách hiện đại"""
        self.paned_window = tk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill="both", expand=True)
        
        self.left_frame = tk.Frame(self.paned_window, bg="#f1f5f9", width=280)
        self.right_frame = tk.Frame(self.paned_window, bg="#f1f5f9")
        
        self.paned_window.add(self.left_frame, minsize=250, width=280)
        self.paned_window.add(self.right_frame)
        
        self.create_left_panel()
        self.create_right_panel()

    def create_left_panel(self):
        """Tạo panel điều khiển bên trái (Sidebar)"""
        content_frame = tk.Frame(self.left_frame, bg="#f1f5f9")
        content_frame.pack(fill="both", expand=True)
        
        canvas = tk.Canvas(content_frame, bg="#f1f5f9", highlightthickness=0)
        scrollbar = tk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f1f5f9")
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=260)

        # Header
        tk.Label(
            scrollable_frame, text="ĐIỀU KHIỂN", 
            bg="#1e293b", fg="white", font=("Segoe UI", 12, "bold"), pady=10
        ).pack(fill="x")

        # === LEGEND (BỔ SUNG ĐỂ ĐỒNG BỘ) ===
        legend_group = tk.LabelFrame(scrollable_frame, text="Chú giải mức độ", 
                                    font=("Segoe UI", 9, "bold"), bg="#f1f5f9", fg="#475569", padx=5, pady=5)
        legend_group.pack(fill="x", padx=10, pady=5)
        
        for color, label in [
            ("#fd7e7e", "Chưa sẵn sàng"),
            ("#fcd143", "Cần cải thiện"),
            ("#c4b5fd", "Sẵn sàng"),
            ("#5eecb3", "Rất sẵn sàng"),
        ]:
            row_f = tk.Frame(legend_group, bg="#f1f5f9")
            row_f.pack(fill="x", pady=1)
            tk.Label(row_f, text="●", font=("Segoe UI", 12), bg="#f1f5f9", fg=color).pack(side="left")
            tk.Label(row_f, text=label, font=("Segoe UI", 8), bg="#f1f5f9", fg="#1e293b").pack(side="left", padx=5)

        # 1. Quản lý Dữ liệu
        group_data = tk.LabelFrame(scrollable_frame, text="1. Dữ liệu & Thị trường", 
                                  font=("Segoe UI", 10, "bold"), bg="#f1f5f9", fg="#475569", padx=10, pady=10)
        group_data.pack(fill="x", padx=10, pady=5)

        tk.Button(
            group_data, text="📂 Tải CSV Sinh viên", command=self.load_student_csv,
            bg="#3b82f6", fg="white", font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2"
        ).pack(fill="x", pady=3)

        tk.Button(
            group_data, text="🌐 Cập nhật từ TopDev", command=self.sync_market_data,
            bg="#10b981", fg="white", font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2"
        ).pack(fill="x", pady=3)

        self.lbl_status = tk.Label(group_data, text="Chưa có dữ liệu SV", fg="#94a3b8", bg="#f1f5f9", font=("Segoe UI", 9))
        self.lbl_status.pack(pady=2)

        # 2. Cấu hình Model
        group_model = tk.LabelFrame(scrollable_frame, text="2. Mô hình dự báo", 
                                   font=("Segoe UI", 10, "bold"), bg="#f1f5f9", fg="#475569", padx=10, pady=10)
        group_model.pack(fill="x", padx=10, pady=5)
        
        tk.Entry(group_model, textvariable=self.model_path, font=("Segoe UI", 8)).pack(fill="x", pady=2)
        tk.Button(group_model, text="Chọn thư mục Model...", command=self.browse_model, font=("Segoe UI", 8)).pack(fill="x")

        # 3. Tìm kiếm
        group_search = tk.LabelFrame(scrollable_frame, text="3. Tra cứu", 
                                    font=("Segoe UI", 10, "bold"), bg="#f1f5f9", fg="#475569", padx=10, pady=10)
        group_search.pack(fill="x", padx=10, pady=5)
        
        self.entry_search_sv = tk.Entry(group_search)
        self.entry_search_sv.pack(fill="x", pady=2)
        tk.Label(group_search, text="(Mã SV hoặc Họ tên)", font=("Segoe UI", 8, "italic"), bg="#f1f5f9", fg="#94a3b8").pack()

        # Nút Chạy Phân Tích
        tk.Button(
            scrollable_frame, text="🚀 CHẠY ĐÁNH GIÁ SẴN SÀNG", command=self.analyze_career,
            bg="#8b5cf6", fg="white", font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2", pady=12
        ).pack(fill="x", padx=10, pady=20)

    def create_right_panel(self):
        """Tạo panel hiển thị bên phải (Notebook)"""
        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.pack(fill="both", expand=True)
        
        # TAB 1: Thị trường lao động (Ưu tiên hiển thị)
        self.tab_market = tk.Frame(self.notebook, bg="#f1f5f9")
        self.notebook.add(self.tab_market, text=" 📊 Thị trường lao động ")
        
        self.txt_market = tk.Text(self.tab_market, font=("Consolas", 10), wrap="word", bg="white", relief="flat", padx=15, pady=15)
        self.txt_market.pack(fill="both", expand=True, padx=10, pady=10)

        # TAB 2: Danh sách sinh viên
        self.tab_data = tk.Frame(self.notebook, bg="#f1f5f9")
        self.notebook.add(self.tab_data, text=" 👥 Dữ liệu sinh viên ")
        
        tree_frame = tk.Frame(self.tab_data)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        scroll_y = ttk.Scrollbar(tree_frame)
        scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal")
        self.tree = ttk.Treeview(tree_frame, yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set, selectmode="extended")
        scroll_y.config(command=self.tree.yview)
        scroll_x.config(command=self.tree.xview)
        
        scroll_y.pack(side=tk.RIGHT, fill="y")
        scroll_x.pack(side=tk.BOTTOM, fill="x")
        self.tree.pack(side=tk.LEFT, fill="both", expand=True)

        # TAB 3: Báo cáo sẵn sàng
        self.tab_report = tk.Frame(self.notebook, bg="#f1f5f9")
        self.notebook.add(self.tab_report, text=" 📋 Báo cáo sẵn sàng ")
        
        self.txt_report = tk.Text(self.tab_report, font=("Consolas", 10), wrap="word", bg="white", relief="flat", padx=15, pady=15)
        self.txt_report.pack(fill="both", expand=True, padx=10, pady=10)

    # ================= LOGIC XỬ LÝ =================

    def initial_load_market(self):
        """Tải dữ liệu từ file csv có sẵn khi khởi tạo"""
        self.task_runner.run_task(CareerAnalyzer.load_real_jobs, callback=self._on_market_loaded)

    def sync_market_data(self):
        """Đồng bộ từ TopDev API trực tiếp"""
        self.loading.show()
        self.task_runner.run_task(CareerAnalyzer.fetch_topdev_data, callback=self._on_sync_complete)

    def _on_sync_complete(self, result):
        self.loading.hide()
        if result:
            self.market_data = result
            # Lưu lại vào file CSV để offline dùng được
            self._save_market_to_csv(result)
            self.show_market_report(result, self.important_subjects)
            messagebox.showinfo("Thành công", f"Đã cập nhật {len(result)} chuyên ngành từ TopDev!")
        else:
            messagebox.showerror("Lỗi", "Không thể kết nối tới API TopDev. Vui lòng kiểm tra internet.")

    def _on_market_loaded(self, result):
        self.market_data = result
        self.show_market_report(result, self.important_subjects)

    def _save_market_to_csv(self, data):
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "input", "real_jobs.csv")
        try:
            import csv
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=["Category", "Role", "JobCount"])
                writer.writeheader()
                writer.writerows(data)
        except Exception as e:
            print(f"Lỗi lưu CSV: {e}")

    def load_student_csv(self):
        """Tải file CSV sinh viên chạy ngầm"""
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path: return
        
        self.loading.show()
        self.task_runner.run_task(load_csv_file, args=(spark, path), callback=lambda df: self._on_students_loaded(df, path))

    def _on_students_loaded(self, df, path):
        self.spark_df = df
        # Đếm số lượng chạy ngầm
        self.task_runner.run_task(df.count, callback=lambda count: self._finish_load_students(count, path, df))

    def _finish_load_students(self, count, path, df):
        self.loading.hide()
        self.lbl_status.config(text=f"✓ {os.path.basename(path)} ({count} SV)", fg="#10b981")
        # Hiển thị mẫu
        sample = df.limit(1000).toPandas()
        self.show_table(sample)
        messagebox.showinfo("Thành công", f"Đã tải {count} sinh viên.")

    def analyze_career(self):
        """Chạy pipeline ML & Matching chạy ngầm"""
        if self.spark_df is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng tải dữ liệu CSV sinh viên trước.")
            return
            
        keyword = self.entry_search_sv.get().strip()
        model_p = self.model_path.get()
        
        self.loading.show()
        self.task_runner.run_task(
            CareerAnalyzer.analyze_students, 
            args=(self.spark_df, keyword, None, model_p),
            callback=self._on_analysis_complete
        )

    def _on_analysis_complete(self, results):
        self.loading.hide()
        if results is None:
            messagebox.showinfo("Kết quả", "Không tìm thấy sinh viên phù hợp.")
            return
            
        pdf, real_jobs, important_subjects = results
        self.df_career_result = pdf
        self.market_data = real_jobs
        self.important_subjects = important_subjects
        
        self.show_table(pdf)
        self.show_market_report(real_jobs, important_subjects)
        self.show_readiness_report(pdf)
        
        self.notebook.select(self.tab_report)
        messagebox.showinfo("Hoàn tất", f"Đã phân tích {len(pdf)} sinh viên.")

    def show_table(self, df):
        """Hiển thị bảng với màu sắc theo nhóm sẵn sàng"""
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(df.columns)
        self.tree["show"] = "headings"
        
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="center")

        # Tags màu
        self.tree.tag_configure("high", background="#5eecb3", foreground="black")
        self.tree.tag_configure("med", background="#c4b5fd", foreground="black")
        self.tree.tag_configure("low", background="#fcd143", foreground="black")
        self.tree.tag_configure("danger", background="#fd7e7e", foreground="black")

        for _, row in df.iterrows():
            tag = ""
            nhom = str(row.get("nhom_san_sang", "")).lower()
            if "rất sẵn sàng" in nhom: tag = "high"
            elif "sẵn sàng" in nhom: tag = "med"
            elif "cần cải thiện" in nhom: tag = "low"
            elif "chưa sẵn sàng" in nhom: tag = "danger"
            
            self.tree.insert("", "end", values=list(row), tags=(tag,))

    def show_market_report(self, real_jobs, important_subjects):
        self.txt_market.delete(1.0, tk.END)
        self.txt_market.insert(tk.END, "📊 THỐNG KÊ THỊ TRƯỜNG LAO ĐỘNG IT (TOPDEV MACRO)\n", "header")
        self.txt_market.insert(tk.END, "="*60 + "\n\n")
        
        if not real_jobs:
            self.txt_market.insert(tk.END, "Chưa có dữ liệu thị trường. Hãy nhấn 'Cập nhật từ TopDev'.\n")
            return

        total_it = sum(int(j.get("JobCount", 0)) for j in real_jobs if j.get("Category") == "IT")
        self.txt_market.insert(tk.END, f"Tổng số việc làm IT đang tuyển: {total_it} jobs\n\n")
        
        sorted_jobs = sorted([(j["Role"], int(j["JobCount"])) for j in real_jobs], key=lambda x: x[1], reverse=True)
        for role, count in sorted_jobs:
            pct = (count / max(total_it, 1)) * 100
            bar = "█" * int(pct / 2)
            self.txt_market.insert(tk.END, f"{role[:30]:<30}: {count:>4} jobs ({pct:>4.1f}%) {bar}\n")

        self.txt_market.tag_configure("header", font=("Segoe UI", 12, "bold"), foreground="#1e293b")

    def show_readiness_report(self, result_df):
        self.txt_report.delete(1.0, tk.END)
        self.txt_report.insert(tk.END, f"BÁO CÁO TỔNG QUAN ({len(result_df)} SINH VIÊN)\n\n", "header")
        
        if "nhom_san_sang" in result_df.columns:
            counts = result_df["nhom_san_sang"].value_counts()
            for group, count in counts.items():
                pct = count / len(result_df) * 100
                self.txt_report.insert(tk.END, f"• {group}: {count} SV ({pct:.1f}%)\n")
        
        self.txt_report.insert(tk.END, "\n" + "="*60 + "\n")
        self.txt_report.insert(tk.END, "LỘ TRÌNH ĐỀ XUẤT CHI TIẾT\n")
        self.txt_report.insert(tk.END, "="*60 + "\n\n")

        for idx, row in result_df.iterrows():
            if idx >= 20: break # Giới hạn hiển thị
            self.txt_report.insert(tk.END, f"[{idx+1}] {row['ma_sv']} - {row['ho_ten']}\n")
            self.txt_report.insert(tk.END, f"    GPA: {row['gpa_tong']} | TOEIC: {row['TOEIC']} | Nhóm: {row['nhom_san_sang']}\n")
            self.txt_report.insert(tk.END, f"    Phân tích: {row.get('top_matched_jobs', '')}\n\n")

    def browse_model(self):
        dirname = filedialog.askdirectory(title="Chọn thư mục model", initialdir="models")
        if dirname:
            self.model_path.set(os.path.relpath(dirname, os.getcwd()) if "models" in dirname else dirname)

    def export_csv(self):
        # implement as before or keep simple
        pass

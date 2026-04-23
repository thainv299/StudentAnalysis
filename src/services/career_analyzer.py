"""
Module đánh giá mức độ sẵn sàng nghề nghiệp và Gợi ý việc làm (Job Matcher)
1. Đánh giá Sẵn sàng: KMeans (Phân cụm) -> Decision Tree (Tìm môn học Feature Importance)
2. Job Matcher: Content-Based Recommendation dùng Cosine Similarity
"""
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import RandomForestClassifier
from src.utils.data_utils import convert_to_4_scale
import os
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class CareerAnalyzerSpark:

    CAREER_FIELDS = {
        "AI / Data Science": {
            "subjects": ["17234", "17221", "17226", "17233"],
            "skills": ["python", "machine learning", "data", "sql"]
        },
        "Web / Mobile Dev": {
            "subjects": ["17340", "17423", "17314", "17523", "17335"],
            "skills": ["javascript", "react", "html", "css", "node", "php"]
        },
        "Network / Security": {
            "subjects": ["17506", "17212", "17332"],
            "skills": ["network", "security", "aws", "docker"]
        },
        "Software Engineering": {
            "subjects": ["17427", "17434", "17426", "17432", "17236"],
            "skills": ["java", "c#", "c++", "oop", "sql", "git"]
        },
        "Core / Foundation": {
            "subjects": ["17206", "17232", "17302", "17301", "17303", "17104"],
            "skills": ["c", "linux", "system"]
        },
    }

    # ==========================
    # 1. LOAD THỊ TRƯỜNG & JOBS
    # ==========================
    @staticmethod
    def load_real_jobs(csv_path=None):
        if csv_path is None:
            csv_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data", "input", "real_jobs.csv"
            )

        if not os.path.exists(csv_path):
            return []
        
        jobs = []
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    jobs.append(row)
        except Exception:
            pass
        return jobs

    @staticmethod
    def fetch_topdev_data():
        """Lấy dữ liệu macro từ TopDev API (giống crawl_market_data.py)"""
        API_URL = "https://api.topdev.vn/td/v2/job-categories/job-category-with-all-type?locale=vi_VN"
        import requests
        try:
            response = requests.get(API_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            categories = data.get("data", [])
            
            job_stats = []
            for cat in categories:
                cat_name = cat.get("name", "Unknown")
                if cat_name != "IT": continue
                    
                roles = cat.get("roles", [])
                if not roles:
                    job_stats.append({
                        "Category": cat_name, "Role": cat_name, 
                        "JobCount": cat.get("job_count", 0)
                    })
                else:
                    for role in roles:
                        job_stats.append({
                            "Category": cat_name,
                            "Role": role.get("name", "Unknown"),
                            "JobCount": role.get("job_count", 0)
                        })
            return job_stats
        except Exception as e:
            print(f"Lỗi API: {e}")
            return None

    # ==========================
    # 2. XỬ LÝ SINH VIÊN
    # ==========================
    @staticmethod
    def compute_field_gpa(df):
        all_subject_cols = []
        for field_name, field_info in CareerAnalyzerSpark.CAREER_FIELDS.items():
            subjects = field_info["subjects"]
            cols_4 = []
            for s in subjects:
                if s in df.columns:
                    col_name_4 = f"_{s}_4"
                    df = df.withColumn(col_name_4, convert_to_4_scale(s))
                    cols_4.append(col_name_4)
                    all_subject_cols.append(col_name_4)
            col_gpa = field_name.replace(" ", "_").replace("/", "_")
            if cols_4:
                avg_expr = sum([F.col(c) for c in cols_4]) / F.lit(len(cols_4))
                df = df.withColumn(f"gpa_{col_gpa}", F.round(avg_expr, 2))
            else:
                df = df.withColumn(f"gpa_{col_gpa}", F.lit(0.0))

        if all_subject_cols:
            total_avg = sum([F.col(c) for c in all_subject_cols]) / F.lit(len(all_subject_cols))
            df = df.withColumn("gpa_tong", F.round(total_avg, 2))
        else:
            df = df.withColumn("gpa_tong", F.lit(0.0))
        return df

    # ==========================
    # 5. MACRO CAREER COMPETITIVENESS
    # ==========================
    @staticmethod
    def compute_career_competitiveness(df_students, jobs_stats):
        """Xác định nhánh nghề nghiệp mạnh nhất của sinh viên và tính Hệ số Cạnh Tranh từ Macro Market Data."""
        from pyspark.sql import functions as F
        from pyspark.sql.types import StringType

        if not jobs_stats or df_students.count() == 0:
            return df_students.withColumn("top_matched_jobs", F.lit("Không tìm thấy C.v Mới"))
            
        # Ánh xạ từ Tên Nhóm Ngành của Sinh Viên sang Category/Role của TopDev
        # TopDev có ngành "IT" chứa các role con. Ta gom chung.
        # Để đơn giản, ta tính tổng số lượng việc làm IT trên thị trường.
        total_it_jobs = 0
        field_jobs_map = {}
        
        for job in jobs_stats:
            cat = job.get("Category", "")
            role = job.get("Role", "")
            try:
                count = int(job.get("JobCount", 0))
            except:
                count = 0
                
            if cat == "IT":
                total_it_jobs += count
                # Map role names to our academic fields
                role_lower = role.lower()
                if "software" in role_lower or "fullstack" in role_lower or "backend" in role_lower:
                    field_jobs_map["Software Engineering"] = field_jobs_map.get("Software Engineering", 0) + count
                elif "data" in role_lower or "machine learning" in role_lower or "ai" in role_lower:
                    field_jobs_map["AI / Data Science"] = field_jobs_map.get("AI / Data Science", 0) + count
                elif "web" in role_lower or "mobile" in role_lower or "frontend" in role_lower:
                    field_jobs_map["Web / Mobile Dev"] = field_jobs_map.get("Web / Mobile Dev", 0) + count
                elif "network" in role_lower or "security" in role_lower or "system" in role_lower or "devops" in role_lower:
                    field_jobs_map["Network / Security"] = field_jobs_map.get("Network / Security", 0) + count
                else:
                    field_jobs_map["Core / Foundation"] = field_jobs_map.get("Core / Foundation", 0) + count

        def process_competitiveness(*gpas):
            fields = list(CareerAnalyzerSpark.CAREER_FIELDS.keys())
            best_track = "Core / Foundation"
            best_gpa = 0.0
            
            for field, gpa_val in zip(fields, gpas):
                val = float(gpa_val) if gpa_val is not None else 0.0
                if val > best_gpa:
                    best_gpa = val
                    best_track = field
            
            # Competitiveness
            track_jobs = field_jobs_map.get(best_track, 10) # default 10 to avoid 0
            # Công thức cạnh tranh: Tỷ lệ phân bổ Job của mảng đó so với tổng IT
            # Nếu mảng đó chiếm % quá thấp -> Cạnh tranh Khốc liệt
            # Điểm mạnh (GPA lớn) cũng tăng lợi thế cạnh tranh
            market_ratio = (track_jobs / max(total_it_jobs, 1)) * 100
            
            competitiveness_label = "Trung bình"
            if best_gpa >= 3.2 and market_ratio > 10:
                competitiveness_label = "Rất Lợi Thế (High Demand + High GPA)"
            elif best_gpa >= 2.5 and market_ratio > 5:
                competitiveness_label = "Lợi thế khá (Đủ đáp ứng thị trường)"
            elif best_gpa < 2.0 or market_ratio < 2:
                competitiveness_label = "Cạnh tranh khốc liệt (Cần cải thiện Điểm/Kỹ năng)"
                
            report_str = f"Ngành Đề Xuất: {best_track} (GPA: {best_gpa:.2f}/4.0)\n"
            report_str += f"Nhu cầu TT: {track_jobs} Jobs (Sức hút: {market_ratio:.1f}% toàn ngành IT)\n"
            report_str += f"=> Đánh giá Cơ hội: {competitiveness_label}"
            
            return report_str

        report_udf = F.udf(process_competitiveness, StringType())
        
        gpa_cols = []
        for field in CareerAnalyzerSpark.CAREER_FIELDS.keys():
            col_name = f"gpa_{field.replace(' ', '_').replace('/', '_')}"
            gpa_cols.append(F.col(col_name))
            
        df_students = df_students.withColumn("top_matched_jobs", report_udf(*gpa_cols))
        return df_students

    @staticmethod
    def analyze_students(df_student, keyword="", market_csv=None, model_path=None):
        from src.ml.unsupervised.readiness_clustering import ReadinessClustering
        from src.ml.supervised.feature_importance_rf import RandomForestFeatureExtractor

        df = df_student
        if keyword:
            cond = None
            for c in ["ma_sv", "ho_ten"]:
                if c in df.columns:
                    expr = F.lower(F.col(c).cast("string")).contains(keyword.lower())
                    cond = expr if cond is None else (cond | expr)
            if cond is not None:
                df = df.filter(cond)

        if df.count() == 0:
            return None, [], []

        # ==========================
        # GIAI ĐOẠN 1: READINESS ML
        # Gọi thuật toán Machine Learning từ Layer src.ml
        # ==========================
        df = CareerAnalyzerSpark.compute_field_gpa(df)
        df = ReadinessClustering.cluster(df, model_path=model_path) # K-Means Unsupervised
        
        # Bỏ phần tính toán các môn học quan trọng (Random Forest) theo yêu cầu
        important_subjects = [] 
        # important_subjects = RandomForestFeatureExtractor.extract_important_subjects(df)

        # ==========================
        # GIAI ĐOẠN 2: MATCHING
        # ==========================
        real_jobs = CareerAnalyzerSpark.load_real_jobs(market_csv)
        
        if real_jobs:
            df = CareerAnalyzerSpark.compute_career_competitiveness(df, real_jobs)

        # Trả về Pandas DF trực tiếp cho GUI dễ render (GUI sẽ không phải tự decode Spark Row)
        pdf = df.toPandas()
        
        return pdf, real_jobs, important_subjects

"""
Phân loại chất lượng môn học bằng KMeans k=4
"""
import os
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans, KMeansModel


class SubjectReadinessClustering:
    """KMeans clustering phân loại chất lượng môn học"""

    @staticmethod
    def cluster(sdf, k=4, seed=42, model_path=None):
        """
        Phân loại môn học thành 4 nhóm: Tiêu cực, Không ổn định, Ổn định, Xuất sắc
        
        Args:
            sdf: Spark DataFrame dữ liệu môn học
            k: Số cụm (4)
            seed: Random seed
            model_path: Đường dẫn file mô hình
            
        Returns:
            DataFrame với cột ChatLuongCluster
        """
        # 10 features: TB, SD, A+%, A%, B%, C%, D%, F%, MTC%, TV
        feature_cols = ["TB", "SD", "A+%", "A%", "B%", "C%", "D%", "F%", "MTC%", "TV"]
        
        from pyspark.sql.types import DoubleType
        # Kiểm tra và ép kiểu an toàn
        for col in feature_cols:
            if col not in sdf.columns:
                sdf = sdf.withColumn(col, F.lit(0.0))
            else:
                sdf = sdf.withColumn(col, F.coalesce(F.col(col).cast(DoubleType()), F.lit(0.0)))
        
        # Vector features
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
        df_assembled = assembler.transform(sdf)

        # Chuẩn hóa (Bắt buộc)
        from pyspark.ml.feature import StandardScaler
        scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
        scaler_model = scaler.fit(df_assembled)
        df_vec = scaler_model.transform(df_assembled)
        
        # Nếu không có model_path, dùng mặc định
        if not model_path:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "models", "subject_quality_kmeans_model"
            )
        
        print(f"🔍 Đang nạp mô hình tại: {model_path}")
        
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Không tìm thấy thư mục mô hình tại {model_path}")
                
            model = KMeansModel.load(model_path)
            
            # Kiểm tra tính tương thích
            centers = model.clusterCenters()
            model_features = len(centers[0])
            if model_features != len(feature_cols):
                raise ValueError(f"Mô hình không tương thích! Yêu cầu {model_features} đặc trưng nhưng dữ liệu có {len(feature_cols)}")
            
            df_clustered = model.transform(df_vec)
        except Exception as e:
            raise RuntimeError(f"Lỗi nạp mô hình phân cụm. Vui lòng Train mô hình trước! Chi tiết: {e}")
        
        # Sắp xếp cluster theo TB (cột 0)
        cluster_quality = [(i, float(c[0])) for i, c in enumerate(centers)]
        cluster_quality.sort(key=lambda x: x[1], reverse=True)
        
        # Map rank → label (K=4)
        label_by_rank = [
            "Xuất sắc",       # rank 0
            "Khá",            # rank 1
            "Không ổn định",   # rank 2
            "Tiêu cực"        # rank 3
        ]
        
        mapping_expr = F.lit("Chưa xác định")
        for rank, (cluster_id, _) in enumerate(cluster_quality):
            label = label_by_rank[rank] if rank < len(label_by_rank) else "Chưa xác định"
            mapping_expr = F.when(F.col("prediction") == cluster_id, F.lit(label)).otherwise(mapping_expr)
        
        df_clustered = df_clustered.withColumn("ChatLuongCluster", mapping_expr)
        
        return df_clustered.drop("features", "raw_features")


class SubjectQualityPriority:
    """Map label → priority cho UI"""
    PRIORITY_MAP = {
        "Tiêu cực": 0,
        "Không ổn định": 1,
        "Khá": 2,
        "Xuất sắc": 3,
    }
    
    @staticmethod
    def get_priority(label):
        return SubjectQualityPriority.PRIORITY_MAP.get(label, 1)

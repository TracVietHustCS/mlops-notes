=== experiments | Nền tảng | Theo dõi thí nghiệm và tái lập kết quả
>> Huấn luyện mô hình là hàng chục, hàng trăm thí nghiệm. Nếu không ghi lại tham số, dữ liệu, mã và kết quả của từng lần, bạn không thể tái lập, so sánh hay giải thích vì sao mô hình hiện tại tốt.

## Cần ghi lại gì

| Nhóm | Nội dung |
|---|---|
| **Mã** | Commit Git (kèm cảnh báo nếu có thay đổi chưa commit) |
| **Dữ liệu** | Phiên bản, mã băm hoặc đường dẫn bất biến |
| **Tham số** | Siêu tham số, cấu hình tiền xử lý, hạt giống ngẫu nhiên |
| **Môi trường** | Phiên bản Python và thư viện, phần cứng, container image |
| **Chỉ số** | Kết quả trên train, validation, test; theo lát cắt; thời gian huấn luyện |
| **Sản phẩm** | Mô hình, biểu đồ, báo cáo, tệp cấu hình |

## MLflow Tracking

MLflow là công cụ mã nguồn mở phổ biến để theo dõi thí nghiệm (cùng registry, đóng gói, phục vụ).

```
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")      # nơi lưu siêu dữ liệu (cũng có thể là máy chủ tracking)
mlflow.set_experiment("churn")

params = {"n_estimators": 200, "max_depth": 6, "random_state": 42}

with mlflow.start_run(run_name="rf-baseline"):
    mlflow.log_params(params)
    model = RandomForestClassifier(**params).fit(X_train, y_train)
    auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
    mlflow.log_metric("valid_auc", auc)
    mlflow.sklearn.log_model(model, "model")          # lưu mô hình kèm môi trường
    mlflow.set_tag("data_version", "train-v3")
    mlflow.set_tag("git_commit", "3f1a9c2")
```

```
mlflow ui --backend-store-uri sqlite:///mlflow.db     # giao diện web tại http://127.0.0.1:5000
```

`mlflow.autolog()` ghi tự động tham số, chỉ số và mô hình cho nhiều thư viện (scikit-learn, XGBoost, PyTorch...) với rất ít mã, nhưng vẫn nên bổ sung **thẻ về phiên bản dữ liệu và commit**, vì autolog không biết điều đó.

Công cụ tương tự: Weights & Biases, Neptune, Comet, ClearML, các dịch vụ của cloud.

## Tái lập kết quả

Cùng mã và dữ liệu nhưng khác kết quả thường do:

| Nguyên nhân | Cách kiểm soát |
|---|---|
| **Hạt giống ngẫu nhiên** (chia dữ liệu, khởi tạo trọng số, xáo trộn) | Đặt `random_state`/seed cho mọi thư viện; ghi lại giá trị |
| **Phiên bản thư viện** | Khoá phiên bản (`requirements.txt` có phiên bản, lock file), hoặc container |
| **Phần cứng và tính không xác định** (GPU, song song) | Bật chế độ xác định nếu cần; chấp nhận sai khác nhỏ và đo bằng nhiều lần chạy |
| **Dữ liệu thay đổi âm thầm** | Phiên bản hoá dữ liệu |
| **Mã chưa commit** | Cảnh báo hoặc chặn chạy khi cây làm việc bẩn |
| **Thứ tự dữ liệu, đọc song song** | Sắp xếp xác định, ghi cách chia |

Đặt mục tiêu thực tế: **tái lập đến mức sai khác giải thích được** (ví dụ AUC dao động 0,002 giữa các lần do ngẫu nhiên), và chạy nhiều hạt giống để biết phương sai trước khi kết luận mô hình A tốt hơn B.

## So sánh thí nghiệm đúng cách

- So sánh trên **cùng tập kiểm định** và **cùng cách chia**.
- Báo cáo **trung bình và độ dao động** qua nhiều hạt giống hoặc cross-validation, không chỉ một con số.
- Giữ **tập kiểm thử cuối cùng** chưa chạm tới cho đến khi chọn xong mô hình: dùng nó nhiều lần là rò rỉ gián tiếp (bạn "học" tập kiểm thử bằng cách chọn).
- Ghi lại **thí nghiệm thất bại** nữa: kiến thức về cái không hiệu quả cũng có giá trị.
- Luôn có **đường cơ sở** đơn giản (dự đoán lớp phổ biến, hồi quy logistic, quy tắc nghiệp vụ) để biết mô hình phức tạp có đáng không.

## Tìm kiếm siêu tham số

Tìm kiếm lưới, ngẫu nhiên, tối ưu Bayes (Optuna, Ray Tune, Hyperopt) tạo hàng trăm lần chạy: ghi mỗi lần như một run con của MLflow, và **dừng sớm** các lần rõ ràng tệ để tiết kiệm. Giới hạn ngân sách tính toán.

## Cấu trúc dự án để tái lập

```
churn-model/
├── data/                # dữ liệu (theo dõi bằng DVC), không commit tệp lớn
├── src/
│   ├── features.py      # tiền xử lý/đặc trưng: dùng chung cho huấn luyện và phục vụ
│   ├── train.py         # nhận tham số từ params.yaml, ghi MLflow
│   └── evaluate.py
├── params.yaml          # mọi tham số ở MỘT nơi, nằm trong Git
├── dvc.yaml             # pipeline (xem trang Pipeline)
├── requirements.txt     # khoá phiên bản
├── tests/
└── Dockerfile           # môi trường huấn luyện/phục vụ giống nhau
```

Đặt mọi tham số trong tệp cấu hình (không viết cứng trong mã) để một lần chạy được xác định hoàn toàn bởi **commit Git + phiên bản dữ liệu**.

::: lab Theo dõi ba lần chạy và so sánh
```
python3 - <<'EOF'
import mlflow, mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

mlflow.set_tracking_uri("sqlite:///mlflow.db"); mlflow.set_experiment("lab-experiments")
X, y = make_classification(n_samples=2000, n_features=20, n_informative=5, random_state=1)
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=1)

configs = [("logreg", LogisticRegression(max_iter=500), {"C": 1.0}),
           ("rf-depth4", RandomForestClassifier(max_depth=4, random_state=0), {"max_depth": 4}),
           ("rf-depth12", RandomForestClassifier(max_depth=12, random_state=0), {"max_depth": 12})]
for name, model, p in configs:
    with mlflow.start_run(run_name=name):
        mlflow.log_params(p); mlflow.set_tag("data_version", "synthetic-seed1")
        model.fit(Xtr, ytr)
        auc = roc_auc_score(yva, model.predict_proba(Xva)[:, 1])
        mlflow.log_metric("valid_auc", auc); print(name, round(auc, 4))
EOF
mlflow ui --backend-store-uri sqlite:///mlflow.db &      # mở http://127.0.0.1:5000, so sánh các run; Ctrl+C khi xong
```
Trong giao diện, so sánh ba run theo `valid_auc`. Điều gì khiến bạn tin kết quả này đủ chắc để chọn mô hình? Bạn cần làm thêm gì (nhiều hạt giống, tập kiểm thử cuối, đường cơ sở) trước khi quyết định?
:::

=== pipelines | Nền tảng | Pipeline huấn luyện và điều phối
>> Pipeline biến việc huấn luyện từ chuỗi thao tác tay trong notebook thành một quy trình tự động, lặp lại được, chạy theo lịch hoặc theo sự kiện. Đây là bước chuyển từ mức MLOps 0 lên mức 1.

## Các bước điển hình

```ascii
  Lấy dữ liệu -> Kiểm tra dữ liệu -> Tính đặc trưng -> Huấn luyện -> Đánh giá -> [CỔNG] -> Đăng ký mô hình
        |                |                                                       |
   (phiên bản)    (dừng nếu dữ liệu xấu)                          (dừng nếu không tốt hơn mô hình hiện tại)
```

Mỗi bước là một đơn vị riêng, đầu vào và đầu ra rõ ràng, **idempotent**, và có thể chạy lại độc lập. Cổng đánh giá quyết định mô hình mới có được đi tiếp hay không ([Đánh giá](#evaluation)).

## DVC pipeline: pipeline gắn với dữ liệu và mã

DVC cho phép mô tả pipeline trong `dvc.yaml`, tự theo dõi phụ thuộc và **chỉ chạy lại bước nào có đầu vào thay đổi**:

```
# params.yaml
train:
  n_estimators: 200
  max_depth: 6
  seed: 42

# dvc.yaml
stages:
  prepare:
    cmd: python src/prepare.py data/raw.csv data/clean.csv
    deps: [src/prepare.py, data/raw.csv]
    outs: [data/clean.csv]
  train:
    cmd: python src/train.py data/clean.csv models/model.joblib
    deps: [src/train.py, data/clean.csv]
    params: [train.n_estimators, train.max_depth, train.seed]
    outs: [models/model.joblib]
    metrics:
      - metrics.json: {cache: false}
```

```
dvc repro                       # chạy các bước cần chạy lại
dvc metrics show                # chỉ số hiện tại
dvc metrics diff HEAD~1         # so với commit trước
dvc exp run -S train.max_depth=8     # thí nghiệm nhanh đổi tham số mà chưa commit
```

Điểm hay: pipeline, dữ liệu, tham số và chỉ số cùng gắn với **commit Git**; quay về một commit là quay về đúng trạng thái (kèm `dvc pull`).

## Công cụ điều phối

| Công cụ | Ghi chú |
|---|---|
| **DVC** | Nhẹ, gắn với Git, hợp pipeline cục bộ và nhóm nhỏ |
| **Apache Airflow, Dagster, Prefect** | Điều phối đa dụng, hợp khi pipeline ML là một phần của luồng dữ liệu lớn hơn ([Orchestration](https://tracviethustcs.github.io/dataops-notes/#orchestration)) |
| **Kubeflow Pipelines, Argo Workflows, Tekton** | Chạy từng bước như container trên Kubernetes, hợp quy mô lớn, cần GPU |
| **Metaflow, ZenML, Flyte** | Khung "Python-first" dành cho ML, ẩn bớt hạ tầng |
| **Dịch vụ của cloud** | SageMaker Pipelines, Vertex AI Pipelines, Azure ML Pipelines |

Chọn theo hạ tầng sẵn có và kỹ năng đội: một pipeline DVC trong CI thường đủ cho khởi đầu.

## Kích hoạt huấn luyện

| Kích hoạt | Khi nào hợp |
|---|---|
| **Theo lịch** (hằng tuần/tháng) | Dữ liệu đổi đều, mô hình cần làm mới định kỳ |
| **Khi có dữ liệu mới** | Dữ liệu đến thành lô |
| **Khi phát hiện drift hoặc suy giảm** | Có giám sát tốt ([Giám sát](#monitoring-drift)) |
| **Khi mã thay đổi** (CI) | Kiểm tra mã huấn luyện còn chạy được, chạy trên dữ liệu nhỏ |
| **Thủ công** | Thí nghiệm có chủ đích |

Học lại tự động (continuous training) rất tiện nhưng nguy hiểm nếu thiếu cổng: mô hình mới huấn luyện trên dữ liệu bị lỗi có thể **tự động triển khai thứ tệ hơn**. Luôn có kiểm tra dữ liệu, so sánh với mô hình hiện tại và (với rủi ro cao) bước duyệt.

## Nguyên tắc thiết kế pipeline

- **Bước nhỏ, đầu vào/đầu ra tường minh**, không giữ trạng thái ẩn.
- **Idempotent và có thể chạy lại**, giống tác vụ dữ liệu ([DataOps](https://tracviethustcs.github.io/dataops-notes/#orchestration)).
- **Tách cấu hình khỏi mã** (`params.yaml`).
- **Dùng cùng mã đặc trưng cho huấn luyện và phục vụ.**
- **Ghi lại tất cả** vào theo dõi thí nghiệm (mã, dữ liệu, tham số, chỉ số).
- **Môi trường đóng gói** (container) giống nhau mọi nơi chạy.
- **Cache** các bước tốn kém (đặc trưng, tiền xử lý) khi đầu vào không đổi.
- **Thử trên dữ liệu nhỏ** trước khi chạy toàn bộ tốn kém.

::: lab Pipeline DVC nhỏ chạy cục bộ
```
mkdir -p ~/dvc-lab/src ~/dvc-lab/data && cd ~/dvc-lab && git init -q . && pip install -q dvc scikit-learn pandas joblib pyyaml
dvc init -q
python3 - <<'EOF'
from sklearn.datasets import make_classification
import pandas as pd
X, y = make_classification(n_samples=1500, n_features=10, n_informative=4, random_state=0)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]); df["y"] = y; df.to_csv("data/raw.csv", index=False)
EOF
cat > src/train.py <<'EOF'
import sys, json, yaml, joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
p = yaml.safe_load(open("params.yaml"))["train"]
df = pd.read_csv(sys.argv[1]); X, y = df.drop(columns="y"), df["y"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=p["seed"])
m = RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], random_state=p["seed"]).fit(Xtr, ytr)
joblib.dump(m, sys.argv[2]); json.dump({"auc": roc_auc_score(yte, m.predict_proba(Xte)[:, 1])}, open("metrics.json", "w"))
EOF
printf 'train:\n  n_estimators: 100\n  max_depth: 5\n  seed: 42\n' > params.yaml
mkdir -p models
dvc stage add -n train -d src/train.py -d data/raw.csv -p train.n_estimators,train.max_depth,train.seed -o models/model.joblib -M metrics.json python src/train.py data/raw.csv models/model.joblib
dvc repro && cat metrics.json
dvc repro                                  # không đổi gì: DVC bỏ qua bước
sed -i 's/max_depth: 5/max_depth: 8/' params.yaml && dvc repro && dvc metrics diff
```
Quan sát: lần `dvc repro` thứ hai nói gì? Sau khi đổi `max_depth`, DVC chạy lại bước nào, và `dvc metrics diff` cho thấy điều gì?
:::

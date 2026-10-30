=== registry | Mô hình | Model registry
>> Model registry là kho trung tâm quản lý các phiên bản mô hình: ai huấn luyện, từ dữ liệu nào, chỉ số ra sao, đang ở giai đoạn nào, và bản nào đang chạy ở production. Nó biến "tệp mô hình trong thư mục" thành tài sản có kiểm soát.

## Vì sao cần

Không có registry, câu hỏi cơ bản không trả lời được: **mô hình đang chạy ở production là bản nào? huấn luyện từ đâu? có thể quay về bản trước không?** Mô hình bị lưu rải rác (`model_final_v2_ok.pkl`), không ai chắc cái nào đang được dùng.

Registry cung cấp:

| Khả năng | Ý nghĩa |
|---|---|
| **Phiên bản** | Mỗi lần đăng ký tạo phiên bản số mới, bất biến |
| **Lineage (nguồn gốc)** | Liên kết phiên bản với run huấn luyện, commit mã, phiên bản dữ liệu, tham số |
| **Siêu dữ liệu và chỉ số** | Chỉ số đánh giá, mô tả, thẻ, tác giả |
| **Giai đoạn / bí danh** | Đánh dấu bản nào là "champion" (production), "challenger" (ứng viên) |
| **Quy trình duyệt** | Chuyển bản ứng viên lên production cần đạt cổng và có người duyệt |
| **Lịch sử và kiểm toán** | Ai đổi gì, khi nào |
| **Quay lui** | Chuyển bí danh về phiên bản trước |

## MLflow Model Registry

```
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Đăng ký mô hình từ một run đã lưu (run_id lấy từ theo dõi thí nghiệm)
result = mlflow.register_model(f"runs:/{run_id}/model", "churn-model")
print("phiên bản mới:", result.version)

# Gắn bí danh thay cho "giai đoạn" (các bản MLflow mới ưu tiên bí danh và thẻ)
client.set_registered_model_alias("churn-model", "challenger", result.version)

# Sau khi vượt cổng đánh giá và được duyệt: nâng thành champion
client.set_registered_model_alias("churn-model", "champion", result.version)

# Dịch vụ tải mô hình theo bí danh, không cần biết số phiên bản
model = mlflow.pyfunc.load_model("models:/churn-model@champion")
preds = model.predict(X_new)
```

Điểm hay của bí danh: **dịch vụ phục vụ tham chiếu `@champion`**, và việc nâng cấp hay quay lui là **đổi con trỏ** (không phải triển khai lại mã). API chính xác (bí danh hay stage, `artifact_path` hay `name`) thay đổi giữa các phiên bản MLflow: đối chiếu tài liệu bản bạn dùng.

Các registry khác: registry của SageMaker, Vertex AI, Azure ML, Weights & Biases, Hugging Face Hub (cho mô hình dùng chung).

## Quy trình đưa mô hình lên production

```ascii
   Huấn luyện --> đăng ký phiên bản N --> [cổng tự động: chỉ số >= champion, kiểm thử hành vi, kiểm tra thiên lệch]
                                                    |
                                                  đạt --> alias "challenger" --> triển khai shadow/canary
                                                                                        |
                                              [duyệt của con người nếu rủi ro cao] --> alias "champion"
                                                                                        |
                                                                     giữ phiên bản cũ để quay lui nhanh
```

- **Cổng tự động** kiểm tra điều kiện khách quan ([Đánh giá](#evaluation)).
- **Duyệt của con người** cho mô hình rủi ro cao (tín dụng, y tế): xem tài liệu mô hình (model card), thiên lệch, tác động.
- **Giữ phiên bản cũ** có thể phục vụ ngay để quay lui.

## Tài liệu đi kèm mô hình

Mỗi phiên bản nên có:

- **Mô tả mục đích và giới hạn** (không dùng cho đối tượng nào).
- **Dữ liệu huấn luyện** (nguồn, khoảng thời gian, phiên bản).
- **Chỉ số đánh giá** tổng và theo lát cắt.
- **Môi trường** (thư viện, phiên bản) để tái lập.
- **Chữ ký (signature)** đầu vào/đầu ra mong đợi.
- **Người chịu trách nhiệm** và cách liên hệ.

Xem [Quản trị](#governance) về model card.

## Chữ ký và hợp đồng đầu vào

Registry (MLflow `signature`) lưu cấu trúc đầu vào/đầu ra; dịch vụ phục vụ có thể **từ chối đầu vào sai schema** thay vì trả dự đoán vô nghĩa. Đây là hợp đồng giữa mô hình và nơi gọi, giống [hợp đồng dữ liệu](https://tracviethustcs.github.io/dataops-notes/#principles-architecture).

## Bảo mật registry

- Mô hình có thể chứa **mã thực thi** (định dạng pickle tải về chạy được mã tuỳ ý): chỉ tải mô hình từ nguồn bạn tin và dùng định dạng an toàn hơn (ONNX, safetensors) khi có thể.
- Phân quyền: ai đăng ký, ai được nâng lên production.
- Ký và xác minh artifact mô hình ([chuỗi cung ứng](https://tracviethustcs.github.io/k8s-study-notes/#adv-supply-chain)).

::: lab Đăng ký, gắn bí danh và quay lui
```
python3 - <<'EOF'
import mlflow, mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
mlflow.set_tracking_uri("sqlite:///mlflow-reg.db"); mlflow.set_experiment("registry-lab")
client = MlflowClient()
X, y = make_classification(n_samples=500, random_state=0)
versions = []
for C in (0.1, 1.0):
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(LogisticRegression(C=C, max_iter=500).fit(X, y), "model")
        v = mlflow.register_model(f"runs:/{run.info.run_id}/model", "demo-model")
        versions.append(v.version)
print("các phiên bản:", versions)
client.set_registered_model_alias("demo-model", "champion", versions[0])
m = mlflow.pyfunc.load_model("models:/demo-model@champion"); print("champion v", versions[0], "dự đoán", m.predict(X[:3]))
client.set_registered_model_alias("demo-model", "champion", versions[1])       # nâng cấp
print("champion giờ là v", client.get_model_version_by_alias("demo-model", "champion").version)
client.set_registered_model_alias("demo-model", "champion", versions[0])       # quay lui chỉ là đổi con trỏ
print("quay lui về v", client.get_model_version_by_alias("demo-model", "champion").version)
EOF
```
Quan sát: việc nâng cấp và quay lui có cần triển khai lại mã dịch vụ không? Dịch vụ tải `models:/demo-model@champion` có thể phát hiện bí danh đổi bằng cách nào (tải lại định kỳ, hay khởi động lại)?
:::

=== evaluation | Mô hình | Đánh giá mô hình và cổng chất lượng
>> Đánh giá quyết định mô hình có được phát hành hay không. Chỉ số tổng thể một con số thường che giấu những chỗ mô hình tệ; cần đánh giá theo lát cắt, so với đường cơ sở và mô hình hiện tại, và kiểm thử hành vi.

## Chọn chỉ số đúng bài toán

| Bài toán | Chỉ số thường dùng | Lưu ý |
|---|---|---|
| Phân loại cân bằng | Accuracy, F1 | Accuracy vô nghĩa khi lớp lệch |
| Phân loại **lệch lớp** (gian lận, bệnh hiếm) | Precision, recall, PR-AUC, F1 | 99% accuracy có thể là mô hình luôn đoán "không gian lận" |
| Xếp hạng điểm | ROC-AUC, PR-AUC | Không phụ thuộc ngưỡng |
| Hồi quy | MAE, RMSE, MAPE | MAE bền với ngoại lệ, RMSE phạt sai lớn |
| Đề xuất, tìm kiếm | NDCG, MAP, recall@k | Đo thứ hạng |
| Dự báo chuỗi thời gian | MAE, sMAPE; đánh giá theo cửa sổ trượt | Chia theo thời gian |

Chỉ số kỹ thuật phải **gắn với chỉ số nghiệp vụ**: precision/recall được chuyển thành chi phí (mỗi cảnh báo giả tốn bao nhiêu, mỗi ca bỏ sót tốn bao nhiêu), và **ngưỡng quyết định** được chọn theo chi phí đó chứ không mặc định 0,5.

## Đường cơ sở và so sánh

- **Đường cơ sở đơn giản** (luôn đoán lớp phổ biến, hồi quy logistic, quy tắc nghiệp vụ): mô hình phức tạp phải thắng nó rõ rệt mới đáng.
- **So với mô hình hiện tại ("champion")** trên **cùng tập đánh giá**: mô hình mới tốt hơn thật hay chỉ nhiễu?
- **Khoảng tin cậy**: so sánh sai khác với độ dao động (nhiều hạt giống, bootstrap), không chỉ hai con số.

## Đánh giá theo lát cắt

Điểm trung bình tốt có thể che mô hình **rất tệ ở một nhóm** (một vùng, một thiết bị, một nhóm khách hàng thiểu số).

```
import pandas as pd
from sklearn.metrics import roc_auc_score

df = X_test.assign(y=y_test, score=model.predict_proba(X_test)[:, 1])
for col in ["region", "device"]:
    for val, g in df.groupby(col):
        if g["y"].nunique() == 2 and len(g) >= 200:                 # đủ mẫu và đủ hai lớp
            print(f"{col}={val}: n={len(g)} auc={roc_auc_score(g['y'], g['score']):.3f}")
```

Đặt **ngưỡng tối thiểu theo lát cắt quan trọng**, và chú ý lát cắt có ít mẫu (chỉ số dao động mạnh).

## Thiên lệch và công bằng

Đánh giá hiệu năng **tách theo nhóm được bảo vệ** (nếu được phép và phù hợp pháp lý) để thấy chênh lệch. Một số chỉ số công bằng:

| Chỉ số | Ý nghĩa |
|---|---|
| **Demographic parity** | Tỷ lệ dự đoán dương như nhau giữa các nhóm |
| **Equalized odds** | Tỷ lệ dương thật và dương giả như nhau giữa các nhóm |
| **Predictive parity** | Độ chính xác của dự đoán dương như nhau |

Các tiêu chí này **mâu thuẫn nhau** trong nhiều trường hợp (không thể thoả mãn đồng thời), nên chọn tiêu chí nào là **quyết định nghiệp vụ và đạo đức**, có sự tham gia của pháp lý, không chỉ kỹ thuật. Thư viện: Fairlearn, AIF360. Xem [Quản trị](#governance).

## Kiểm thử hành vi

Ngoài chỉ số, kiểm thử mô hình như phần mềm:

| Loại | Ví dụ |
|---|---|
| **Bất biến (invariance)** | Đổi tên người, đổi thứ tự không liên quan thì dự đoán không đổi |
| **Hướng mong đợi (directional)** | Tăng thu nhập (giữ nguyên các thứ khác) không làm điểm tín dụng giảm |
| **Ca biên** | Giá trị thiếu, cực trị, đầu vào hiếm |
| **Hồi quy** | Các ca đã từng sai được lưu lại thành bộ kiểm thử, mô hình mới không được tệ hơn |
| **Ổn định** | Nhiễu nhỏ ở đầu vào không làm dự đoán đổi lớn |

```
def test_dat_hang_khong_phu_thuoc_vao_ten(model, sample):
    a = sample.copy(); b = sample.copy()
    a["name"] = "An"; b["name"] = "Bình"
    assert model.predict(a)[0] == model.predict(b)[0]

def test_thu_nhap_cao_hon_khong_lam_giam_diem(model, sample):
    low = sample.copy(); high = sample.copy()
    low["income"] = 10_000; high["income"] = 20_000
    assert model.predict_proba(high)[0, 1] >= model.predict_proba(low)[0, 1] - 1e-6
```

## Cổng chất lượng trước khi đăng ký

Mô hình mới chỉ đi tiếp nếu đạt **mọi** điều kiện:

```
def quality_gate(new, champion, min_auc=0.75, max_drop=0.005):
    checks = {
        "AUC tối thiểu":            new["auc"] >= min_auc,
        "Không tệ hơn champion":    new["auc"] >= champion["auc"] - max_drop,
        "Lát cắt yếu nhất đủ":      new["worst_slice_auc"] >= 0.70,
        "Độ trễ dự đoán đạt":       new["p95_latency_ms"] <= 50,
        "Kiểm thử hành vi qua":     new["behavior_tests_passed"],
    }
    failed = [k for k, ok in checks.items() if not ok]
    return (not failed), failed
```

- Cổng là **mã**, nằm trong pipeline và Git, thay đổi qua review.
- Ngưỡng cần có lý do (từ yêu cầu nghiệp vụ), không tuỳ ý.
- **Thất bại thì dừng và báo**, không tự động triển khai.
- Ghi kết quả cổng vào registry (thẻ) để kiểm toán.

## Đánh giá ngoại tuyến và trực tuyến

Chỉ số ngoại tuyến (trên dữ liệu lịch sử) là **đại diện** cho hiệu quả thật, không phải bản thân nó. Mô hình tốt ngoại tuyến vẫn có thể không cải thiện chỉ số kinh doanh. Xác nhận bằng thí nghiệm trực tuyến (A/B, canary) trước khi coi là thành công ([Chiến lược triển khai](#deploy-strategies)).

::: lab Cổng chất lượng với đánh giá theo lát cắt
```
python3 - <<'EOF'
import numpy as np, pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

X, y = make_classification(n_samples=4000, n_features=15, n_informative=6, random_state=3)
region = np.random.RandomState(3).choice(["A", "B", "C"], size=len(y), p=[0.6, 0.3, 0.1])
Xtr, Xte, ytr, yte, rtr, rte = train_test_split(X, y, region, test_size=0.3, random_state=3)

champion = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
challenger = RandomForestClassifier(n_estimators=150, random_state=3).fit(Xtr, ytr)

def report(m):
    s = m.predict_proba(Xte)[:, 1]
    out = {"auc": roc_auc_score(yte, s)}
    out["worst_slice_auc"] = min(roc_auc_score(yte[rte == r], s[rte == r]) for r in "ABC")
    return out

c, n = report(champion), report(challenger)
print("champion  :", {k: round(v, 3) for k, v in c.items()})
print("challenger:", {k: round(v, 3) for k, v in n.items()})
checks = {"AUC >= 0.75": n["auc"] >= 0.75, "không tệ hơn champion": n["auc"] >= c["auc"] - 0.005, "lát cắt yếu nhất >= 0.70": n["worst_slice_auc"] >= 0.70}
print(checks, "->", "ĐI TIẾP" if all(checks.values()) else "DỪNG")
EOF
```
Challenger có vượt cổng không? Lát cắt nào yếu nhất, và nếu AUC tổng thể tốt nhưng một vùng nhỏ rất tệ, bạn sẽ có quyết định gì và với ai?
:::

=== serving | Sản xuất | Phục vụ mô hình: batch, online và streaming
>> Mô hình chỉ có giá trị khi tạo ra dự đoán đúng chỗ, đúng lúc. Cách phục vụ phụ thuộc vào độ trễ yêu cầu và khối lượng: tính trước hàng loạt, trả lời từng yêu cầu, hoặc xử lý luồng.

## Ba kiểu phục vụ

| Kiểu | Cách | Độ trễ | Hợp với |
|---|---|---|---|
| **Batch** (theo lô) | Chạy mô hình định kỳ trên cả tập dữ liệu, ghi kết quả vào bảng | Giờ đến ngày | Báo cáo, điểm số tính trước (khách có khả năng rời bỏ), đề xuất gửi email |
| **Online** (thời gian thực) | Dịch vụ API trả dự đoán cho từng yêu cầu | Mili giây đến vài trăm ms | Tìm kiếm, gian lận lúc thanh toán, đề xuất trong ứng dụng |
| **Streaming** | Xử lý sự kiện liên tục, dự đoán trên luồng | Giây | Phát hiện bất thường, IoT |

**Batch** thường là lựa chọn đầu tiên nên cân nhắc: đơn giản, rẻ, dễ kiểm tra, dễ quay lui. Chọn online khi dự đoán phụ thuộc vào ngữ cảnh chỉ có lúc yêu cầu và độ trễ giây là không chấp nhận.

## Dịch vụ online đơn giản với FastAPI

```
# serve.py
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()
model = joblib.load("model.joblib")                # tải một lần lúc khởi động, không tải mỗi yêu cầu

class Features(BaseModel):
    f0: float
    f1: float
    f2: float
    tenure_months: int = Field(ge=0, le=600)       # kiểm tra đầu vào: từ chối giá trị vô lý

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(x: Features):
    row = np.array([[x.f0, x.f1, x.f2, x.tenure_months]])
    p = float(model.predict_proba(row)[0, 1])
    return {"churn_probability": p, "model_version": "1"}
```

```
uvicorn serve:app --host 0.0.0.0 --port 8000
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"f0":0.1,"f1":-0.3,"f2":1.2,"tenure_months":14}'
```

Điểm quan trọng khi phục vụ:

- **Kiểm tra đầu vào** (schema, miền giá trị): đầu vào lạ không được trả dự đoán vô nghĩa mà không báo.
- **Tải mô hình một lần**, giữ trong bộ nhớ.
- **Healthcheck** và `readiness` đúng nghĩa ([Probe](https://tracviethustcs.github.io/k8s-study-notes/#ops-probes)): chỉ nhận traffic khi mô hình đã tải xong.
- **Ghi log** đầu vào và dự đoán (có kiểm soát PII) để giám sát và học lại ([Giám sát](#monitoring-drift)).
- **Trả kèm phiên bản mô hình** để truy vết.
- **Giới hạn thời gian và tài nguyên**; dự đoán mất quá lâu thì có phương án dự phòng (giá trị mặc định, quy tắc đơn giản).

## Các nền tảng phục vụ chuyên dụng

| Công cụ | Ghi chú |
|---|---|
| **MLflow Models** | `mlflow models serve -m models:/name@champion` dựng nhanh REST API |
| **BentoML** | Đóng gói mô hình thành dịch vụ có sẵn batching, Docker |
| **KServe, Seldon Core** | Phục vụ trên Kubernetes, tự co giãn, triển khai canary |
| **TorchServe, TensorFlow Serving, Triton Inference Server** | Tối ưu hiệu năng cho mô hình học sâu và GPU |
| Dịch vụ cloud | SageMaker, Vertex AI, Azure ML endpoints |
| **ONNX Runtime** | Chạy mô hình ở định dạng trung lập, nhanh và nhẹ |

KServe trên Kubernetes:

```
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: churn
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 5
    model:
      modelFormat: {name: sklearn}
      storageUri: s3://models/churn/1
```

Tính năng, tên trường và định dạng hỗ trợ thay đổi theo phiên bản: đối chiếu tài liệu.

## Hiệu năng suy luận

| Kỹ thuật | Tác dụng |
|---|---|
| **Gộp yêu cầu (dynamic batching)** | Tăng thông lượng, nhất là với GPU, đổi lại tăng độ trễ một chút |
| **Lượng tử hoá, chưng cất, cắt tỉa** | Giảm kích thước và độ trễ, đổi lại chút độ chính xác |
| **Chuyển sang ONNX/TensorRT** | Tối ưu cho phần cứng |
| **Cache kết quả** | Cho đầu vào lặp lại |
| **Tính trước (batch) rồi tra cứu** | Biến bài toán online thành tra cứu |
| **Giảm đặc trưng tốn kém ở đường nóng** | Đặc trưng cần truy vấn database làm chậm toàn bộ |

Đo **p95 và p99** (không chỉ trung bình), và kiểm thử tải ([Kiểm thử hiệu năng](https://tracviethustcs.github.io/devsecops-testing-notes/#performance-testing)).

## Đóng gói bằng container

```
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serve.py model.joblib ./
USER 10001
EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

Phụ thuộc phải **khớp phiên bản** lúc huấn luyện (đặc biệt scikit-learn: mô hình lưu bằng một phiên bản có thể không tải được hoặc cho kết quả khác ở phiên bản khác). Ghim phiên bản, dùng cùng image nền cho huấn luyện và phục vụ ([Dockerfile tốt](https://tracviethustcs.github.io/docker-container-notes/#build-best)).

## Bảo mật khi phục vụ

- **Không tải mô hình từ nguồn không tin cậy** (pickle thực thi mã).
- **Xác thực và giới hạn tốc độ** API; mô hình có thể bị dò để đánh cắp hoặc tấn công đối nghịch.
- **Không trả thông tin nhạy cảm** về dữ liệu huấn luyện.
- Quyền tối thiểu cho dịch vụ truy cập kho mô hình ([Bảo mật container](https://tracviethustcs.github.io/docker-container-notes/#security)).

::: lab Dựng và gọi một dịch vụ dự đoán
```
mkdir -p ~/serve-lab && cd ~/serve-lab
python3 - <<'EOF'
import joblib
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
X, y = make_classification(n_samples=800, n_features=4, n_informative=3, n_redundant=0, random_state=0)
joblib.dump(LogisticRegression(max_iter=500).fit(X, y), "model.joblib")
EOF
cat > serve.py <<'EOF'
import joblib, numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
app = FastAPI(); model = joblib.load("model.joblib")
class Features(BaseModel):
    f0: float; f1: float; f2: float
    tenure_months: int = Field(ge=0, le=600)
@app.get("/healthz")
def health(): return {"status": "ok"}
@app.post("/predict")
def predict(x: Features):
    p = float(model.predict_proba(np.array([[x.f0, x.f1, x.f2, x.tenure_months]]))[0, 1])
    return {"churn_probability": round(p, 4), "model_version": "1"}
EOF
uvicorn serve:app --port 8000 >/dev/null 2>&1 & sleep 3
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"f0":0.1,"f1":-0.3,"f2":1.2,"tenure_months":14}'; echo
curl -s -o /dev/null -w "đầu vào vô lý -> HTTP %{http_code}\n" -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"f0":0.1,"f1":-0.3,"f2":1.2,"tenure_months":-5}'
kill $!
```
Quan sát: dự đoán và mã trả về với đầu vào vô lý là gì (kỳ vọng 422)? Bạn sẽ thêm gì để ghi log dự đoán phục vụ giám sát mà không lộ dữ liệu nhạy cảm?
:::

=== deploy-strategies | Sản xuất | Chiến lược triển khai mô hình
>> Mô hình mới tốt ngoại tuyến chưa chắc tốt ở production. Triển khai dần với quan sát giúp biết chắc, và quay lui nhanh khi không như mong đợi.

## Vì sao khác triển khai phần mềm thường

- Lỗi của mô hình **không gây lỗi** (không ngoại lệ, không mã 500): chỉ là **dự đoán kém** mà không ai thấy ngay.
- Chất lượng phụ thuộc **dữ liệu thật**, không kiểm thử đầy đủ được trước.
- Nhãn đúng thường **đến trễ**, nên phản hồi chậm.

Vì vậy triển khai mô hình cần chiến lược **quan sát trước, quyết định sau**, với chỉ số rõ ràng.

## Các chiến lược

| Chiến lược | Cách làm | Ưu | Lưu ý |
|---|---|---|---|
| **Thay thế trực tiếp** | Đổi bản mới cho mọi yêu cầu | Đơn giản | Rủi ro cao, không có tín hiệu trước |
| **Shadow (bóng)** | Bản mới nhận **bản sao** yêu cầu thật nhưng dự đoán **không được dùng**; so với bản hiện tại | An toàn tuyệt đối với người dùng; so sánh trên traffic thật | Tốn gấp đôi tài nguyên; chưa đo tác động nghiệp vụ |
| **Canary** | Bản mới phục vụ một phần nhỏ yêu cầu thật, tăng dần nếu ổn | Giới hạn rủi ro, đo được tác động thật | Cần chỉ số tốt, đủ lưu lượng |
| **A/B test** | Chia người dùng ngẫu nhiên hai nhóm (cũ, mới), so chỉ số nghiệp vụ có ý nghĩa thống kê | Bằng chứng nhân quả về tác động | Cần thiết kế thí nghiệm đúng, đủ mẫu và thời gian |
| **Champion / challenger** | Chạy nhiều ứng viên song song trên một phần traffic, chọn bản tốt nhất | Tìm bản tốt nhất liên tục | Phức tạp hơn |
| **Bandit đa cánh** | Tự điều chỉnh tỷ lệ traffic theo hiệu quả quan sát được | Giảm chi phí "thử nghiệm" | Khó phân tích hơn A/B |
| **Interleaving** (xếp hạng) | Trộn kết quả của hai bản trong cùng một danh sách | Nhạy, cần ít mẫu | Chỉ cho hệ xếp hạng |

```ascii
   Yêu cầu --> Router --+--> Mô hình CHAMPION  --> dự đoán trả về người dùng
                        |
                        +--> Mô hình CHALLENGER (shadow) --> dự đoán chỉ GHI LẠI để so sánh
                        |
                        +--> (canary: một phần nhỏ yêu cầu trả kết quả của challenger)
```

## Chỉ số để quyết định

| Loại | Ví dụ |
|---|---|
| **Sức khoẻ kỹ thuật** | Độ trễ p95, tỷ lệ lỗi, tài nguyên |
| **Chất lượng dự đoán** | Phân phối điểm so với champion, tỷ lệ đồng thuận, chỉ số khi có nhãn |
| **Nghiệp vụ** | Tỷ lệ chuyển đổi, doanh thu, tỷ lệ khiếu nại, tỷ lệ gian lận bị chặn |
| **Ràng buộc/công bằng** | Chỉ số theo lát cắt không xấu đi |

Quy tắc canary: định **ngưỡng dừng và quay lui tự động** trước ([Progressive delivery](https://tracviethustcs.github.io/cicd-release-notes/#progressive-delivery)). Với mô hình, chỉ số nghiệp vụ thường quan trọng hơn chỉ số kỹ thuật vì mô hình có thể chạy hoàn hảo về kỹ thuật mà dự đoán tệ.

## A/B test mô hình: cẩn thận về thống kê

- **Chia ngẫu nhiên theo đơn vị đúng** (người dùng, không phải yêu cầu) để một người luôn thấy cùng một bản.
- **Định kích thước mẫu và thời gian trước**, đừng "nhìn lén" kết quả rồi dừng khi thấy đẹp (tăng tỷ lệ kết luận sai).
- **Một chỉ số chính** quyết định, các chỉ số phụ để quan sát.
- Chú ý **hiệu ứng mới lạ** (người dùng phản ứng khác với thứ mới) và tính **mùa vụ**.
- Thí nghiệm có thể **ảnh hưởng lẫn nhau** nếu chạy nhiều cái cùng lúc trên cùng người dùng.

## Triển khai và quay lui

- Triển khai là **đổi con trỏ** (bí danh trong registry, trọng số trong router), nên quay lui nhanh ([Registry](#registry)).
- Giữ **phiên bản cũ luôn sẵn sàng** phục vụ.
- **Đặt phiên bản mô hình và phiên bản mã/đặc trưng cùng nhau**: mô hình mới có thể cần đặc trưng mới; hai thứ phải triển khai tương thích ([Chiến lược phát hành](https://tracviethustcs.github.io/cicd-release-notes/#deploy-strategies)).
- Cho **phương án dự phòng (fallback)**: khi mô hình lỗi hoặc quá chậm, dùng quy tắc đơn giản thay vì làm hỏng dịch vụ.
- Với mô hình học liên tục, cẩn thận **vòng phản hồi**: mô hình ảnh hưởng dữ liệu về sau của chính nó.

::: lab Mô phỏng shadow so sánh hai mô hình
```
python3 - <<'EOF'
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
X, y = make_classification(n_samples=6000, n_features=12, n_informative=5, random_state=5)
Xtr, Xlive, ytr, ylive = train_test_split(X, y, test_size=0.5, random_state=5)
champion = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
challenger = GradientBoostingClassifier(random_state=5).fit(Xtr, ytr)

# "traffic thật": cả hai dự đoán; chỉ champion trả về người dùng
pc, pn = champion.predict_proba(Xlive)[:, 1], challenger.predict_proba(Xlive)[:, 1]
agree = np.mean((pc > 0.5) == (pn > 0.5))
print("tỷ lệ đồng thuận quyết định:", round(agree, 3))
print("điểm trung bình champion/challenger:", round(pc.mean(), 3), round(pn.mean(), 3))
# nhãn đến trễ: sau đó mới so sánh chất lượng
from sklearn.metrics import roc_auc_score
print("AUC khi nhãn về: champion", round(roc_auc_score(ylive, pc), 3), "challenger", round(roc_auc_score(ylive, pn), 3))
EOF
```
Nếu tỷ lệ đồng thuận thấp, challenger đang làm điều gì rất khác? Bạn sẽ chờ tối thiểu bao lâu (nhãn về) trước khi chuyển sang canary, và ngưỡng dừng nào bạn đặt trước?
:::

=== monitoring-drift | Sản xuất | Giám sát và phát hiện drift
>> Mô hình suy giảm mà không báo lỗi. Giám sát học máy nhìn vào dữ liệu vào, dự đoán và (khi có) kết quả thật, để phát hiện suy giảm trước khi người dùng phát hiện.

## Các loại thay đổi làm mô hình suy giảm

| Loại | Ý nghĩa | Ví dụ |
|---|---|---|
| **Data drift / covariate shift** | **Phân phối đầu vào** đổi | Người dùng trẻ hơn trước, thiết bị mới, thay đổi ở nguồn dữ liệu |
| **Concept drift** | **Quan hệ giữa đầu vào và đáp án** đổi | Hành vi gian lận đổi, dịch bệnh làm thói quen mua sắm khác |
| **Label / prior shift** | Tỷ lệ các lớp đổi | Tỷ lệ khách rời bỏ tăng |
| **Thay đổi ở pipeline dữ liệu** | Đặc trưng bị tính sai, đơn vị đổi, cột mới rỗng | Lỗi, không phải drift thật: **nguyên nhân hay gặp nhất** |
| **Training-serving skew** | Đặc trưng lúc phục vụ khác lúc huấn luyện | Xem [Dữ liệu và đặc trưng](#data-features) |

Trước khi kết luận "drift", **loại trừ lỗi pipeline dữ liệu**: nhiều "drift" thật ra là cột bị thiếu hay đổi đơn vị.

## Cần giám sát những gì

```ascii
   Dữ liệu vào  ->  [ MÔ HÌNH ]  ->  Dự đoán  ->  (sau một thời gian) Kết quả thật / nhãn
        |                              |                       |
   drift đặc trưng,              phân phối điểm,        hiệu năng thật (AUC, lỗi),
   thiếu, ngoài miền             tỷ lệ lớp dự đoán      so với lúc huấn luyện
        +---- cùng với: sức khoẻ dịch vụ (độ trễ, lỗi, tài nguyên), chỉ số nghiệp vụ ----+
```

| Tầng | Chỉ số | Nhãn cần? |
|---|---|---|
| **Sức khoẻ dịch vụ** | Độ trễ, tỷ lệ lỗi, lưu lượng, tài nguyên | Không |
| **Chất lượng dữ liệu vào** | Tỷ lệ thiếu, ngoài miền, kiểu sai, độ tươi | Không |
| **Drift đầu vào** | So sánh phân phối từng đặc trưng với tập huấn luyện | Không |
| **Drift dự đoán** | Phân phối điểm, tỷ lệ dự đoán dương | Không |
| **Hiệu năng thật** | AUC, precision/recall, lỗi dự báo | **Có** (thường đến trễ) |
| **Nghiệp vụ** | Chuyển đổi, doanh thu, khiếu nại | Không |

Khi **nhãn đến trễ** (vài tuần hoặc vài tháng), giám sát drift đầu vào và dự đoán là **cảnh báo sớm**, còn hiệu năng thật xác nhận sau.

## Đo drift

| Phương pháp | Dùng cho | Ghi chú |
|---|---|---|
| **PSI (Population Stability Index)** | Đặc trưng số và phân loại (sau khi chia khoảng) | Quy ước hay dùng: dưới 0,1 ổn định; 0,1 đến 0,25 thay đổi vừa; trên 0,25 thay đổi đáng kể |
| **Kiểm định Kolmogorov-Smirnov (KS)** | Đặc trưng số liên tục | Cho p-value; rất nhạy khi mẫu lớn |
| **Chi-bình phương** | Đặc trưng phân loại | So tần suất các giá trị |
| **Khoảng cách Jensen-Shannon, Wasserstein** | So sánh phân phối | Đo "mức khác nhau" |
| **Phân loại miền (domain classifier)** | Nhiều đặc trưng cùng lúc | Huấn luyện mô hình phân biệt "huấn luyện" với "sản xuất"; AUC cao nghĩa là phân phối khác nhau |

```
import numpy as np

def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)        # tránh chia cho 0
    return float(np.sum((a - e) * np.log(a / e)))
```

Các ngưỡng là quy ước kinh nghiệm, không phải quy luật: hiệu chỉnh theo từng đặc trưng và theo tác động lên mô hình. **Drift trong một đặc trưng không quan trọng với mô hình thì không đáng báo động**; ưu tiên đặc trưng có tầm quan trọng cao.

Công cụ: **Evidently**, **WhyLabs/whylogs**, **NannyML**, các tính năng giám sát của SageMaker, Vertex AI, Arize, Fiddler.

## Ghi log để giám sát

Dịch vụ phục vụ cần ghi lại (có kiểm soát PII và chi phí lưu trữ):

- **Đặc trưng đầu vào** (hoặc mẫu), **dự đoán**, **phiên bản mô hình**, thời điểm, mã định danh yêu cầu.
- Một **khoá liên kết** để sau này ghép với **nhãn thật** khi nó về.

Mẫu hợp lý nếu lưu lượng quá lớn (lấy mẫu có hệ thống, nhưng luôn giữ các trường hợp bất thường).

## Cảnh báo và hành động

| Phát hiện | Hành động |
|---|---|
| Lỗi pipeline dữ liệu (cột rỗng, đơn vị đổi) | Sửa nguồn; có thể tạm quay lui hoặc dùng phương án dự phòng |
| Drift đầu vào vượt ngưỡng ở đặc trưng quan trọng | Điều tra nguyên nhân; chuẩn bị học lại |
| Phân phối dự đoán đổi mạnh | Điều tra; kiểm tra tiền xử lý |
| Hiệu năng thật giảm dưới ngưỡng | Quay lui về bản trước hoặc học lại và triển khai bản mới |
| Chỉ số nghiệp vụ xấu đi | Điều tra theo cả nhóm: mô hình, dữ liệu, thay đổi sản phẩm |

**Học lại tự động khi drift** là tuỳ chọn có sẵn nhưng phải kèm cổng chất lượng ([Pipeline](#pipelines), [Đánh giá](#evaluation)): học lại trên dữ liệu đang lỗi sẽ tạo mô hình lỗi.

Định **SLO** cho mô hình (ví dụ "AUC trên 0,78 trong cửa sổ 30 ngày, p95 độ trễ dưới 80 ms") và đo như mọi dịch vụ khác.

::: lab Phát hiện drift bằng PSI và KS
```
python3 - <<'EOF'
import numpy as np
from scipy.stats import ks_2samp

def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

rng = np.random.default_rng(0)
train = rng.normal(40, 10, 20000)                 # tuổi lúc huấn luyện
stable = rng.normal(40, 10, 5000)                 # tuần này, không đổi
drifted = rng.normal(46, 12, 5000)                # tuần này, người dùng già hơn và phân tán hơn
broken = np.where(rng.random(5000) < 0.3, np.nan, rng.normal(40, 10, 5000))   # 30% thiếu (lỗi pipeline)

print("PSI ổn định :", round(psi(train, stable), 4))
print("PSI drift   :", round(psi(train, drifted), 4))
print("KS drift p  :", ks_2samp(train, drifted).pvalue)
print("tỷ lệ NULL ở dữ liệu lỗi:", round(float(np.isnan(broken).mean()), 3), "<- bắt bằng kiểm tra chất lượng, không phải PSI")
EOF
```
PSI của bản ổn định và bản lệch khác nhau ra sao so với ngưỡng 0,1 và 0,25? Vì sao với mẫu rất lớn p-value của KS gần như luôn "có ý nghĩa", và bạn sẽ dùng độ lớn của PSI thay vì chỉ p-value như thế nào?
:::

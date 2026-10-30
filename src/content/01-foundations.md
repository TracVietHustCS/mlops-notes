=== overview | Bắt đầu | Tổng quan và cách học
>> Bộ ghi chú về MLOps: đưa mô hình học máy từ notebook thành hệ thống chạy ổn định, tái lập được và được giám sát. Đi từ vòng đời, dữ liệu và đặc trưng, theo dõi thí nghiệm, pipeline, registry, đánh giá, phục vụ, triển khai, giám sát drift, CI/CD/CT, hạ tầng và quản trị.

::: warn Đọc trước khi dùng
Đây là ghi chú **tự học, tự soạn**, không phải tài liệu chính thức của MLflow, Kubeflow, DVC, Feast hay nhà cung cấp nào. Công cụ và API học máy thay đổi nhanh (nhất là MLflow, các nền tảng phục vụ mô hình); ghi chú không tra cứu trực tuyến nên có thể lỗi thời, hãy đối chiếu tài liệu hiện hành và thử trên dữ liệu, mô hình nhỏ trước. Mô hình ảnh hưởng tới quyết định về con người (tín dụng, tuyển dụng, y tế...) cần được đánh giá, kiểm soát và tuân thủ pháp lý kỹ hơn rất nhiều so với nội dung ở đây.
:::

## Bạn sẽ làm được gì sau khi học xong

- Giải thích vì sao đưa mô hình vào sản xuất khó hơn huấn luyện, và MLOps khác DevOps, DataOps ở đâu.
- Theo dõi thí nghiệm và tái lập kết quả (dữ liệu, mã, tham số, môi trường) bằng MLflow và DVC.
- Dựng pipeline huấn luyện, registry mô hình và cổng đánh giá trước khi phát hành.
- Phục vụ mô hình (batch, online), triển khai an toàn (shadow, canary) và giám sát drift.
- Thiết kế CI/CD/CT cho học máy, kiểm thử dữ liệu và mô hình, và hiểu quản trị mô hình.

## Cấu trúc

| Nhóm | Trang | Nội dung |
|---|---|---|
| Nền tảng | [Vòng đời và nguyên tắc](#mlops-principles), [Dữ liệu và đặc trưng](#data-features), [Theo dõi thí nghiệm](#experiments), [Pipeline](#pipelines) | Làm cho ML tái lập được |
| Mô hình | [Registry](#registry), [Đánh giá](#evaluation) | Quản lý và kiểm soát chất lượng |
| Sản xuất | [Phục vụ](#serving), [Chiến lược triển khai](#deploy-strategies), [Giám sát và drift](#monitoring-drift) | Đưa lên và giữ ổn định |
| Tự động hoá | [CI/CD/CT và kiểm thử ML](#ci-cd-ct), [Hạ tầng và chi phí](#infra-cost), [Quản trị](#governance) | Quy trình, hạ tầng, trách nhiệm |
| Thực hành | [Dự án mẫu cục bộ](#lab-stack) | scikit-learn + MLflow + FastAPI |
| Tham chiếu | [Cheat sheet](#cheatsheet), [Luyện tập 1](#quiz-1), [Luyện tập 2](#quiz-2) | Tra nhanh và tự kiểm tra |

## Môi trường thực hành

Phần lớn lab chạy cục bộ trên CPU, với mô hình nhỏ: Python 3.10 trở lên, scikit-learn, MLflow, FastAPI. Không cần GPU hay cloud.

```
python3 -m venv ~/mlops-venv && . ~/mlops-venv/bin/activate
pip install scikit-learn pandas numpy scipy mlflow fastapi uvicorn httpx joblib pytest
python -c "import sklearn, mlflow; print(sklearn.__version__, mlflow.__version__)"
```

Liên quan: dữ liệu đầu vào ở [DataOps](https://tracviethustcs.github.io/dataops-notes/), pipeline ở [CI/CD](https://tracviethustcs.github.io/cicd-release-notes/), đóng gói ở [Docker](https://tracviethustcs.github.io/docker-container-notes/), chạy trên [Kubernetes](https://tracviethustcs.github.io/k8s-study-notes/), ứng dụng mô hình ngôn ngữ lớn ở [LLMOps](https://tracviethustcs.github.io/llmops-notes/).

=== mlops-principles | Nền tảng | Vòng đời học máy và nguyên tắc MLOps
>> Huấn luyện được một mô hình tốt trong notebook chỉ là phần nhỏ của công việc. MLOps lo phần còn lại: dữ liệu, pipeline, phục vụ, giám sát và vận hành liên tục, vì mô hình hỏng theo cách mà phần mềm thường không hỏng.

## Mô hình trong sản xuất khác phần mềm thường

Hệ thống học máy có **ba thứ thay đổi** thay vì một: **mã**, **dữ liệu** và **mô hình**. Và mô hình có thể **suy giảm mà không có thay đổi mã nào**, chỉ vì thế giới đổi (hành vi người dùng, mùa vụ, dữ liệu nguồn).

```ascii
        MÃ (code)  +  DỮ LIỆU (data)  --huấn luyện-->  MÔ HÌNH (model)  --phục vụ-->  DỰ ĐOÁN
            |              |                               |                              |
        phiên bản      phiên bản                       phiên bản                     giám sát: dữ liệu vào,
        trong Git      (DVC, snapshot)                 (registry)                    dự đoán, kết quả thật
     Mỗi kết quả phải truy ngược được tới đúng bộ ba mã + dữ liệu + cấu hình.
```

Bài báo "Hidden Technical Debt in Machine Learning Systems" chỉ ra phần mã học máy chỉ là hộp nhỏ giữa nhiều thành phần hạ tầng (thu thập dữ liệu, kiểm tra, phục vụ, giám sát...). Nợ kỹ thuật điển hình của ML: mã keo (glue code), "rừng pipeline", phụ thuộc dữ liệu không được khai báo, vòng phản hồi (mô hình ảnh hưởng dữ liệu tương lai của chính nó), nợ cấu hình.

## Vòng đời

```ascii
  Xác định bài toán -> Thu thập/chuẩn bị dữ liệu -> Đặc trưng -> Huấn luyện & thí nghiệm
        ^                                                                  |
        |                                                                  v
  Giám sát & học lại  <-  Phục vụ/triển khai  <-  Đăng ký (registry)  <-  Đánh giá & kiểm thử
        |__________ phản hồi: drift, hiệu năng giảm, dữ liệu mới, yêu cầu mới ___________|
```

| Giai đoạn | Việc chính | Rủi ro điển hình |
|---|---|---|
| Xác định bài toán | Mục tiêu nghiệp vụ, chỉ số thành công, ràng buộc (độ trễ, chi phí) | Tối ưu sai chỉ số, bài toán không cần ML |
| Dữ liệu | Thu thập, làm sạch, gán nhãn, chia tập | Rò rỉ dữ liệu, lệch mẫu, nhãn sai |
| Đặc trưng | Biến đổi dữ liệu thành đầu vào mô hình | **Lệch huấn luyện/phục vụ** (training-serving skew) |
| Huấn luyện | Thí nghiệm, tinh chỉnh tham số | Không tái lập được |
| Đánh giá | Chỉ số ngoại tuyến, kiểm thử hành vi | Đánh giá trên dữ liệu không đại diện |
| Triển khai | Đóng gói, phục vụ, phát hành dần | Hành vi khác môi trường huấn luyện |
| Giám sát | Dữ liệu, dự đoán, hiệu năng | Phát hiện muộn khi mô hình đã suy giảm |

## Các mức trưởng thành của MLOps

Mô tả phổ biến (theo cách Google chia):

| Mức | Đặc điểm | Hệ quả |
|---|---|---|
| **0: thủ công** | Notebook, huấn luyện và triển khai thủ công, mô hình đưa "qua rào" cho kỹ sư triển khai | Hiếm khi cập nhật, khó tái lập, dễ lệch |
| **1: tự động hoá pipeline huấn luyện** | Pipeline tự động, **huấn luyện liên tục (CT)** khi có dữ liệu mới, kiểm tra dữ liệu và mô hình tự động | Cập nhật thường xuyên, tái lập được |
| **2: tự động hoá CI/CD cho pipeline** | Kiểm thử, xây dựng và triển khai chính pipeline tự động; thí nghiệm nhanh, an toàn | Lặp nhanh ở quy mô nhiều mô hình, nhiều nhóm |

Không phải dự án nào cũng cần mức 2: nhiều mô hình quy mô nhỏ ở mức 1 là đủ. Chọn mức theo tần suất thay đổi và hậu quả khi sai.

## Nguyên tắc cốt lõi

1. **Tái lập được**: cùng mã, dữ liệu, cấu hình, môi trường cho cùng kết quả (hoặc sai khác giải thích được).
2. **Phiên bản hoá mọi thứ**: mã, dữ liệu, tham số, mô hình, môi trường.
3. **Tự động hoá** pipeline từ dữ liệu tới triển khai; thủ công chỉ ở nơi có chủ đích (duyệt).
4. **Kiểm thử cả dữ liệu và mô hình**, không chỉ mã.
5. **Một đường cho huấn luyện và phục vụ**: dùng cùng mã đặc trưng để tránh lệch.
6. **Giám sát liên tục** dữ liệu và dự đoán; mô hình suy giảm lặng lẽ.
7. **Phát hành dần và quay lui được** như mọi hệ thống sản xuất ([Chiến lược triển khai](#deploy-strategies)).
8. **Trách nhiệm và minh bạch**: tài liệu mô hình, kiểm soát thiên lệch, nhật ký kiểm toán ([Quản trị](#governance)).
9. **Bắt đầu đơn giản**: mô hình đơn giản trong hệ thống tốt thắng mô hình phức tạp trong hệ thống tệ; thường nên có đường cơ sở dễ (quy tắc, hồi quy logistic).

## Vai trò

| Vai trò | Trọng tâm |
|---|---|
| Nhà khoa học dữ liệu | Bài toán, đặc trưng, mô hình, đánh giá |
| Kỹ sư học máy (ML engineer) | Biến mô hình thành pipeline và dịch vụ chạy được, tối ưu |
| Kỹ sư dữ liệu | Dữ liệu và đặc trưng đáng tin cậy ([DataOps](https://tracviethustcs.github.io/dataops-notes/)) |
| Kỹ sư nền tảng/DevOps | Hạ tầng, CI/CD, giám sát, bảo mật |
| Chủ sản phẩm, pháp lý, rủi ro | Mục tiêu, ràng buộc, tuân thủ |

Ranh giới mờ và khác nhau theo tổ chức; điều quan trọng là có chủ sở hữu rõ cho từng phần của hệ thống.

## So với DevOps và DataOps

| | DevOps | DataOps | MLOps |
|---|---|---|---|
| Thứ được quản lý | Mã | Dữ liệu và đường ống | Mã, dữ liệu **và mô hình** |
| Cách hỏng | Lỗi mã, hạ tầng | Dữ liệu xấu, trễ | Cả hai, cộng **suy giảm theo thời gian** |
| Kiểm thử | Mã | Dữ liệu | Dữ liệu, mô hình, hành vi |
| Phát hành | Phiên bản ứng dụng | Biến đổi, bảng | Mô hình kèm đặc trưng, có đánh giá |

::: lab Phân tích một hệ thống học máy
Chọn một ví dụ (dự đoán khách rời bỏ, phát hiện gian lận, đề xuất sản phẩm). Trả lời:

1. Mục tiêu nghiệp vụ và **chỉ số thành công** là gì? Sai thì ai chịu hậu quả?
2. **Dữ liệu** đến từ đâu, nhãn có sau bao lâu (ngay, hay vài tuần)?
3. Điều gì có thể **thay đổi** theo thời gian khiến mô hình suy giảm (hành vi, mùa vụ, sản phẩm mới)?
4. Bạn sẽ **phát hiện** suy giảm bằng cách nào khi nhãn đến trễ?
5. Mức MLOps hiện tại của một dự án bạn biết là 0, 1 hay 2, và bước nâng cấp rẻ nhất, có ích nhất là gì?
:::

=== data-features | Nền tảng | Dữ liệu và đặc trưng cho học máy
>> Dữ liệu và đặc trưng quyết định chất lượng mô hình nhiều hơn thuật toán. Hai nguy cơ lớn nhất không nằm ở mô hình: rò rỉ dữ liệu và lệch giữa huấn luyện với phục vụ.

## Phiên bản hoá dữ liệu

Mã nằm trong Git, nhưng dữ liệu quá lớn cho Git. Cần cách gắn **đúng phiên bản dữ liệu** với đúng commit mã và mô hình.

**DVC (Data Version Control)** lưu **con trỏ** nhỏ trong Git và dữ liệu thật ở kho từ xa (S3, GCS, SSH...):

```
git init && dvc init
dvc add data/train.csv                  # tạo data/train.csv.dvc (con trỏ, commit vào Git) và bỏ tệp lớn khỏi Git
git add data/train.csv.dvc data/.gitignore && git commit -m "thêm dữ liệu huấn luyện v1"
dvc remote add -d storage s3://my-bucket/dvc
dvc push                                 # đẩy dữ liệu lên kho từ xa
# ở máy khác, hoặc commit cũ:
git checkout <commit> && dvc pull        # lấy đúng dữ liệu của commit đó
```

Các cách khác: bảng có **time travel** (Delta, Iceberg: truy vấn bảng tại một thời điểm), **snapshot bất biến** trong object storage với đường dẫn có phiên bản, hoặc lưu mã băm dữ liệu cùng thí nghiệm. Quy tắc chung: mỗi mô hình phải truy ngược được tới **đúng dữ liệu đã huấn luyện nó**.

## Kiểm tra dữ liệu trước khi huấn luyện

Huấn luyện trên dữ liệu xấu cho mô hình xấu mà không báo lỗi. Kiểm tra tự động trước mỗi lần huấn luyện:

| Kiểm tra | Ví dụ |
|---|---|
| **Schema** | Đủ cột, đúng kiểu |
| **Miền giá trị** | Tuổi trong 0..120, giá không âm |
| **Thiếu dữ liệu** | Tỷ lệ NULL không vượt ngưỡng |
| **Phân phối** | So với bộ dữ liệu huấn luyện trước: có lệch mạnh không |
| **Nhãn** | Tỷ lệ các lớp hợp lý, không có nhãn rỗng |
| **Trùng lặp** | Dòng trùng, rò rỉ giữa tập huấn luyện và kiểm thử |

Công cụ: Pandera, Great Expectations, TensorFlow Data Validation, Evidently. Xem thêm [Chất lượng dữ liệu](https://tracviethustcs.github.io/dataops-notes/#data-quality).

## Chia dữ liệu và rò rỉ

| Tập | Dùng để |
|---|---|
| **Huấn luyện (train)** | Học tham số |
| **Kiểm định (validation)** | Chọn siêu tham số, so sánh mô hình |
| **Kiểm thử (test)** | Đánh giá cuối cùng, **dùng ít, giữ nguyên** |

**Rò rỉ dữ liệu (data leakage)** là khi thông tin từ tương lai hoặc từ tập kiểm thử lọt vào huấn luyện, làm điểm đánh giá đẹp giả tạo rồi sụp đổ khi triển khai. Các dạng phổ biến:

- **Rò rỉ theo thời gian**: chia ngẫu nhiên dữ liệu chuỗi thời gian, để mô hình "nhìn thấy tương lai". Với dữ liệu thời gian phải chia **theo thời gian** (huấn luyện trên quá khứ, kiểm thử trên tương lai).
- **Đặc trưng chứa đáp án**: đặc trưng chỉ có sau khi biết nhãn (ví dụ "số lần gọi hỗ trợ chấm dứt hợp đồng" khi dự đoán khách rời bỏ).
- **Tiền xử lý trên toàn bộ dữ liệu**: chuẩn hoá hoặc điền thiếu bằng thống kê tính trên cả tập kiểm thử. Phải **fit trên train, rồi áp dụng** cho validation/test.
- **Trùng giữa các tập**: cùng một người/đối tượng xuất hiện ở cả hai tập.

```
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Pipeline đảm bảo scaler chỉ fit trên dữ liệu train (kể cả trong cross-validation)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

## Lệch giữa huấn luyện và phục vụ (training-serving skew)

Đây là nguyên nhân hàng đầu khiến mô hình tốt ngoại tuyến lại kém khi chạy thật: đặc trưng được **tính khác nhau** ở hai nơi (huấn luyện bằng SQL/pandas trên kho, phục vụ bằng mã khác trong dịch vụ). Giải pháp:

- **Một định nghĩa đặc trưng, dùng ở cả hai nơi**: đóng gói tiền xử lý vào chính mô hình (pipeline), hoặc dùng **feature store**.
- Kiểm tra bằng cách ghi lại **đặc trưng thật lúc phục vụ** rồi so với đặc trưng tính ngoại tuyến cho cùng thực thể.
- Kiểm thử tương đương: cùng đầu vào cho cùng đặc trưng ở hai đường.

## Feature store

Feature store là nơi **định nghĩa, tính, lưu và phục vụ đặc trưng** cho cả huấn luyện lẫn suy luận.

```ascii
   Nguồn dữ liệu --> định nghĩa đặc trưng (một lần) --+--> KHO NGOẠI TUYẾN (offline): lịch sử lớn -> huấn luyện
                                                      |        get_historical_features (point-in-time đúng)
                                                      +--> KHO TRỰC TUYẾN (online): giá trị mới nhất, độ trễ ms -> suy luận
                                                               get_online_features
```

| Thành phần | Vai trò |
|---|---|
| **Kho ngoại tuyến** | Lưu lịch sử đặc trưng để dựng tập huấn luyện |
| **Kho trực tuyến** | Phục vụ giá trị mới nhất với độ trễ thấp |
| **Point-in-time join** | Khi dựng tập huấn luyện, lấy giá trị đặc trưng **đúng tại thời điểm** của mỗi mẫu, không lấy giá trị về sau (chặn rò rỉ thời gian) |
| **Tái sử dụng** | Nhiều mô hình dùng chung một đặc trưng đã kiểm thử |

Công cụ: Feast (mã nguồn mở), các feature store của Databricks, SageMaker, Vertex AI, Tecton. Nó không bắt buộc: dự án nhỏ có thể đóng gói tiền xử lý trong pipeline của mô hình và đủ.

## Gán nhãn và chất lượng nhãn

- Nhãn sai làm trần chất lượng của mô hình. Đo **độ đồng thuận giữa người gán nhãn**, rà soát mẫu.
- **Nhãn đến trễ** (khách rời bỏ biết sau 3 tháng): ảnh hưởng cách đánh giá và giám sát ([Giám sát](#monitoring-drift)).
- Vòng phản hồi: nhãn có thể bị ảnh hưởng bởi chính mô hình (chỉ biết kết quả của người được duyệt vay).

::: lab Rò rỉ do chọn đặc trưng trên toàn bộ dữ liệu
```
python3 - <<'EOF'
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 2000))          # 2000 đặc trưng HOÀN TOÀN là nhiễu
y = rng.integers(0, 2, size=100)          # nhãn ngẫu nhiên: không có gì để học, điểm đúng phải gần 0,5

# SAI: chọn 20 đặc trưng "tốt nhất" bằng TOÀN BỘ dữ liệu (kể cả phần sẽ dùng để kiểm thử), rồi mới cross-validate
X_leak = SelectKBest(f_classif, k=20).fit_transform(X, y)
leak = cross_val_score(LogisticRegression(max_iter=1000), X_leak, y, cv=5).mean()

# ĐÚNG: việc chọn đặc trưng nằm TRONG pipeline, nên chỉ fit trên phần train của từng fold
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
good = cross_val_score(pipe, X, y, cv=5).mean()

print("rò rỉ (SAI) :", round(leak, 3), " <- đẹp một cách vô lý trên dữ liệu ngẫu nhiên")
print("pipeline (ĐÚNG):", round(good, 3), " <- gần mức ngẫu nhiên (0,5; dao động theo mẫu)")
EOF
```
Hai con số khác nhau thế nào, và vì sao điểm "rò rỉ" cao dù dữ liệu hoàn toàn ngẫu nhiên? Lập luận tương tự áp dụng cho chuẩn hoá, điền thiếu, mã hoá theo trung bình nhãn. Viết ba cách rò rỉ khác bạn có thể gặp trong dự án của mình.
:::

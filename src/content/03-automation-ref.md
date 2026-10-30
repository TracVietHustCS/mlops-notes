=== ci-cd-ct | Tự động hoá | CI/CD/CT và kiểm thử học máy
>> Hệ thống học máy cần ba vòng tự động hoá: CI kiểm tra mã và dữ liệu, CD triển khai mô hình và dịch vụ, và CT (huấn luyện liên tục) làm mới mô hình khi dữ liệu thay đổi. Kèm theo là bộ kiểm thử riêng cho dữ liệu và mô hình.

## Ba vòng tự động hoá

| Vòng | Kích hoạt | Làm gì | Sản phẩm |
|---|---|---|---|
| **CI** (Continuous Integration) | Commit, pull request | Kiểm tra mã, kiểm thử đơn vị, kiểm tra dữ liệu mẫu, huấn luyện thử nhỏ | PR xanh hoặc đỏ |
| **CT** (Continuous Training) | Lịch, dữ liệu mới, drift, hiệu năng giảm | Chạy pipeline huấn luyện đầy đủ, đánh giá, đăng ký nếu qua cổng | Phiên bản mô hình mới trong registry |
| **CD** (Continuous Delivery/Deployment) | Mô hình mới đạt cổng, hoặc mã dịch vụ đổi | Đóng gói, triển khai dần (shadow, canary), quay lui nếu xấu | Mô hình/dịch vụ chạy ở production |

```ascii
   CODE đổi --> CI: lint, test mã, test dữ liệu mẫu, train thử ---> merge
                                                                      |
   DỮ LIỆU mới / lịch / drift --> CT: train đầy đủ -> đánh giá -> CỔNG -> đăng ký phiên bản mới
                                                                      |
                                                  CD: shadow -> canary -> champion; giám sát; quay lui
```

Điểm khác với phần mềm thường: **ba thứ có thể kích hoạt** (mã, dữ liệu, thời gian/drift), và một phiên bản triển khai là bộ ba **mã + dữ liệu + mô hình**, không chỉ mã.

## Các loại kiểm thử trong ML

| Nhóm | Kiểm tra | Ví dụ |
|---|---|---|
| **Mã** | Hàm tiền xử lý, đặc trưng chạy đúng | Kiểm thử đơn vị với đầu vào mẫu nhỏ |
| **Dữ liệu** | Schema, miền, thiếu, phân phối, nhãn | Pandera, Great Expectations ([Dữ liệu và đặc trưng](#data-features)) |
| **Pipeline** | Cả pipeline chạy được từ đầu đến cuối trên dữ liệu nhỏ | "Smoke test" huấn luyện vài giây |
| **Mô hình: chỉ số** | Đạt ngưỡng, không tệ hơn champion | Cổng chất lượng ([Đánh giá](#evaluation)) |
| **Mô hình: lát cắt, công bằng** | Không quá tệ ở nhóm nào | Đánh giá theo lát cắt |
| **Mô hình: hành vi** | Bất biến, hướng mong đợi, ca biên, hồi quy | Kiểm thử hành vi |
| **Xác định/tái lập** | Cùng hạt giống cho cùng kết quả | Chạy hai lần so sánh |
| **Phục vụ** | API đúng schema, độ trễ, đầu vào sai | Kiểm thử API, kiểm thử tải |
| **Tương thích** | Mô hình tải được ở môi trường phục vụ; đặc trưng khớp | Tải mô hình trong container phục vụ |

```
# tests/test_model_contract.py
import joblib, numpy as np, pytest

@pytest.fixture(scope="module")
def model():
    return joblib.load("models/model.joblib")

def test_dau_ra_la_xac_suat_hop_le(model):
    p = model.predict_proba(np.random.RandomState(0).randn(50, 10))[:, 1]
    assert ((p >= 0) & (p <= 1)).all()

def test_xac_dinh_voi_cung_dau_vao(model):
    x = np.random.RandomState(1).randn(20, 10)
    assert np.allclose(model.predict_proba(x), model.predict_proba(x))

def test_khong_te_hon_duong_co_so():
    import json
    m = json.load(open("metrics.json"))
    assert m["auc"] > 0.65                                  # phải thắng mô hình ngây thơ rõ rệt
```

## Workflow CI cho dự án ML

```
name: ml-ci
on:
  pull_request:
    paths: ["src/**", "tests/**", "params.yaml", "dvc.yaml", "requirements.txt"]

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12", cache: pip}
      - run: pip install -r requirements.txt
      - run: ruff check src tests                          # lint
      - run: pytest tests/unit -q                          # mã và đặc trưng
      - run: python src/validate_data.py data/sample.csv   # kiểm tra dữ liệu mẫu nhỏ
      - run: dvc repro --single-item smoke_train           # huấn luyện thử vài giây trên mẫu nhỏ
      - run: pytest tests/model -q                         # kiểm thử mô hình vừa huấn luyện thử
```

Giữ CI **nhanh** (vài phút): kiểm tra trên **mẫu nhỏ**, không huấn luyện đầy đủ. Huấn luyện đầy đủ thuộc về CT.

## CT: huấn luyện liên tục an toàn

- **Kích hoạt** theo lịch, dữ liệu mới hoặc cảnh báo drift ([Giám sát](#monitoring-drift)).
- **Kiểm tra dữ liệu trước**, dừng nếu xấu.
- **Đánh giá và cổng** so với champion; **không tự động triển khai** thứ chưa qua cổng.
- Mô hình mới thành **challenger** trong registry, rồi triển khai dần ([Chiến lược triển khai](#deploy-strategies)).
- Với rủi ro cao: **bước duyệt của con người** giữa CT và CD.
- Ghi lại mọi thứ vào theo dõi thí nghiệm để truy vết.
- Cẩn thận **vòng phản hồi** và **học lại trên dữ liệu nhiễm lỗi hoặc bị đầu độc**.

## CD cho mô hình

| Việc | Cách làm |
|---|---|
| Đóng gói | Container có mô hình hoặc dịch vụ tải theo bí danh (`@champion`) |
| Triển khai | GitOps: cập nhật tham chiếu phiên bản trong repo cấu hình ([GitOps](https://tracviethustcs.github.io/cicd-release-notes/#gitops-cd)) |
| Phát hành dần | Shadow, canary với cổng tự động |
| Quay lui | Đổi con trỏ phiên bản; giữ bản cũ sẵn sàng |
| Kiểm tra sau triển khai | Smoke test dự đoán, kiểm tra sức khoẻ, so sánh với champion |

Mã dịch vụ phục vụ (ít đổi) và **mô hình** (đổi thường xuyên) nên có **vòng đời tách nhau**: cập nhật mô hình không cần triển khai lại mã dịch vụ.

## Chỉ số của quy trình MLOps

| Chỉ số | Cho biết |
|---|---|
| Thời gian từ dữ liệu mới đến mô hình mới ở production | Tốc độ làm mới |
| Số lần huấn luyện lại thành công / thất bại, lý do | Độ tin cậy pipeline |
| Thời gian phát hiện suy giảm và thời gian khắc phục | Mức nhạy của giám sát |
| Tỷ lệ mô hình bị quay lui | Chất lượng cổng và triển khai |
| Tỷ lệ mô hình có tài liệu, người chịu trách nhiệm, giám sát | Mức trưởng thành |

::: lab Bộ kiểm thử hợp đồng cho mô hình
```
mkdir -p ~/mltest/tests && cd ~/mltest
python3 - <<'EOF'
import joblib, json
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
X, y = make_classification(n_samples=1500, n_features=10, n_informative=4, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
m = RandomForestClassifier(n_estimators=80, random_state=0).fit(Xtr, ytr)
joblib.dump(m, "model.joblib"); json.dump({"auc": roc_auc_score(yte, m.predict_proba(Xte)[:, 1])}, open("metrics.json", "w"))
EOF
cat > tests/test_model.py <<'EOF'
import joblib, json, numpy as np
m = joblib.load("model.joblib")
def test_xac_suat_hop_le():
    p = m.predict_proba(np.random.RandomState(0).randn(50, 10))[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
def test_xac_dinh():
    x = np.random.RandomState(1).randn(20, 10)
    assert np.allclose(m.predict_proba(x), m.predict_proba(x))
def test_thang_duong_co_so():
    assert json.load(open("metrics.json"))["auc"] > 0.7
def test_bat_bien_voi_dac_trung_khong_dung_den():
    x = np.random.RandomState(2).randn(30, 10)
    y = x.copy(); y[:, 9] += 5          # đặc trưng f9 có thể ít quan trọng: kiểm tra mức nhạy
    d = np.abs(m.predict_proba(x)[:, 1] - m.predict_proba(y)[:, 1]).mean()
    assert d < 0.5
EOF
pip install -q pytest && pytest -q
```
Bài tập: thêm một kiểm thử mà bạn tin sẽ bắt được lỗi "huấn luyện nhầm dữ liệu rỗng" hoặc "đặc trưng đổi thứ tự cột". Kiểm thử nào của bạn sẽ đỏ nếu ai đó vô tình đổi thứ tự cột đầu vào?
:::

=== infra-cost | Tự động hoá | Hạ tầng, GPU và chi phí
>> Học máy tiêu tốn tài nguyên theo cách khác dịch vụ web: huấn luyện nặng và gián đoạn, suy luận cần phần cứng đặc biệt. Quản lý hạ tầng và chi phí tốt quyết định dự án có bền vững không.

## Huấn luyện và suy luận khác nhau

| | Huấn luyện | Suy luận (phục vụ) |
|---|---|---|
| Tải | Nặng, **gián đoạn** (chạy hàng giờ rồi dừng) | Liên tục hoặc theo lưu lượng |
| Ưu tiên | Thông lượng, tận dụng phần cứng | Độ trễ, độ sẵn sàng |
| Tài nguyên | CPU/GPU lớn, bộ nhớ lớn, lưu trữ dữ liệu | Thường nhỏ hơn, đôi khi cần GPU |
| Co giãn | Dựng khi cần rồi tắt | Co giãn theo lưu lượng |
| Chịu lỗi | Checkpoint để tiếp tục | Nhiều bản sao, health check |

Nhiều mô hình cổ điển (cây quyết định, hồi quy) chạy tốt trên CPU. **Đừng mặc định dùng GPU**: chỉ khi mô hình học sâu thật sự cần.

## GPU trên Kubernetes

Kubernetes quản lý GPU qua **device plugin** (ví dụ của NVIDIA), khai báo như một loại tài nguyên:

```
apiVersion: v1
kind: Pod
metadata: {name: train-job}
spec:
  restartPolicy: Never
  nodeSelector: {accelerator: nvidia-gpu}
  tolerations:
    - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
  containers:
    - name: train
      image: registry.example.com/ml/train:1.2.0
      resources:
        limits:
          nvidia.com/gpu: 1            # GPU chỉ khai báo trong limits, và là số nguyên
          memory: 32Gi
        requests:
          cpu: "8"
          memory: 32Gi
```

- GPU thường **không chia nhỏ mặc định** (một Pod một GPU nguyên); có kỹ thuật chia sẻ (time-slicing, MIG) với đánh đổi riêng.
- Đặt **taint** cho node GPU và `toleration` cho Pod cần GPU để workload thường không chiếm máy đắt tiền ([Scheduling](https://tracviethustcs.github.io/k8s-study-notes/#scheduling)).
- **Job** cho huấn luyện (chạy xong là giải phóng), **Deployment** cho phục vụ ([Job, CronJob](https://tracviethustcs.github.io/k8s-study-notes/#obj-batch)).
- Đặt `requests` và `limits` bộ nhớ hợp lý để tránh OOM giữa chừng ([Tài nguyên](https://tracviethustcs.github.io/k8s-study-notes/#ops-resources)).

## Giảm chi phí huấn luyện

| Kỹ thuật | Mô tả | Đánh đổi |
|---|---|---|
| **Spot / preemptible** | Dùng máy giá rẻ có thể bị thu hồi bất kỳ lúc | Cần **checkpoint** và khả năng tiếp tục |
| **Checkpointing** | Lưu trạng thái định kỳ để tiếp tục khi bị ngắt | Thêm I/O |
| **Dừng sớm** | Dừng huấn luyện khi chỉ số không cải thiện | Cần tập kiểm định đáng tin |
| **Thí nghiệm nhỏ trước** | Thử trên mẫu và mô hình nhỏ rồi mới mở rộng | Phải đại diện đủ |
| **Mixed precision, tối ưu batch** | Tận dụng GPU tốt hơn | Cần kiểm tra ổn định số học |
| **Cache đặc trưng, tiền xử lý** | Không tính lại | Quản lý cache |
| **Tắt tài nguyên rảnh** | Tự động tắt cụm, notebook bỏ quên | Kỷ luật vận hành |
| **Chọn kích thước phù hợp** | Mô hình lớn hơn không phải lúc nào cũng đáng | Cần đo lợi ích |

## Giảm chi phí suy luận

- **Batch thay online** khi độ trễ không bắt buộc.
- **Lượng tử hoá, chưng cất mô hình, ONNX** để chạy rẻ hơn, nhanh hơn.
- **Co giãn theo lưu lượng** và co về nhỏ khi vắng; **gộp nhiều mô hình lên một máy** nếu phù hợp.
- **Cache dự đoán** cho đầu vào lặp lại.
- **Mô hình nhỏ làm bộ lọc trước**, mô hình lớn chỉ xử lý ca khó.

## Môi trường và tái lập

- **Container hoá** môi trường huấn luyện và phục vụ, ghim phiên bản thư viện và driver (CUDA), vì lệch phiên bản gây lỗi khó đoán ([Docker](https://tracviethustcs.github.io/docker-container-notes/)).
- Mô tả hạ tầng bằng mã (Terraform, Helm) để dựng lại ([IaC](https://tracviethustcs.github.io/iac-terraform-ansible-notes/)).
- **Dữ liệu gần tính toán** (cùng vùng) để tránh phí truyền và độ trễ.

## Theo dõi và kiểm soát chi phí

| Biện pháp | Mô tả |
|---|---|
| **Gắn thẻ chi phí** | Theo dự án, mô hình, người, môi trường |
| **Ngân sách và cảnh báo** | Cảnh báo khi vượt, giới hạn cứng cho thí nghiệm |
| **Đo chi phí mỗi lần huấn luyện, mỗi dự đoán** | Biết đơn vị chi phí để quyết định đánh đổi |
| **Hạn mức theo nhóm** | Quota GPU, ResourceQuota ([Tài nguyên](https://tracviethustcs.github.io/k8s-study-notes/#ops-resources)) |
| **Rà soát tài nguyên bỏ quên** | Notebook, máy GPU, điểm cuối suy luận không còn dùng |

Xem thêm về công cụ như OpenCost ([Giám sát và add-on](https://tracviethustcs.github.io/k8s-study-notes/#tool-observe)).

::: lab Ước tính chi phí và quyết định batch hay online
Điền số giả định cho một mô hình của bạn rồi so sánh:

| | Online | Batch |
|---|---|---|
| Số dự đoán mỗi ngày | 2.000.000 | 2.000.000 |
| Tài nguyên | 2 máy luôn bật (24/7) | 1 máy chạy 1 giờ mỗi đêm |
| Chi phí máy mỗi giờ (giả định) | 0,5 | 0,5 |

1. Tính chi phí máy mỗi ngày của hai cách. Chi phí mỗi triệu dự đoán là bao nhiêu?
2. Với bài toán nào độ trễ vài giờ là chấp nhận được (khiến batch hợp lý)? Với bài toán nào thì không?
3. Bạn sẽ dùng spot cho huấn luyện nhưng không dùng cho phục vụ online: vì sao, và cần chuẩn bị gì?
:::

=== governance | Tự động hoá | Quản trị mô hình và học máy có trách nhiệm
>> Mô hình đưa ra quyết định ảnh hưởng đến con người. Quản trị bảo đảm mô hình được ghi chép, kiểm soát, giám sát về thiên lệch, và có người chịu trách nhiệm khi nó sai.

::: warn Đây không phải tư vấn pháp lý
Quy định về AI và dữ liệu (như đạo luật AI của EU phân loại theo mức rủi ro, các luật bảo vệ dữ liệu cá nhân) đang phát triển và khác nhau theo khu vực và lĩnh vực. Nội dung dưới đây là khung kỹ thuật và quy trình tham khảo; hãy làm việc với bộ phận pháp lý, tuân thủ và rủi ro của tổ chức cho yêu cầu cụ thể.
:::

## Tài liệu: model card và datasheet

| Tài liệu | Nội dung |
|---|---|
| **Model card** | Mục đích sử dụng và **không** nên dùng; dữ liệu huấn luyện; chỉ số tổng và theo lát cắt; hạn chế đã biết; cân nhắc đạo đức; người chịu trách nhiệm |
| **Datasheet cho tập dữ liệu** | Nguồn gốc, cách thu thập, gán nhãn, khoảng thời gian, thiên lệch đã biết, giấy phép, quyền riêng tư |

Một model card ngắn nhưng trung thực đáng giá hơn tài liệu dài không ai đọc. Đặt nó **cạnh mô hình trong registry**, cập nhật mỗi phiên bản.

```
# Model card (rút gọn)
Tên: churn-model v7                 Chủ sở hữu: nhóm Khách hàng (an@example.com)
Mục đích: xếp hạng khách có nguy cơ rời bỏ để ưu tiên chăm sóc. KHÔNG dùng để quyết định giá hay từ chối dịch vụ.
Dữ liệu huấn luyện: bảng silver.customers_v3, 2025-01 đến 2026-06, đã loại khách thử nghiệm.
Chỉ số (test, thời gian): AUC 0,81 tổng; thấp nhất 0,74 ở nhóm khách mới (n=3.200).
Hạn chế: kém với khách dưới 1 tháng; chưa kiểm thử cho thị trường mới.
Giám sát: PSI hàng tuần; AUC hàng tháng khi nhãn về; kích hoạt học lại khi AUC < 0,76.
```

## Kiểm soát thiên lệch và công bằng

- Đo chỉ số **theo nhóm** (nếu phù hợp pháp lý và đạo đức) và theo dõi theo thời gian ([Đánh giá](#evaluation)).
- Xem xét **thiên lệch trong dữ liệu** (mẫu không đại diện, nhãn phản ánh quyết định thiên lệch trong quá khứ) vì mô hình học và có thể khuếch đại nó.
- **Đặc trưng đại diện (proxy)**: bỏ thuộc tính nhạy cảm chưa chắc hết thiên lệch nếu còn đặc trưng tương quan chặt (mã bưu chính).
- Chọn tiêu chí công bằng là **quyết định nghiệp vụ và đạo đức** có sự tham gia của người có thẩm quyền, không chỉ kỹ sư.

## Giải thích được

Tuỳ rủi ro, cần mức giải thích khác nhau:

| Công cụ/Cách | Ghi chú |
|---|---|
| Mô hình **tự giải thích** (tuyến tính, cây nhỏ) | Dễ giải thích nhất; đánh đổi một phần độ chính xác |
| **Tầm quan trọng đặc trưng**, SHAP, LIME | Giải thích hậu nghiệm; **là xấp xỉ**, không phải sự thật tuyệt đối, có thể gây hiểu lầm |
| **Mã lý do (reason codes)** | Bắt buộc ở một số lĩnh vực (từ chối tín dụng cần lý do) |
| **Có con người trong vòng lặp** | Với quyết định rủi ro cao, mô hình hỗ trợ chứ không tự quyết định |

## Quyền riêng tư và dữ liệu

- **Giảm thiểu dữ liệu** thu thập; che và ẩn danh sớm ([Quản trị dữ liệu](https://tracviethustcs.github.io/dataops-notes/#governance-security)).
- **Mô hình có thể ghi nhớ dữ liệu huấn luyện** (đặc biệt mô hình lớn); cẩn thận với dữ liệu cá nhân trong tập huấn luyện và đầu ra.
- **Yêu cầu xoá dữ liệu**: nếu dữ liệu của một người nằm trong tập huấn luyện, cần có quy trình (kể cả học lại mô hình) theo yêu cầu pháp lý.
- Kiểm soát truy cập vào dữ liệu và mô hình.

## Kiểm toán và truy vết

Với mỗi dự đoán quan trọng cần có thể trả lời: **phiên bản mô hình nào, đặc trưng đầu vào nào, vào lúc nào, ai/hệ thống nào dùng kết quả**. Điều này cần:

- Registry với lineage đầy đủ ([Registry](#registry)).
- Log dự đoán có kiểm soát ([Giám sát](#monitoring-drift)).
- Nhật ký thay đổi: ai đã nâng bản nào lên production và dựa trên cổng nào.

## Phân loại rủi ro và quy trình duyệt

Không phải mô hình nào cũng cần cùng mức kiểm soát. Một cách làm: **phân tầng theo tác động**.

| Mức | Ví dụ | Kiểm soát |
|---|---|---|
| Thấp | Gợi ý nội dung, sắp xếp nội bộ | CI/CD tự động, giám sát cơ bản |
| Trung bình | Dự báo nhu cầu, chấm điểm khách hàng tiềm năng | Cổng đánh giá, model card, giám sát drift |
| Cao | Tín dụng, tuyển dụng, y tế, an toàn, pháp lý | Duyệt của con người, đánh giá thiên lệch, giải thích, kiểm toán, có thể cần tuân thủ quy định riêng |

## Bảo mật cho học máy

| Rủi ro | Biện pháp |
|---|---|
| **Đầu độc dữ liệu** (kẻ tấn công chèn dữ liệu huấn luyện xấu) | Kiểm soát nguồn dữ liệu, kiểm tra bất thường, cẩn thận với học liên tục từ dữ liệu người dùng |
| **Ví dụ đối nghịch** (đầu vào được chế tác để lừa mô hình) | Kiểm thử độ bền, xác thực đầu vào, lớp bảo vệ bổ sung |
| **Đánh cắp mô hình, suy ra dữ liệu huấn luyện** | Giới hạn tốc độ và truy vấn, xác thực, không lộ điểm chi tiết |
| **Chuỗi cung ứng** (mô hình/thư viện độc hại, pickle chạy mã) | Chỉ dùng nguồn tin cậy, định dạng an toàn, ký và xác minh ([Chuỗi cung ứng](https://tracviethustcs.github.io/k8s-study-notes/#adv-supply-chain)) |

::: lab Viết model card và kế hoạch giám sát cho mô hình của bạn
Chọn một mô hình thật hoặc giả định, viết một model card theo mẫu trên (dưới 20 dòng). Rồi trả lời:

1. Có ai bị ảnh hưởng nặng nếu mô hình sai không, và mô hình thuộc mức rủi ro nào?
2. Bạn đo hiệu năng theo **lát cắt** nào, và ngưỡng tối thiểu của lát cắt yếu nhất là gì?
3. Ai được **nâng mô hình lên production**, dựa trên cổng nào, và ghi ở đâu?
4. Nếu một khách hàng hỏi "vì sao tôi bị từ chối", bạn có trả lời được không, và bằng cách nào?
5. Dữ liệu cá nhân nào nằm trong tập huấn luyện, và quy trình khi có yêu cầu xoá là gì?
:::

=== lab-stack | Thực hành | Dự án mẫu cục bộ: huấn luyện, đăng ký, phục vụ, giám sát
>> Một vòng MLOps nhỏ chạy trên máy bạn: huấn luyện nhiều mô hình với MLflow, cổng đánh giá, đăng ký champion, phục vụ qua FastAPI, rồi mô phỏng drift và phát hiện nó. CPU, không cần cloud.

## Mục tiêu

Đi qua gần đủ vòng đời: **dữ liệu → thí nghiệm → cổng → registry → phục vụ → giám sát**, với mô hình nhỏ để chạy trong vài giây.

## Bước 1: Môi trường và dữ liệu

```
python3 -m venv ~/mlops-venv && . ~/mlops-venv/bin/activate
pip install scikit-learn pandas numpy scipy mlflow fastapi uvicorn httpx joblib pytest
mkdir -p ~/mlops-lab && cd ~/mlops-lab
python3 - <<'EOF'
import numpy as np, pandas as pd
rng = np.random.default_rng(11)
n = 6000
df = pd.DataFrame({
    "tenure_months": rng.integers(1, 72, n),
    "monthly_spend": rng.normal(60, 20, n).clip(5),
    "support_calls": rng.poisson(1.5, n),
})
logit = -1.0 - 0.04 * df.tenure_months + 0.35 * df.support_calls + 0.01 * (df.monthly_spend - 60)
df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df.to_csv("train.csv", index=False)
print(df.churn.mean().round(3), "tỷ lệ rời bỏ;", len(df), "dòng")
EOF
```

## Bước 2: Thí nghiệm, cổng chất lượng và đăng ký champion

```
cat > train.py <<'EOF'
import mlflow, mlflow.sklearn, pandas as pd, numpy as np, json
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlflow.set_tracking_uri("sqlite:///mlflow.db"); mlflow.set_experiment("churn")
df = pd.read_csv("train.csv"); X, y = df.drop(columns="churn"), df["churn"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

candidates = {
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "gbm": GradientBoostingClassifier(random_state=0),
}
best = None
for name, model in candidates.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(Xtr, ytr)
        auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
        mlflow.log_param("model", name); mlflow.log_metric("test_auc", auc)
        mlflow.set_tag("data_version", "train.csv-v1")
        mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")   # xem ghi chú bên dưới
        print(f"{name}: AUC={auc:.4f}")
        if best is None or auc > best[1]: best = (run.info.run_id, auc, name)

run_id, auc, name = best
client = MlflowClient()
GATE_MIN_AUC = 0.60
if auc < GATE_MIN_AUC:
    raise SystemExit(f"CỔNG: AUC {auc:.3f} < {GATE_MIN_AUC}, dừng, không đăng ký")
mv = mlflow.register_model(f"runs:/{run_id}/model", "churn-model")
client.set_registered_model_alias("churn-model", "champion", mv.version)
json.dump({"auc": auc, "model": name}, open("champion_metrics.json", "w"))
print("đã đăng ký", name, "phiên bản", mv.version, "thành champion")
EOF
python3 train.py
```

::: note Về định dạng lưu mô hình sklearn của MLflow
Các bản MLflow mới có thể mặc định lưu mô hình scikit-learn bằng định dạng **skops** (an toàn hơn pickle) và **từ chối loại "không tin cậy"**, trong đó có các mô hình cây (Random Forest, Gradient Boosting): `log_model` báo lỗi "untrusted types" trừ khi bạn khai báo rõ danh sách kiểu tin cậy (`skops_trusted_types`) sau khi đã xem xét. Trong lab này mô hình do chính bạn huấn luyện nên dùng `serialization_format="cloudpickle"` cho gọn. Với mô hình đến từ **nguồn bên ngoài**, hãy làm ngược lại: giữ định dạng an toàn, chỉ tin các kiểu bạn đã kiểm tra ([Registry](#registry), [Phục vụ](#serving)). Tên tham số và mặc định thay đổi theo phiên bản MLflow: đối chiếu tài liệu.
:::

## Bước 3: Phục vụ champion

```
cat > serve.py <<'EOF'
import mlflow, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/churn-model@champion")
app = FastAPI()
class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=600)
    monthly_spend: float = Field(ge=0)
    support_calls: int = Field(ge=0)
@app.get("/healthz")
def health(): return {"status": "ok"}
@app.post("/predict")
def predict(c: Customer):
    p = float(model.predict_proba(pd.DataFrame([c.model_dump()]))[0, 1])
    return {"churn_probability": round(p, 4)}
EOF
uvicorn serve:app --port 8000 >serve.log 2>&1 &
sleep 5
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"tenure_months":3,"monthly_spend":80,"support_calls":4}'; echo
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"tenure_months":60,"monthly_spend":50,"support_calls":0}'; echo
kill $!
```

## Bước 4: Giám sát drift

```
cat > monitor.py <<'EOF'
import numpy as np, pandas as pd
from scipy.stats import ks_2samp

def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

ref = pd.read_csv("train.csv")
rng = np.random.default_rng(99)
live = pd.DataFrame({                                       # "tuần này": nhiều khách mới và gọi hỗ trợ nhiều hơn
    "tenure_months": rng.integers(1, 20, 2000),
    "monthly_spend": rng.normal(62, 20, 2000).clip(5),
    "support_calls": rng.poisson(3.0, 2000),
})
for col in ["tenure_months", "monthly_spend", "support_calls"]:
    v = psi(ref[col].values, live[col].values)
    status = "ỔN ĐỊNH" if v < 0.1 else ("THAY ĐỔI VỪA" if v < 0.25 else "DRIFT ĐÁNG KỂ")
    print(f"{col:15s} PSI={v:.3f}  {status}")
EOF
python3 monitor.py
```

## Bước 5: Kiểm thử và thử phá

```
mkdir -p tests && cat > tests/test_lab.py <<'EOF'
import mlflow, pandas as pd, json
mlflow.set_tracking_uri("sqlite:///mlflow.db")
def test_champion_tai_duoc_va_du_doan_hop_le():
    m = mlflow.sklearn.load_model("models:/churn-model@champion")
    p = m.predict_proba(pd.DataFrame([{"tenure_months": 5, "monthly_spend": 70.0, "support_calls": 3}]))[0, 1]
    assert 0 <= p <= 1
def test_champion_qua_cong():
    assert json.load(open("champion_metrics.json"))["auc"] >= 0.60
EOF
pytest -q
```

Thử phá và quan sát:

1. Đặt `GATE_MIN_AUC = 0.99` rồi chạy lại `train.py`: điều gì xảy ra với registry? Champion cũ có bị đụng tới không?
2. Chạy `train.py` lần nữa và đổi một ứng viên: bí danh `champion` trỏ vào phiên bản nào, và dịch vụ đang chạy có tự biết không?
3. Trong `monitor.py`, đặc trưng nào drift mạnh nhất? Đó là drift thật hay lỗi pipeline (dựa trên cách bạn đã tạo dữ liệu)?
4. Gửi `{"tenure_months": -5, ...}` tới dịch vụ: nó trả về gì và vì sao đó là hành vi mong muốn?

## Câu hỏi mở rộng

- Bạn biến bước 2 thành **pipeline DVC** ([Pipeline](#pipelines)) và **job CI** ([CI/CD/CT](#ci-cd-ct)) thế nào?
- Khi nhãn "rời bỏ" chỉ có sau 3 tháng, bạn giám sát hiệu năng bằng cách nào ngoài PSI ([Giám sát](#monitoring-drift))?
- Bạn đóng gói `serve.py` thành container và triển khai canary 10% thế nào ([Phục vụ](#serving), [Chiến lược triển khai](#deploy-strategies))?
- Model card cho mô hình này sẽ ghi gì ở phần hạn chế ([Quản trị](#governance))?

=== cheatsheet | Tham chiếu | Cheat sheet MLOps
>> Tra nhanh khái niệm, lệnh và quy tắc theo nhu cầu.

## Lệnh và API hay dùng

| Muốn | Cách |
|---|---|
| Theo dõi một run | `mlflow.start_run()`, `mlflow.log_params()`, `mlflow.log_metric()`, `mlflow.sklearn.log_model(m, "model")` |
| Giao diện thí nghiệm | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| Đăng ký mô hình | `mlflow.register_model("runs:/RUN_ID/model", "ten-mo-hinh")` |
| Gắn bí danh | `MlflowClient().set_registered_model_alias("ten", "champion", version)` |
| Tải theo bí danh | `mlflow.pyfunc.load_model("models:/ten@champion")` |
| Phiên bản hoá dữ liệu | `dvc add data/x.csv`, `dvc push`, `dvc pull` |
| Chạy pipeline | `dvc repro`, `dvc exp run -S train.max_depth=8` |
| So chỉ số | `dvc metrics show`, `dvc metrics diff HEAD~1` |
| Phục vụ nhanh | `uvicorn serve:app --port 8000`, `mlflow models serve -m models:/ten@champion` |

## Nguyên tắc nhớ nhanh

| Chủ đề | Quy tắc |
|---|---|
| Ba thứ thay đổi | Mã, dữ liệu, mô hình: phiên bản hoá cả ba |
| Tái lập | Cùng commit + phiên bản dữ liệu + cấu hình + môi trường |
| Rò rỉ dữ liệu | Fit tiền xử lý trên train, chia thời gian theo thời gian, đặc trưng không chứa đáp án |
| Lệch huấn luyện/phục vụ | Một định nghĩa đặc trưng ở cả hai nơi (pipeline hoặc feature store) |
| Đánh giá | Đường cơ sở, so với champion, theo lát cắt, kiểm thử hành vi |
| Chỉ số | Gắn với nghiệp vụ; lớp lệch thì không dùng accuracy |
| Cổng | Bằng mã, trong pipeline; thất bại thì dừng, không tự triển khai |
| Triển khai | Shadow, canary, A/B; đổi con trỏ để quay lui |
| Giám sát | Dữ liệu vào, dự đoán, hiệu năng thật, nghiệp vụ, sức khoẻ dịch vụ |
| Drift | Loại trừ lỗi pipeline dữ liệu trước; PSI < 0,1 ổn, 0,1–0,25 vừa, > 0,25 đáng kể (quy ước) |
| Nhãn trễ | Dùng drift đầu vào và dự đoán làm cảnh báo sớm |
| CT | Kích hoạt theo lịch/dữ liệu/drift, luôn có kiểm tra và cổng |
| Hạ tầng | Đừng mặc định GPU; spot kèm checkpoint cho huấn luyện; batch nếu độ trễ cho phép |
| Quản trị | Model card, thiên lệch theo nhóm, truy vết, người chịu trách nhiệm |
| Bảo mật | Không tải mô hình pickle từ nguồn lạ; chống đầu độc dữ liệu; giới hạn truy vấn |

## Chọn kiểu phục vụ

| Nếu | Chọn |
|---|---|
| Độ trễ giờ/ngày chấp nhận được | **Batch** |
| Cần trả lời từng yêu cầu trong ms | **Online** |
| Xử lý luồng sự kiện liên tục | **Streaming** |

## Mức trưởng thành MLOps

| Mức | Đặc điểm |
|---|---|
| 0 | Thủ công, notebook, triển khai tay |
| 1 | Pipeline huấn luyện tự động, huấn luyện liên tục, kiểm tra dữ liệu và mô hình |
| 2 | CI/CD cho chính pipeline, nhiều mô hình và nhóm |

=== quiz-1 | Luyện tập | Luyện tập 1: Dữ liệu, thí nghiệm, registry và đánh giá
>> 6 câu tình huống về rò rỉ dữ liệu, tái lập, registry và đánh giá. Chọn đáp án rồi bấm "Kiểm tra" để xem giải thích. Kết quả được lưu trong trình duyệt.

## Câu hỏi

@quiz
Q: Mô hình dự báo nhu cầu cho AUC/độ chính xác rất cao ở ngoại tuyến nhưng kém hẳn khi triển khai. Dữ liệu là chuỗi thời gian và bạn đã chia train/test ngẫu nhiên. Nguyên nhân khả dĩ nhất?
- Mô hình quá nhỏ
+ Rò rỉ theo thời gian: chia ngẫu nhiên khiến mô hình "nhìn thấy tương lai"; phải chia theo thời gian (huấn luyện quá khứ, kiểm thử tương lai)
- Cần thêm GPU
- Cần tăng số cây
E: Với dữ liệu chuỗi thời gian, chia ngẫu nhiên làm các mẫu gần nhau về thời gian nằm ở cả train và test, tạo điểm đánh giá đẹp giả. Phải mô phỏng đúng cách dùng thật: học từ quá khứ để dự đoán tương lai.
Q: Cùng một mã, bạn chạy lại huấn luyện sau một tháng và nhận kết quả khác, không ai biết vì sao. Hai thực hành nào quan trọng nhất để tránh tình huống này?
- Chạy huấn luyện nhiều lần cho đến khi kết quả đẹp
+ Phiên bản hoá dữ liệu (DVC, snapshot) và ghi lại đầy đủ mã (commit), tham số, hạt giống, phiên bản thư viện trong theo dõi thí nghiệm
- Không thay đổi dữ liệu nữa
- Lưu mô hình dưới tên `model_final.pkl`
E: Tái lập cần cố định cả ba: mã, dữ liệu và môi trường/cấu hình. Theo dõi thí nghiệm ghi lại tất cả để mỗi mô hình truy ngược được tới điều kiện tạo ra nó.
Q: Chọn TẤT CẢ phát biểu đúng về rò rỉ dữ liệu và lệch huấn luyện/phục vụ.
+ Chuẩn hoá hoặc điền thiếu bằng thống kê tính trên toàn bộ dữ liệu (kể cả tập kiểm thử) gây rò rỉ
+ Đặc trưng chỉ biết được sau khi biết đáp án (ví dụ "đã huỷ hợp đồng") không được dùng khi dự đoán rời bỏ
- Dùng hai đoạn mã khác nhau để tính đặc trưng ở huấn luyện và phục vụ là an toàn miễn cùng ý tưởng
+ Đóng gói tiền xử lý vào pipeline của mô hình hoặc dùng feature store giúp một định nghĩa đặc trưng dùng ở cả hai nơi
E: Hai đường tính đặc trưng khác nhau là nguyên nhân hàng đầu của lệch huấn luyện/phục vụ, dù ý tưởng giống. Pipeline hoặc feature store bảo đảm cùng một logic ở cả hai nơi; tiền xử lý phải fit chỉ trên train.
Q: Mô hình mới có AUC tổng thể cao hơn champion 0,01, nhưng AUC ở nhóm khách hàng mới (5% dữ liệu) giảm mạnh. Quyết định hợp lý nhất?
- Triển khai ngay vì AUC tổng thể tốt hơn
+ Đánh giá theo lát cắt cho thấy rủi ro ở nhóm khách mới; áp dụng ngưỡng tối thiểu theo lát cắt, điều tra nguyên nhân, và chỉ triển khai khi chấp nhận được (hoặc triển khai kèm biện pháp bảo vệ nhóm đó)
- Bỏ qua vì nhóm đó nhỏ
- Xoá nhóm khách mới khỏi tập đánh giá
E: Điểm tổng thể có thể che mô hình rất tệ ở một nhóm nhỏ nhưng quan trọng. Cổng chất lượng nên có ngưỡng theo lát cắt, và quyết định chấp nhận rủi ro cần có người chịu trách nhiệm.
Q: Dịch vụ phục vụ tải mô hình qua `models:/churn-model@champion`. Mô hình mới vừa được xác nhận tệ hơn dự kiến sau khi nâng bí danh. Cách quay lui nhanh nhất?
- Huấn luyện lại mô hình cũ từ đầu
+ Đổi bí danh `champion` về phiên bản trước trong registry (và để dịch vụ tải lại hoặc khởi động lại); không cần triển khai lại mã
- Xoá registry rồi dựng lại
- Sửa mã dịch vụ để trỏ vào tệp cũ
E: Bí danh là con trỏ tới phiên bản; quay lui chỉ là đổi con trỏ. Điều này đòi hỏi giữ lại phiên bản cũ và dịch vụ biết tải lại khi bí danh đổi.
Q: Một đồng nghiệp đề xuất tải mô hình do người ngoài chia sẻ dưới dạng tệp pickle về chạy thẳng trên máy chủ phục vụ. Rủi ro chính và cách xử lý?
- Không có rủi ro vì đó chỉ là dữ liệu
+ Pickle có thể thực thi mã tuỳ ý khi tải; chỉ tải từ nguồn tin cậy, ưu tiên định dạng an toàn hơn (ONNX, safetensors), ký và xác minh artifact, chạy trong môi trường bị giới hạn quyền
- Rủi ro duy nhất là tệp lớn
- Chỉ cần đổi đuôi tệp
E: Định dạng pickle là mã Python được tuần tự hoá: tải nó có thể chạy lệnh của kẻ tấn công. Là một phần của chuỗi cung ứng mô hình cần được kiểm soát.

=== quiz-2 | Luyện tập | Luyện tập 2: Phục vụ, triển khai, drift và CI/CD/CT
>> 6 câu về triển khai, giám sát, CI/CD/CT và hạ tầng.

## Câu hỏi

@quiz
Q: Mô hình mới tốt hơn champion ngoại tuyến, nhưng chưa ai biết nó phản ứng thế nào với traffic thật và bạn không muốn rủi ro với người dùng. Bước tiếp theo an toàn nhất?
- Thay thế champion ngay
+ Chạy shadow: mô hình mới nhận bản sao yêu cầu thật nhưng dự đoán không được dùng, so sánh với champion; sau đó canary với ngưỡng dừng định trước
- Chỉ chạy thêm kiểm thử ngoại tuyến
- Triển khai cho toàn bộ người dùng vào cuối tuần
E: Shadow cho thấy hành vi trên traffic thật mà không ảnh hưởng người dùng; canary rồi giới hạn rủi ro và đo tác động thật. Cả hai cần chỉ số và ngưỡng quyết định đặt trước.
Q: Nhãn "khách rời bỏ" chỉ có sau 3 tháng. Làm sao biết sớm mô hình đang suy giảm?
- Chờ 3 tháng rồi tính AUC
+ Giám sát drift đầu vào và phân phối dự đoán làm cảnh báo sớm, theo dõi chỉ số nghiệp vụ, và xác nhận bằng hiệu năng thật khi nhãn về
- Không thể biết trước
- Tăng tần suất huấn luyện lên mỗi ngày
E: Khi nhãn đến trễ, drift của đặc trưng và dự đoán là tín hiệu không cần nhãn. Hiệu năng thật xác nhận về sau. Học lại thật nhiều không thay thế giám sát.
Q: PSI của đặc trưng `account_age` đột ngột lên 0,9. Việc đầu tiên nên làm?
- Học lại mô hình ngay
+ Kiểm tra xem đây là lỗi pipeline dữ liệu (đơn vị đổi, cột thiếu/đổi nguồn) hay drift thật; trước khi học lại phải loại trừ nguyên nhân dữ liệu
- Bỏ đặc trưng đó khỏi mô hình luôn
- Tắt cảnh báo drift
E: Rất nhiều "drift" là lỗi ở nguồn dữ liệu hay tính toán đặc trưng. Học lại trên dữ liệu lỗi sẽ tạo mô hình lỗi. Điều tra trước, rồi quyết định sửa nguồn, quay lui hay học lại.
Q: Chọn TẤT CẢ biện pháp đúng khi bật huấn luyện liên tục (CT) tự động theo dữ liệu mới.
+ Kiểm tra chất lượng dữ liệu trước khi huấn luyện và dừng nếu xấu
+ Có cổng đánh giá so với champion; mô hình không qua cổng thì không tự triển khai
- Tự động triển khai mọi mô hình vừa huấn luyện xong lên 100% traffic để luôn mới nhất
+ Với mô hình rủi ro cao, thêm bước duyệt của con người giữa CT và CD
E: CT thiếu cổng có thể tự động đưa mô hình tệ hơn lên production, nhất là khi dữ liệu mới có lỗi. Kiểm tra dữ liệu, cổng chất lượng và (khi cần) duyệt của con người là rào chắn.
Q: Nhóm muốn dùng GPU cho mọi dịch vụ dự đoán, kể cả mô hình cây quyết định xử lý bảng dữ liệu nhỏ. Đánh giá hợp lý nhất?
- Hợp lý vì GPU luôn nhanh hơn
+ Không cần: mô hình cổ điển chạy tốt trên CPU; GPU đắt và chỉ đáng khi mô hình học sâu thực sự cần, hãy đo độ trễ và chi phí trước khi quyết định
- Hợp lý vì GPU rẻ hơn CPU
- GPU không thể dùng trên Kubernetes
E: GPU có chi phí cao và cần quản lý riêng. Với mô hình bảng dữ liệu cổ điển, CPU thường đủ nhanh và rẻ hơn nhiều. Quyết định bằng số đo độ trễ và chi phí trên mỗi dự đoán.
Q: Pipeline CI của dự án ML chạy huấn luyện đầy đủ trên toàn bộ dữ liệu (3 giờ) cho mỗi pull request. Cải thiện hợp lý?
- Giữ nguyên vì phải kiểm thử kỹ
+ Giữ CI nhanh bằng kiểm thử mã, kiểm tra dữ liệu mẫu nhỏ và huấn luyện thử vài giây; huấn luyện đầy đủ thuộc về CT, chạy theo lịch hoặc khi có dữ liệu mới
- Bỏ hẳn kiểm thử khi mở pull request
- Chạy huấn luyện đầy đủ chỉ trên máy cá nhân
E: Vòng phản hồi của CI cần ngắn (vài phút). Huấn luyện đầy đủ mất nhiều giờ làm pull request tắc nghẽn; thay bằng smoke test nhỏ, còn CT lo phần huấn luyện lớn.

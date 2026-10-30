# MLOps notes

Ghi chú tự học về MLOps: vòng đời học máy, dữ liệu và đặc trưng, theo dõi thí nghiệm, pipeline, registry, đánh giá, phục vụ, chiến lược triển khai, giám sát drift, CI/CD/CT, hạ tầng và quản trị. 17 trang, 12 câu luyện tập, tiếng Việt.

Site: https://tracviethustcs.github.io/mlops-notes/
Trang chủ: https://tracviethustcs.github.io/

Đây là ghi chú tự soạn, không phải tài liệu chính thức của bất kỳ nhà cung cấp nào. Công cụ và cú pháp thay đổi nhanh: đối chiếu tài liệu hiện hành và thử với dữ liệu hoặc môi trường thử nghiệm.

## Cấu trúc

- `index.html`: trang đã build, một file duy nhất (HTML, CSS, JS), không dùng framework.
- `src/content/*.md`: nội dung từng trang, viết bằng markdown rút gọn.
- `src/build.py`, `src/site.json`, `src/tpl/`: bộ build và cấu hình giao diện.

## Thêm hoặc sửa một trang

1. Mở một file trong `src/content/` (hoặc tạo file `.md` mới; số đầu tên file quyết định thứ tự).
2. Mỗi trang bắt đầu bằng `=== id | Nhóm | Tiêu đề`.
3. Cú pháp: `>> ` là câu dẫn, `## ` và `### ` là tiêu đề, `- ` và `1. ` là danh sách, dòng bắt đầu bằng `|` là bảng (dùng `\|` cho dấu `|` trong ô), khối code bao bằng ba dấu huyền (thêm `ascii` cho sơ đồ), `**đậm**`, `` `code` ``, `[chữ](#id-trang)`.
4. Khối nổi bật: `::: tip Tiêu đề` (hoặc `warn`, `note`, `lab`), nội dung, rồi dòng `:::`.
5. Câu hỏi luyện tập: dòng `@quiz`, mỗi câu gồm `Q: ...`, đáp án `+ ` (đúng) hoặc `- ` (sai), và `E: giải thích`.
6. Build: `python3 src/build.py` (Python 3, không cần thư viện ngoài), rồi commit `index.html`.

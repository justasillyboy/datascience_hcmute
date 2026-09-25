# Báo cáo kiểm chứng khả thi dữ liệu

**Ngày thực hiện:** 2026-09-07, bổ sung §6 ngày 2026-09-08 · **Nguồn:** Tiki public API
**Mẫu cuối:** 203.510 review / 2.438 sản phẩm / 366 nhà bán / 10 ngành hàng
(mẻ trước 117.819 review ở `data/archive_v2/`; mẻ đầu 66.851 review ở `data/archive_v1/`)
*(§1–§5 viết trên mẻ thử 3.811 review / 120 sản phẩm; các kết luận định tính đều giữ nguyên trên mẫu đầy đủ.)*

Tài liệu này ghi lại *bằng chứng đo được*, không phải phỏng đoán. Mọi con số dưới đây đều tái lập được bằng `src/collect/`.

---

## 1. Vấn đề cần giải quyết

Đề cương gốc dựa trên **Olist Brazilian E-Commerce** (Kaggle), trong đó `Delivery Promise Gap` được tính bằng:

```
Gap = order_delivered_customer_date − order_estimated_delivery_date
```

Cả hai trường này là **dữ liệu vận hành nội bộ** của doanh nghiệp. Giảng viên yêu cầu tự cào dữ liệu nền tảng Việt Nam. Câu hỏi sống còn:

> **Có thể quan sát được (ngày đặt, ngày giao thực tế, ngày giao hứa hẹn, mức hài lòng) ở cấp đơn hàng từ nguồn công khai Việt Nam không?**

Nếu không → đề tài không khả thi và phải đổi. Vì vậy phải kiểm chứng **trước** khi viết bất kỳ dòng code phân tích nào.

---

## 2. Kết quả: 3/4 quan sát được, 1/4 phải định nghĩa lại

### 2.1 Ngày đặt hàng — ✅ CÓ, độ phủ 100%

`created_by.purchased_at` (unix timestamp) trong response của `/api/v2/reviews`.

**Kiểm chứng đây là cấp đơn hàng, không phải cấp tài khoản:** với cùng một `customer_id` có nhiều review, các mốc `purchased_at` **khác nhau**:

```
cust=7178520  n=3 review → 3 mốc khác nhau: [1648613021, 1653154326, 1654967536]
cust=6106142  n=3 review → 3 mốc khác nhau: [1592216587, 1598028475, 1624979888]
cust=17478971 n=2 review → 1 mốc  duy nhất: [1616691485]   ← 2 món cùng 1 đơn
```

Trường hợp cuối chính là cấu trúc `order_items` của Olist: nhiều dòng sản phẩm chung một đơn.

### 2.2 Ngày giao thực tế — ✅ CÓ, độ phủ 98.0%

`timeline.delivery_date`, định dạng `"YYYY-MM-DD HH:MM:SS"`.

```json
"timeline": {
  "review_created_date": "2024-07-26 15:54:42",
  "delivery_date":       "2024-06-03 21:51:54",
  "content": "Đã dùng 2 tháng"
}
```

Đây là tương đương trực tiếp của `order_delivered_customer_date` — **không phải proxy**.

**Phân bố lead time** `delivery_date − purchased_at` (n=490 cặp đầy đủ):

> ⚠️ **Bảng này tính bằng đồng hồ chưa sửa.** Ngày 2026-09-09 phát hiện `purchased_at`
> (epoch, đọc theo UTC) và `delivery_date` (chuỗi giờ VN) nằm trên hai múi giờ khác
> nhau, nên mọi giá trị dưới đây **phồng lên đúng 0.2917 ngày** — xem
> [`FINDINGS.md §6.1`](FINDINGS.md). Cột "đã sửa" là phép tịnh tiến đúng của cột gốc
> (trừ hằng số thì phân vị dịch theo, không cần đo lại); riêng tỉ lệ âm thì **không**
> suy ra được bằng tịnh tiến nên để nguyên và đánh dấu.

| Thống kê | Giá trị gốc (ngày) | Đã sửa múi giờ |
|---|---|---|
| min | −705.14 | −705.43 |
| p25 | 0.47 | 0.18 |
| **median** | **1.07** | **0.78** |
| p75 | 2.12 | 1.83 |
| max | 50.98 | 50.69 |
| âm (<0) | 13 / 490 = **2.7%** | *chưa đo lại* |
| > 60 ngày | 0 | 0 |

Đây là số **pilot** (n=490). Con số chính thức nay lấy từ mẻ đầy đủ 198.355 dòng
phân tích được (mẻ 117.819 trước đây: 114.672): trung vị có trọng số **1.26 ngày**
(trước: 1.30), tỉ lệ âm **0.88%** (trước: 1.00%) ([`FINDINGS.md §2`](FINDINGS.md)).

Giá trị âm **là vấn đề dữ liệu thật** — đã điều tra xong ngày 2026-09-09: nguyên
nhân là `purchased_at` ghi lần mua **gần nhất** của khách với sản phẩm đó, nên khi
khách mua lại sau khi đã review thì nó không còn cùng đơn với `delivery_date`.
Giả thuyết "đơn đổi/trả" đã bị bác bỏ. Bằng chứng: [`FINDINGS.md §6.2`](FINDINGS.md).

### 2.3 Mức hài lòng — ✅ CÓ, độ phủ 100% + một trường Olist không có

- `rating` (1–5) — tương đương `review_score`, độ phủ 100%.
- **`delivery_rating[]`** — bộ 4 câu hỏi có cấu trúc, độ phủ 18.1%:

```json
[{"question":"Thời gian giao hàng?",    "option":"Giao đúng hẹn"},
 {"question":"Thái độ của shipper?",    "option":"Lịch sự"},
 {"question":"Giờ giao hàng?",          "option":"Có hẹn giờ trước"},
 {"question":"Cách đóng gói sản phẩm?", "option":"Cẩn thận"}]
```

Phân bố (n=689 review có nhãn):

| Câu hỏi | Lựa chọn | Tỉ lệ |
|---|---|---|
| Thời gian giao hàng? | Giao đúng hẹn / **Giao trễ hẹn** | 98.7% / **1.3%** |
| Thái độ của shipper? | Lịch sự / Thô lỗ | 99.6% / 0.4% |
| Giờ giao hàng? | Có hẹn giờ trước / Không hẹn trước | 93.6% / 6.4% |
| Cách đóng gói sản phẩm? | Cẩn thận / Cẩu thả | 98.4% / 1.6% |

**Ý nghĩa:** `"Thời gian giao hàng?"` là **nhãn đúng-hẹn/trễ-hẹn do chính khách hàng báo cáo**. Olist **không có** trường này — ở Olist "trễ" chỉ là đại lượng *suy ra*. Ở đây ta có cả đại lượng suy ra **và** nhãn quan sát được → cho phép **kiểm chứng chéo**, chính là tầng "external sanity check" của khung chứng minh đúng.

### 2.4 Ngày giao hứa hẹn — ❌ KHÔNG có theo từng đơn

Đã dò và **loại trừ**: `/api/v2/products/{pid}` không chứa trường giao hàng nào (chỉ có `stock_item.preorder_date`, luôn `null`). 5 endpoint ước tính giao hàng phỏng đoán đều trả **404**.

Thứ **có** là SLA site-wide, nhúng trong schema.org của trang sản phẩm dưới `@id: "#TikiShippingPolicy"`:

```json
"deliveryTime": {
  "handlingTime": {"minValue": 0, "maxValue": 1, "unitCode": "DAY"},
  "transitTime":  {"minValue": 1, "maxValue": 4, "unitCode": "DAY"}
}
```

→ **SLA công bố = 1–5 ngày, giống hệt nhau cho mọi sản phẩm.** Promise là hằng số, không biến thiên theo đơn.

---

## 3. Thiên lệch chọn mẫu — rủi ro lớn nhất, đã có cách xử lý

### 3.1 Phát hiện

Cào theo cách ngây thơ (`sort=top_seller` + sort review mặc định) cho ra mẫu **cực lệch**:

```
rating:  5★ 3655 (95.9%) | 4★ 117 | 1★ 16 | 3★ 13 | 2★ 10     (n=3811)
"Giao trễ hẹn": 9 / 689 = 1.3%
```

Biến mục tiêu gần như không có phương sai → mô hình sẽ học được "đoán 5 sao luôn" và mọi kết luận đều vô nghĩa.

### 3.2 Lối thoát — đã kiểm chứng hoạt động

API nhận tham số **`sort`** với giá trị lọc theo sao:

```
sort=stars|1       sort=stars|1|2|3 ("Chưa hài lòng")     sort=stars|5      sort=id|desc
```

Test trên `product_id=115078652` (histogram cho biết có đúng 1 review 1★):

```
sort=stars|1      → paging.total=1   ratings=[1]     ✅ đúng
sort=stars|1|2|3  → paging.total=1   ratings=[1]     ✅ đúng (2★=0, 3★=0)
sort=stars|5      → paging.total=83  ratings=[5,5,…] ✅ đúng
```

Đồng thời **mọi** response đều kèm histogram cấp quần thể:

```json
"stars": {"1":{"count":0}, "2":{"count":0}, "3":{"count":1}, "4":{"count":7}, "5":{"count":131}}
```

Đây là **đếm thật của toàn bộ quần thể**, không thiên lệch, lấy được với chi phí 1 request.

### 3.3 Thiết kế lấy mẫu bắt buộc

**Stratified sampling với kích thước tầng đã biết:**

1. Với mỗi sản phẩm, 1 request → đọc histogram `stars` → biết chính xác `N_h` của từng tầng sao `h`.
2. Cào **vét cạn** tầng hiếm (1★, 2★, 3★) — chúng nhỏ nên rẻ; **lấy mẫu** tầng 5★.
3. Gán trọng số `w_h = N_h / n_h` cho mỗi review.
4. **Mọi ước lượng quần thể** (tỉ lệ trễ hẹn, mean rating, tỉ lệ vượt SLA…) **phải nhân trọng số**.

Ước tính chi phí: sản phẩm điển hình có 100–500 review với 1–3% ở tầng thấp → vét cạn tầng hiếm chỉ tốn ~2–5 request/sản phẩm.

> Điều này biến rủi ro chí mạng thành **điểm mạnh phương pháp luận**: đây là survey sampling đúng bài, gần như không nhóm sinh viên nào làm.

---

## 4. Bảng ánh xạ Tiki → Olist

| Bảng Olist | Trường Olist | Nguồn Tiki | Độ phủ |
|---|---|---|---|
| `orders` | `order_purchase_timestamp` | `created_by.purchased_at` | 100% |
| `orders` | `order_delivered_customer_date` | `timeline.delivery_date` | 98.0% |
| `orders` | `order_estimated_delivery_date` | SLA site-wide 1–5 ngày (hằng số) | — |
| `reviews` | `review_score` | `rating` | 100% |
| `reviews` | `review_creation_date` | `timeline.review_created_date` | 98.0% |
| `customers` | `customer_id` | `customer_id` | 100% |
| `customers` | `customer_city/state` | — | **❌ 0%** |
| `sellers` | `seller_id` | `seller.id`, `seller.name` | 100% |
| `sellers` | `seller_city/state` | — | ❌ chưa dò được |
| `products` | `category`, `price`, `weight` | listing + product detail | 100% / 100% / một phần |
| `order_items` | `price`, `freight_value` | `price` ✅ / freight ❌ | — |
| — | *(không có ở Olist)* | **`delivery_rating` — nhãn trễ hẹn** ⭐ | 18.1% |

**Ba khoảng trống phải khai báo là limitation:** địa lý khách hàng, phí vận chuyển, ngày giao hứa hẹn theo đơn.
**Một lợi thế Olist không có:** nhãn đúng-hẹn do khách tự báo.

---

## 5. Ba phương án định nghĩa lại Delivery Promise Gap

Vì §2.4 cho thấy promise là hằng số, `Gap = actual − promise` sẽ biến thiên **chỉ qua `actual`**. Ba lựa chọn:

| | Định nghĩa | Ưu | Nhược |
|---|---|---|---|
| **A** | `Gap = lead_time − 5 ngày` (SLA công bố) | Đơn giản, trung thực, bám sát đề cương gốc | Gap ≡ lead time dịch chuyển; promise không có thông tin riêng |
| **B** | Dùng thẳng nhãn `"Giao đúng hẹn"/"Giao trễ hẹn"` làm biến trễ | Nhãn thật do khách báo | Chỉ 18.1% có nhãn, và lệch 98.7/1.3 |
| **C** | **Tính Gap theo A, rồi đối chiếu với nhãn B** | Có cả đại lượng suy ra lẫn nhãn quan sát → **kiểm chứng chéo được** | Phức tạp hơn |

**Khuyến nghị: C.** Lý do — nó cho phép trả lời đúng câu hỏi trong tiêu đề đề tài:

> **SLA có phải thước đo đúng của trải nghiệm khách hàng không?**
>
> - Đơn **nằm trong** SLA nhưng khách nói **trễ hẹn** → SLA quá lỏng, doanh nghiệp "đạt KPI" mà khách vẫn bất mãn.
> - Đơn **vượt** SLA nhưng khách nói **đúng hẹn** → SLA đặt sai chỗ.
>
> Cả hai ô lệch nhau này **chính là "bẫy SLA"**. Đây là phát hiện phản trực giác mà Olist không thể tạo ra, vì Olist không có nhãn khách tự báo.

Phương án C bám tiêu đề **"BẺ GÃY BẪY SLA"** sát hơn thiết kế Olist ban đầu: nó thật sự *bẻ gãy* chỉ số SLA thay vì chỉ đo chỉ số đó.

**→ Cần nhóm chốt trước khi sang bước Feature Engineering.**

---

## 6. Vì sao chọn Tiki chứ không phải sàn lớn hơn?

Câu hỏi phản biện hiển nhiên: Shopee và Lazada có lượng đơn lớn hơn Tiki nhiều
lần. Vì sao không lấy nguồn đông hơn?

### 6.1 Trả lời ngắn

**Không chọn Tiki vì dễ. Chọn vì đây là sàn TMĐT Việt Nam duy nhất công khai dấu
thời gian giao hàng ở cấp đơn hàng — chính là biến phụ thuộc của đề tài.**

Đề tài đo `Delivery Promise Gap`. Không có ngày giao thực tế thì **không tồn tại
đề tài**, dù có 10 triệu review. Số lượng chỉ có giá trị sau khi biến cần đo đã
tồn tại; nó không thay thế được biến đó.

### 6.2 Bằng chứng đo được

Chạy lại bằng `python3 scripts/probe_platforms.py`; log lưu tại
[`docs/evidence/platform_probe_2026-09-08.log`](evidence/platform_probe_2026-09-08.log).

| Sàn | Truy cập | Bằng chứng thô | Kết luận |
|---|---|---|---|
| **Tiki** | HTTP 200 | 4/4 trường bắt buộc có mặt (`purchased_at`, `timeline.delivery_date`, `rating`, `delivery_rating`) | ✅ **Dùng được** |
| **Shopee** | HTTP 403 | `{"error":90309999,"redirect_to_error_page":true}` | ❌ WAF chặn, không lấy được payload |
| **Lazada** | HTTP 200 | 0 `itemId` nhúng sẵn, phát hiện anti-bot `baxia` | ❌ Dữ liệu render bằng JS |
| **Sendo** | — | `ConnectTimeout` sau 25s | ❌ Không truy cập được |

Với Shopee và Lazada, ta **chưa xác minh được** schema review của họ có chứa ngày
giao hay không — vì không tiếp cận nổi payload. Đây là điểm cần nói chính xác:
kết luận là *"không truy cập được"*, **không phải** *"chắc chắn không có dữ liệu"*.
Nhưng với một đồ án có hạn nộp, đặt cược toàn bộ vào dữ liệu **chưa xác minh được
là tồn tại**, trên nền tảng **chủ động chặn thu thập**, là rủi ro trắng tay.

### 6.3 Ranh giới đạo đức đã giữ

Shopee trả 403 ở **đúng request đầu tiên** và ta **dừng lại tại đó**. Không xoay
vòng IP, không giả chữ ký request, không giải CAPTCHA. Vượt rào kiểm soát truy
cập là vượt quá phạm vi một đồ án môn học và không phải thứ trình bày được trước
hội đồng. Việc sàn từ chối cũng là một *kết quả nghiên cứu*, và được ghi nhận
đúng như vậy.

### 6.4 Hai vế của câu hỏi, tách riêng

**"Sàn khác nhiều dữ liệu hơn"** — đây là vấn đề giả, và giải được trong nội bộ
Tiki mà không cần đổi nguồn:

| | Hiện tại | Dư địa |
|---|---|---|
| Review đã lấy | 203.510 | **46,4%** của 438.295 review có sẵn *trên chính 2.438 sản phẩm này* |
| Sản phẩm | 2.438 | giới hạn bởi `--max-pages 20`, còn nâng được |
| Ngành hàng | 10 | Tiki có hàng nghìn category |

Nếu cần thêm dữ liệu thì nới `sample_cap` và chạy lại, chứ không phải đổi sàn.

**Đã kiểm chứng ngày 2026-09-08:** mở rộng từ 66.851 → **117.819 review (+76%)**
chỉ tốn 1 giờ 37 phút, không cần đổi nguồn. **Kiểm chứng lần hai ngày 2026-09-09:**
nới `--sample-cap 60 → 200` (giữ nguyên 2.438 sản phẩm) đưa mẫu lên **203.510 review
(+73%)**, tốn 1 giờ 30 phút, độ phủ quần thể 26,9% → 46,4%. Nhưng cả hai lần mở
rộng đều cho thấy thêm dữ liệu **không tự động** làm mọi kết quả chắc hơn — ở lần
đầu hai ước lượng bị lật vì cỡ mẫu hiệu dụng quá thấp (`FINDINGS.md §5`); ở lần hai,
kết luận chính tiếp tục sống sót và `n_eff` tăng 139% (1.960 → 4.689). Cái cần tăng
là `n_eff` của đúng nhóm con liên quan, không phải tổng số dòng.

**"Sàn khác chất lượng hơn"** — với đề tài *này* thì ngược lại. Tiki cho **cả
hai** thứ cần thiết cùng lúc:

* **đại lượng suy ra** — `lead_days` tính từ hai dấu thời gian, và
* **nhãn quan sát được** — `dr_thoi_gian` do chính khách hàng báo cáo.

Có đồng thời cả hai mới **kiểm chứng chéo** được, và đó chính là nguồn gốc của
toàn bộ kết quả trong `FINDINGS.md`. Ngay cả **Olist — dataset tham chiếu — cũng
không có nhãn này**.

### 6.5 Điểm yếu thật, khai báo trước

Tiki **không đại diện** cho toàn bộ TMĐT Việt Nam: thị phần nhỏ hơn Shopee đáng
kể, thiên về hàng chính hãng, và có hệ thống logistics riêng (TikiNOW) nên **giao
hàng tốt hơn mặt bằng chung**.

Hướng thiên lệch này cần được nêu rõ, nhưng nó **không làm yếu kết luận** — nó
làm kết luận trở nên *thận trọng*: nếu ngay trên một sàn giao hàng tốt mà SLA đã
bỏ sót phần lớn bất mãn của khách, thì trên sàn giao kém hơn, cái bẫy chỉ có thể
nghiêm trọng hơn. Kết quả của nhóm vì vậy là một **chặn dưới**.

> Khi trích thị phần trong báo cáo, phải dẫn nguồn thật (Metric.vn hoặc YouNet
> ECI), không được nói chay.

### 6.6 Một phát hiện phụ

Trong 4 sàn TMĐT lớn ở Việt Nam, **chỉ 1 sàn công khai dấu thời gian giao hàng**.
Đây tự nó là một kết quả về khả dụng dữ liệu, và giải thích vì sao gần như không
có nghiên cứu học thuật nào về SLA giao hàng dùng dữ liệu TMĐT Việt Nam.

---

## 7. Kết luận

**Đề tài KHẢ THI** trên dữ liệu tự cào từ Tiki, với hai điều chỉnh bắt buộc:

1. **Định nghĩa lại Promise Gap** theo §5 (khuyến nghị phương án C).
2. **Lấy mẫu phân tầng theo sao + hiệu chỉnh trọng số** theo §3.3 — không thương lượng.

Và ba limitation phải khai báo trung thực: không có địa lý khách hàng, không có phí vận chuyển, promise là SLA hằng số chứ không theo từng đơn.

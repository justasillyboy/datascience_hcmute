# Kết quả phân tích

**Mẫu:** 117.819 review · 2.438 sản phẩm · 366 nhà bán · 10 ngành hàng · quần thể ước lượng 438.248 review
**Nguồn:** tự cào từ Tiki public API · **Cập nhật:** 2026-09-09

> Mọi con số dưới đây sinh ra bằng hai lệnh, không có số nào chép tay:
> `python3 -m src.clean.run_clean` → `python3 -m src.evaluate.run_analysis`.
> Log đầy đủ: [`evidence/analysis_2026-09-09.txt`](evidence/analysis_2026-09-09.txt)
> và [`evidence/lead_anomaly_2026-09-09.txt`](evidence/lead_anomaly_2026-09-09.txt).

---

## 0. Đọc trước khi trích dẫn

Tài liệu này đã qua **hai lần bị lật số**:

1. **Mở rộng mẫu** 66.851 → 117.819 review đã lật hai trong ba kết quả — §5.
2. **Sửa lỗi múi giờ** ngày 2026-09-09 đã dịch **toàn bộ** thang thời gian giao
   hàng xuống 0,2917 ngày — §6.

Hệ quả: **mọi con số thời gian giao hàng in trước 2026-09-09 đều sai lệch đúng
7 giờ theo hướng phồng lên.** Tỉ lệ vượt SLA từng báo cáo là 8,54%; con số đúng
là **7,27%**. Nếu thấy 8,54% · trung vị 1,59 · p90 4,64 ở bất kỳ đâu → đã cũ.

Kết luận chính **không đổi dấu và không mất ý nghĩa thống kê** qua cả hai lần —
đó là lý do nó được dùng làm kết luận, xem §4.

---

## 1. Chất lượng dữ liệu

| `quality_flag` | n | % |
|---|---|---|
| ok — dùng được | 114.672 | **97,33%** |
| thiếu `delivery_date` | 1.806 | 1,53% |
| `lead_days` âm | 1.181 | 1,00% |
| thiếu `purchased_at` | 150 | 0,13% |
| `lead_days` > 90 ngày | 10 | 0,01% |

Data contract: **22/22 phép kiểm đạt** trên bảng đã làm sạch.

> **Đính chính:** bản trước ghi "21/21 đạt trên *cả bảng thô lẫn* bảng đã làm
> sạch". Vế đó sai. Bảng thô có **8 `review_id` trùng** (một sản phẩm nằm ở hai
> ngành hàng, phân trang chồng lấn) nên nó **không** qua được phép kiểm khoá duy
> nhất — phép kiểm ấy chỉ đúng *sau* bước khử trùng lặp. Contract đã được xếp lại
> theo đúng tầng: bảng thô kiểm "bản sao phải trùng khớp nội dung" (0 bản sao mâu
> thuẫn — bản sao do phân trang là vô hại, bản sao *lệch nội dung* mới là hỏng),
> bảng sạch kiểm khoá duy nhất.

### Kiểm chứng độc lập thiết kế trọng số

Tổng trọng số của mẫu phải bằng tổng số review thật của 2.438 sản phẩm. Hai con số
này đến từ **hai endpoint khác nhau** và không hề được ép cho khớp:

| Nguồn | Giá trị |
|---|---|
| `Σ weight` — cộng từ histogram sao của endpoint review | 438.248 |
| `Σ review_count` — bộ đếm độc lập của endpoint listing | 438.295 |
| **Chênh lệch** | **47 review (0,0108%)** |

Khớp tới bốn chữ số. Đây là bằng chứng độc lập rằng thiết kế phân tầng và công thức
trọng số `w_h = N_h / n_h` **tính đúng** — một lỗi trong đó sẽ làm hai con số lệch xa.

Dòng lỗi được **gắn cờ, không xoá**. `gap_days` và `sla_breach` để rỗng ở các dòng
này, và contract có một phép kiểm riêng bảo đảm không rò rỉ.

1.181 dòng `lead_days` âm **đã truy được nguyên nhân** — xem §6.2.

## 2. Thời gian giao hàng

Phân vị **có trọng số** (n = 114.672):

| p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|
| 0,12 | 0,66 | **1,30** | 2,65 | **4,35** | 5,93 | **17,75** |

Trung bình 2,21 ngày · tối đa 90,0 ngày · **vượt SLA (>5 ngày): 7,27%**

### ⚠️ Không được trích con số 7,27% như tình hình "hiện tại"

Mẫu trải 12 năm và **không đồng nhất theo thời gian**:

| Năm | n | p90 | Vượt SLA |
|---|---|---|---|
| 2017 | 220 | 6,35 | 20,84% |
| 2018 | 564 | 5,17 | 11,04% |
| 2019 | 2.064 | 4,79 | 8,45% |
| 2020 | 10.072 | 4,38 | 6,99% |
| **2021** | **26.184** | **7,86** | **17,48%** |
| 2022 | 29.206 | 3,96 | 5,06% |
| 2023 | 15.765 | 3,16 | 1,98% |
| 2024 | 11.913 | 2,95 | 1,56% |
| 2025 | 11.093 | 3,71 | 3,97% |
| 2026 | 7.520 | 3,61 | 3,06% |

2021 + 2022 chiếm **47%** mẫu, và riêng 2021 có tỉ lệ vượt SLA **17,48%** — giai
đoạn giãn cách. Con số gộp 7,27% bị kéo lên chủ yếu bởi năm này. Khi nói về hiện
trạng, dùng số theo năm: 2024–2026 nằm trong khoảng **1,6% – 4,0%**.

> Bảng trong bản trước bỏ sót hai dòng 2017 và 2018 dù chúng có mặt trong log gốc
> và vượt ngưỡng hiển thị `n ≥ 200`. Đã bổ sung.

## 3. Bảng chéo quyết định

Trên 11.391 review có nhãn khách tự báo:

| | Khách: đúng hẹn | Khách: trễ hẹn |
|---|---|---|
| **Trong SLA** (≤5 ngày) | 10.859 | **207** |
| **Vượt SLA** (>5 ngày) | **274** | 51 |

Hai ô in đậm là hai kiểu sai của SLA: **bỏ sót** (khách bất mãn mà SLA báo đạt) và
**báo động giả** (SLA báo vi phạm mà khách hài lòng).

Sau khi sửa múi giờ, nhóm "vượt SLA" trong bảng này co từ 442 xuống 325 dòng
(**−26,5%**); trên toàn bộ 114.672 dòng phân tích được thì từ 11.927 xuống 9.912
(**−16,9%**). Nghĩa là **một phần sáu số ca từng bị coi là vi phạm SLA thực ra chưa
bao giờ vi phạm** — tỉ lệ này cao hơn trong nhóm có nhãn vì nhóm đó lệch về các đơn
gần ngưỡng. Cả hai ô lệch vẫn khác 0.

## 4. Kết luận chính

> **Nhãn khách hàng phân biệt mức hài lòng tốt hơn chỉ báo SLA — chênh lệch +0,234 điểm rating, KTC 95% [0,113 · 0,393].**
>
> `n = 11.391 · n_eff = 1.960 · 1.914 cụm sản phẩm`

Khoảng tin cậy **không chứa 0** → có ý nghĩa thống kê.

Diễn giải: nếu SLA đo đúng thứ khách quan tâm, hai chỉ báo phải giải thích mức hài
lòng ngang nhau. Thực tế nhãn khách mạnh hơn rõ rệt. **SLA đang đo sai thứ.**

### Vì sao tin được kết luận này mà không tin hai kết luận kia

Đây là ước lượng duy nhất sống sót qua **cả hai** phép thử độc lập:

| | Mẫu 66.851 | Mẫu 117.819 | + sửa múi giờ |
|---|---|---|---|
| Chênh lệch sức phân biệt | +0,209 | +0,241 | **+0,234** |
| KTC 95% | [0,06 · 0,44] | [0,129 · 0,398] | [0,113 · 0,393] |
| `n_eff` | 1.019 | 1.960 | 1.960 |

Mẫu tăng 76% rồi thang thời gian dịch 0,29 ngày — điểm ước lượng vẫn nằm gọn
trong KTC cũ và KTC vẫn hẹp lại. Đó là hành vi của một hiệu ứng thật, không phải
của nhiễu.

### Hai kết quả phụ — trình bày dưới dạng khoảng, không phải điểm

| | Ước lượng | KTC 95% | n_eff | Trạng thái |
|---|---|---|---|---|
| % than phiền trễ hẹn đến từ đơn **trong SLA** | 86,3% | [77,3 · 93,0] | **48** | ⚠️ mong manh |
| % đơn **vượt SLA** mà khách nói đúng hẹn | 88,4% | [80,9 · 94,0] | **99** | ⚠️ mong manh |

Cả hai đều **hỗ trợ** kết luận chính, nhưng không đủ chắc để làm kết luận.

Đáng chú ý: sau khi sửa múi giờ, ước lượng thứ hai tụt từ `n_eff = 129` xuống
**99** và **tự động bật cờ ⚠️ MONG MANH** — cơ chế ở §5 hoạt động đúng như thiết
kế, không cần ai nhớ ra phải cảnh báo.

## 5. Bài học phương pháp 1: số dòng ≠ lượng thông tin

### Chuyện đã xảy ra

Bản phân tích đầu tiên (66.851 review) báo cáo:

* 91,9% [85,1 · 95,7] — than phiền trễ hẹn đến từ đơn trong SLA
* 94,3% [91,2 · 96,2] — đơn vượt SLA mà khách nói đúng hẹn
* +0,209 [0,06 · 0,44] — chênh lệch sức phân biệt

Sau khi mở rộng mẫu lên 117.819 review (+76%), chạy lại **đúng cùng một đoạn code**:

| | Mẫu cũ | Mẫu mới | |
|---|---|---|---|
| Than phiền từ đơn trong SLA | 91,9% | **81,8%** | ❌ lệch 10 điểm |
| Vượt SLA mà khách nói đúng hẹn | 94,3% | **88,9%** | ❌ lệch 5,4 điểm |
| Chênh lệch sức phân biệt | +0,209 | **+0,241** | ✅ vững, KTC hẹp hơn |

Ước lượng mới **nằm ngoài khoảng tin cậy cũ** ở cả hai dòng đầu. Khoảng tin cậy cũ
đã hứa một độ chính xác không có thật.

> Cả hai cột trên đều tính bằng đồng hồ *chưa sửa* (§6.1). Lỗi múi giờ là một hằng
> số chung cho cả hai, nên nó **triệt tiêu trong phép so sánh này** — bài học dưới
> đây thuần tuý là hiệu ứng cỡ mẫu. Giá trị hiện hành nằm ở §4.

### Nguyên nhân

Cỡ mẫu hiệu dụng Kish: `n_eff = (Σw)² / Σw²` — số quan sát *thực sự* đóng góp
thông tin sau khi nhân trọng số.

| Nhóm ước lượng | n dòng (cũ) | **n_eff (cũ)** | n_eff (mới) |
|---|---|---|---|
| Khách nói "trễ hẹn" | 182 | **25,4** | 48,1 |
| Đơn vượt SLA | 279 | 84,8 | 129,0 |
| Toàn bộ có nhãn | 7.516 | **1.019** | 1.960 |

**Con số 91,9% được dựng trên cỡ mẫu hiệu dụng bằng 25.** Trên giấy là 182 dòng,
nhưng trọng số phân tán tới `w_max = 80,8`, khiến 1% dòng nặng nhất chiếm 13,7%
tổng trọng số. Vài review 5★ trọng số cao chi phối toàn bộ ước lượng.

Quy luật rút ra: **thiết kế phân tầng làm mẫu tổng thể trông rất lớn, nhưng nhóm
con nằm ở giao của hai điều kiện hiếm — "khách phàn nàn trễ" × "có điền nhãn" —
thì bé tí.** `n` lớn không cứu được `n_eff` nhỏ.

### Đã sửa trong code, không chỉ trong tài liệu

`Estimate` giờ **luôn mang theo** `n_eff` và số cụm, và tự in `⚠️ MONG MANH` khi
`n_eff < 100` (`src/evaluate/stats.py`). Không thể báo cáo một ước lượng mà giấu đi
độ tin cậy của nó nữa.

## 6. Bài học phương pháp 2: một API, hai đồng hồ

### 6.1 Lỗi múi giờ — mọi thời gian giao hàng từng phồng lên 7 giờ

Tiki trả về mốc thời gian ở **hai định dạng nằm trên hai đồng hồ khác nhau**:

| Trường | Kiểu | Đồng hồ |
|---|---|---|
| `created_by.purchased_at` | unix epoch (giây) | UTC theo định nghĩa |
| `timeline.delivery_date` | chuỗi `YYYY-MM-DD HH:MM:SS` | **giờ Việt Nam (UTC+7)** |
| `timeline.review_created_date` | chuỗi | **giờ Việt Nam (UTC+7)** |

Code cũ đọc epoch bằng `pd.to_datetime(..., unit="s")` — cho ra giờ UTC — rồi trừ
thẳng vào chuỗi giờ Việt Nam. **Mọi `lead_days` vì thế cộng dư đúng 7 giờ.**

Đây không phải suy đoán. Review có **cả hai** dạng cho *cùng một sự kiện*:
`created_at` (epoch) và `review_created_date` (chuỗi) đều là lúc viết review. Hiệu
của chúng đo trực tiếp độ lệch:

```
n = 115.948 review · trung bình 7,0000 h · độ lệch chuẩn 0,000000 h · min = max = 7,0000 h
```

**Phương sai bằng 0.** Đây là quy ước múi giờ, không phải nhiễu — và nó khoá chặt
hằng số `VN_UTC_OFFSET = 7h` trong `src/clean/reviews.py`.

Phép đo này trải **2014–2026** mà độ lệch chuẩn vẫn bằng 0, nên một hằng số là đủ:
Việt Nam không có giờ mùa hè (DST) trong toàn bộ khoảng thời gian của dữ liệu. Nếu
sau này mở rộng sang sàn có DST thì phải đổi sang chuyển đổi theo múi giờ thật, chứ
không cộng hằng số.

Hai kiểm chứng độc lập cùng chỉ một hướng:

* **Giờ trong ngày.** Đơn hàng dồn vào 9–14 giờ nếu đọc theo giờ VN; đọc theo UTC
  thì thành 2–7 giờ sáng — không ai đi chợ mạng lúc đó. Ngày giao dồn vào 9–11 giờ
  sáng theo giờ VN, đúng khung shipper hoạt động.
* **`timeline.current_date`** — dấu thời gian máy chủ đóng vào mỗi response — bám
  đúng đồng hồ Việt Nam của máy chạy cào. Mẻ v2 chạy từ `12:25:41` đến `14:02:11`
  giờ máy (UTC+7, xem `logs_crawl_v2.txt`); `current_date` của các response tương
  ứng trải từ `12:xx` đến `14:02:03`. Khớp tới từng giây, lệch 0 giờ chứ không lệch 7.

**Tác động:** trung vị 1,59 → **1,30** ngày · p90 4,64 → **4,35** · vượt SLA
8,54% → **7,27%** · nhóm "vượt SLA" trong bảng chéo co từ 442 → 325 dòng.

Kết luận chính không đổi (§4). **Luận điểm trung tâm thì mạnh thêm:** SLA công bố
1–5 ngày còn lỏng hơn ta tưởng — 92,7% đơn nằm trong ngưỡng, và trung vị chỉ bằng
**26% của trần SLA**. "Tỉ lệ đạt SLA" gần như không có sức phân biệt nào.

Ba test hồi quy khoá hành vi này lại (`tests/test_clean.py`); ca gắt nhất là *đặt
và giao cùng một thời điểm thì `lead_days` phải bằng 0* — bản cũ trả về 0,2917.

### 6.2 1.181 dòng `lead_days` âm — đã truy ra nguyên nhân

Giả thuyết cũ ghi trong tài liệu là "đơn đổi/trả". **Sai.** Bằng chứng
(`python3 -m src.evaluate.lead_anomaly`):

**Mốc bị lệch chỗ là `purchased_at`, không phải `delivery_date`.**

| Dấu hiệu | Dòng âm | Dòng bình thường |
|---|---|---|
| Ngày đặt rơi **sau** ngày viết review | **92,7%** | 0,00% |
| Ngày giao trước ngày viết review | 100,0% | — |
| Trễ viết review (trung vị) | 2,79 ngày | 4,05 ngày |

Quan hệ **giao → review vẫn còn nguyên** ở các dòng này (Tiki tự tính "đã dùng N
ngày" từ chính hai mốc đó). Chỉ có ngày đặt rơi ra ngoài — trung vị **muộn hơn
ngày viết review 47,8 ngày**. Một đơn hàng không thể được đặt sau khi review của
nó đã được viết.

**Nguyên nhân:** `purchased_at` nằm trong `created_by` — đối tượng *người viết*,
không phải `timeline` — đối tượng *đơn hàng*. Nó ghi lần mua **gần nhất** của
khách với sản phẩm đó. Khách mua lại sau khi đã review thì trường này bị đẩy tới,
còn `delivery_date` vẫn thuộc đơn cũ, và hai mốc không còn cùng một đơn.

Ba dự đoán rơi ra từ giả thuyết đó, cả ba đều đúng:

1. **Review càng cũ, tỉ lệ càng cao** (càng nhiều thời gian để mua lại) — 0,36%
   ở review dưới 1 năm tăng đều lên 1,47% ở review trên 6 năm, **gấp 4,1 lần**,
   tương quan +0,95 theo nhóm tuổi.
2. **Tập trung ở ngành hàng mua lặp** — Thể thao 2,34× · Thời trang nữ 2,04× ·
   Mẹ và bé 1,38×, so với Nhà sách Tiki 0,18× (sách hiếm khi mua lại đúng cuốn đó).
3. **Khách mua nhiều thì tỉ lệ cao hơn** — 0,94% ở khách có 1 review, 1,34% ở
   khách trên 10 review.

Không phải lỗi cào: trải trên 511 sản phẩm và 116 nhà bán, nhiều nhất 35 dòng trên
một sản phẩm.

### 6.3 Loại 1.181 dòng đó ra có làm lệch kết luận không? — Không

Đây là câu hỏi thật sự quan trọng, và **đếm thô trả lời sai nó**:

| | Đếm thô ❌ | Có trọng số ✅ |
|---|---|---|
| rating trung bình, dòng âm | 4,257 | **4,854** |
| rating trung bình, dòng còn lại | 4,336 | **4,780** |
| % 1 sao, dòng âm | 10,58% | **1,87%** |
| % 1 sao, dòng còn lại | 5,43% | **1,47%** |

Đếm thô nói nhóm bị loại **tệ hơn gấp đôi** về tỉ lệ 1 sao — đúng nỗi lo ghi trong
bản trước, rằng ta đang âm thầm gạt một nhóm khách bất mãn ra khỏi phân tích.
Nhưng mẫu được lấy **phân tầng theo sao**, nên đếm thô phóng đại sao thấp trong
*mọi* nhóm con. Nhân trọng số thì nhóm bị loại hoá ra **hài lòng hơn** phần còn
lại (4,854 so với 4,780) và tỉ lệ ≤3 sao **thấp hơn** (3,24% so với 3,99%).

Điều đó nhất quán với chính cơ chế đã tìm ra ở §6.2: đây là **khách mua lại** —
người ta không mua lại thứ mình ghét.

Nhóm này chiếm **1,52% quần thể có trọng số**. Loại nó ra không cắt mất khách bất
mãn; nếu có thì hơi làm *giảm* rating trung bình còn lại. Quyết định giữ nguyên
cách xử lý — gắn cờ và loại khỏi phân tích lead time — nay có bằng chứng, không
còn là mặc định.

> Đây là **cùng một cái bẫy** với §5, ở dạng khác: một lần nữa, con số thô từ mẫu
> phân tầng dẫn tới kết luận ngược. `tests/test_lead_anomaly.py` có một test dựng
> đúng tình huống đó để khoá lại.

### 6.4 Phần chưa nhìn thấy được

Cơ chế ở §6.2 chỉ **lộ ra** khi lần mua lại xảy ra *sau* ngày giao của đơn cũ.
Nếu khách mua lại trong khoảng giữa lúc đặt và lúc nhận đơn đầu, `lead_days` vẫn
dương nhưng đã sai. Cửa sổ đó rộng trung vị **1,30 ngày**, so với nhiều năm phơi
nhiễm sau đó — nên phần ẩn là bậc nhỏ hơn nhiều so với 1,00% đã đo được. Ghi lại
để không ai đọc "1,00%" thành "đã sạch tuyệt đối".

## 7. Giới hạn phải khai báo

**Nhãn khách hàng là MNAR (missing not at random).** Chỉ 9,8% review có nhãn. Trường
`delivery_rating` chỉ tồn tại từ 2023, và nhóm có nhãn giao nhanh hơn nhóm không
nhãn. Vì vậy các ước lượng ở §4 là **ước lượng trên nhóm có nhãn**, không phải trên
toàn quần thể. Thiên lệch còn lại nghiêng theo chiều làm giảm các tỉ lệ này.

**Không đại diện toàn thị trường.** Tiki thị phần nhỏ hơn Shopee đáng kể, thiên về
hàng chính hãng, có logistics riêng → giao hàng tốt hơn mặt bằng chung. Kết quả vì
vậy là **chặn dưới**: sàn giao kém hơn thì cái bẫy chỉ nghiêm trọng hơn. Lý do chọn
Tiki và bằng chứng đo được: [`FEASIBILITY.md §6`](FEASIBILITY.md).

**"Lời hứa" là SLA site-wide 1–5 ngày**, không phải cam kết theo từng đơn — Tiki
không công khai mức đó. Xem [`FEASIBILITY.md §5`](FEASIBILITY.md) phương án C.

**Không có** địa lý khách hàng và phí vận chuyển → không tái tạo được phân tích
vùng miền của Olist.

## 8. Việc cần làm tiếp

1. ~~Điều tra 1.157 dòng `lead_days` âm~~ — **xong**, §6.2 và §6.3.
2. **Phân tích độ nhạy MNAR** — chặn trên/chặn dưới với giả định xấu nhất/tốt nhất
   cho nhóm không có nhãn. Đây giờ là việc đáng làm nhất còn lại.
3. **Tăng `n_eff` cho hai ước lượng phụ** — cào thêm sản phẩm sau 2023 (nơi nhãn tồn
   tại) để nhóm "khách nói trễ hẹn" đủ lớn. Cả hai giờ đều dưới ngưỡng 100.
4. **Phân rã theo ngành hàng và nhà bán** — bẫy SLA có đồng đều không?
5. **Modeling** — dự đoán `is_low_rating`, thang baseline, ablation trên nhóm biến giao hàng.
6. **Đối chiếu Olist** làm tham chiếu quốc tế (chỉ so sánh, không phải nguồn phân tích).

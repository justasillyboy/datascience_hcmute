# Kết quả phân tích

**Mẫu:** 203.510 review · 2.438 sản phẩm · 366 nhà bán · 10 ngành hàng · quần thể ước lượng 438.255 review
(mẻ trước 117.819 review ở `data/archive_v2/` · mẻ đầu 66.851 review ở `data/archive_v1/`)
**Nguồn:** tự cào từ Tiki public API · **Cập nhật:** 2026-09-09

> Mọi con số dưới đây sinh ra bằng hai lệnh, không có số nào chép tay:
> `python3 -m src.clean.run_clean` → `python3 -m src.evaluate.run_analysis`.
> Log đầy đủ: [`evidence/analysis_2026-09-09.txt`](evidence/analysis_2026-09-09.txt)
> và [`evidence/lead_anomaly_2026-09-09.txt`](evidence/lead_anomaly_2026-09-09.txt).

---

## 0. Đọc trước khi trích dẫn

Tài liệu này đã qua **ba lần bị lật số**:

1. **Mở rộng mẫu** 66.851 → 117.819 review đã lật hai trong ba kết quả — §5.
2. **Sửa lỗi múi giờ** ngày 2026-09-09 đã dịch **toàn bộ** thang thời gian giao
   hàng xuống 0,2917 ngày — §6.
3. **Mở rộng mẫu lần hai** 117.819 → 203.510 review (2026-09-09, nới `sample_cap`
   60→200 trên cùng 2.438 sản phẩm) — lần này **không** lật kết luận nào, xem §4.

Hệ quả: **mọi con số thời gian giao hàng in trước 2026-09-09 đều sai lệch đúng
7 giờ theo hướng phồng lên.** Tỉ lệ vượt SLA từng báo cáo là 8,54%; con số đúng
là **7,27%** (117.819), nay **7,12%** trên mẫu 203.510. Nếu thấy 8,54% · trung vị
1,59 · p90 4,64 ở bất kỳ đâu → đã cũ.

Kết luận chính **không đổi dấu và không mất ý nghĩa thống kê** qua cả ba lần —
đó là lý do nó được dùng làm kết luận, xem §4.

---

## 1. Chất lượng dữ liệu

| `quality_flag`        | n       | %          |
| --------------------- | ------- | ---------- |
| ok — dùng được        | 198.355 | **97,47%** |
| thiếu `delivery_date` | 3.064   | 1,51%      |
| `lead_days` âm        | 1.799   | 0,88%      |
| thiếu `purchased_at`  | 280     | 0,14%      |
| `lead_days` > 90 ngày | 12      | 0,01%      |

Data contract: **22/22 phép kiểm đạt** trên bảng đã làm sạch.

> **Đính chính:** bản trước ghi "21/21 đạt trên _cả bảng thô lẫn_ bảng đã làm
> sạch". Vế đó sai. Bảng thô có **8 `review_id` trùng** (một sản phẩm nằm ở hai
> ngành hàng, phân trang chồng lấn) nên nó **không** qua được phép kiểm khoá duy
> nhất — phép kiểm ấy chỉ đúng _sau_ bước khử trùng lặp. Contract đã được xếp lại
> theo đúng tầng: bảng thô kiểm "bản sao phải trùng khớp nội dung" (0 bản sao mâu
> thuẫn — bản sao do phân trang là vô hại, bản sao _lệch nội dung_ mới là hỏng),
> bảng sạch kiểm khoá duy nhất.

> **Đính chính 2 (mẻ 203.510):** ở `sample_cap` lớn hơn, trùng lặp tăng lên **20
> `review_id`**, và lần này **3 cặp thật sự mâu thuẫn** — cùng một review (`product_id`,
> `rating`, `purchased_at` khớp tuyệt đối) nhưng một lần fetch Tiki trả
> `delivery_date`/`review_created_date` = null, lần khác trả đầy đủ. Đây là API trả
> thiếu trường một cách ngẫu nhiên, không phải review khác nhau đội lốt cùng ID.
> Contract bắt đúng thiết kế — dừng pipeline thay vì âm thầm ghi số sai. Xử lý: giữ
> bản đầy đủ, bỏ bản null (203.530 → 203.527 dòng thô), rồi bước khử trùng lặp
> thường lệ bỏ tiếp 17 bản sao vô hại → 203.510 dòng sạch.

### Kiểm chứng độc lập thiết kế trọng số

Tổng trọng số của mẫu phải bằng tổng số review thật của 2.438 sản phẩm. Hai con số
này đến từ **hai endpoint khác nhau** và không hề được ép cho khớp:

| Nguồn                                                  | Giá trị                 |
| ------------------------------------------------------ | ----------------------- |
| `Σ weight` — cộng từ histogram sao của endpoint review | 438.255                 |
| `Σ review_count` — bộ đếm độc lập của endpoint listing | 438.295                 |
| **Chênh lệch**                                         | **40 review (0,0091%)** |

Khớp tới bốn chữ số — sát hơn cả lần đo trước (438.248 vs 438.295, lệch 0,0108%).
Đây là bằng chứng độc lập rằng thiết kế phân tầng và công thức trọng số
`w_h = N_h / n_h` **tính đúng** — một lỗi trong đó sẽ làm hai con số lệch xa.

Dòng lỗi được **gắn cờ, không xoá**. `gap_days` và `sla_breach` để rỗng ở các dòng
này, và contract có một phép kiểm riêng bảo đảm không rò rỉ.

1.799 dòng `lead_days` âm **đã truy được nguyên nhân** — xem §6.2.

## 2. Thời gian giao hàng

Phân vị **có trọng số** (n = 198.355, mẻ 203.510):

| p10  | p25  | p50      | p75  | p90      | p95  | p99       |
| ---- | ---- | -------- | ---- | -------- | ---- | --------- |
| 0,11 | 0,65 | **1,26** | 2,63 | **4,33** | 5,92 | **18,62** |

Trung bình 2,21 ngày · tối đa 90,0 ngày · **vượt SLA (>5 ngày): 7,12%**
(mẻ 117.819: 7,27% — gần như không đổi, xem §4 để biết vì sao đây không phải là
số nên trích khi nói về "hiện tại").

### ⚠️ Không được trích con số 7,12% như tình hình "hiện tại"

Mẫu trải 12 năm và **không đồng nhất theo thời gian**:

| Năm      | n          | p90      | Vượt SLA   |
| -------- | ---------- | -------- | ---------- |
| 2017     | 439        | 6,12     | 17,17%     |
| 2018     | 887        | 5,66     | 12,59%     |
| 2019     | 3.269      | 4,93     | 9,11%      |
| 2020     | 18.458     | 4,23     | 5,98%      |
| **2021** | **52.584** | **7,61** | **16,75%** |
| 2022     | 54.829     | 3,93     | 4,62%      |
| 2023     | 26.591     | 3,10     | 2,07%      |
| 2024     | 18.369     | 2,92     | 1,42%      |
| 2025     | 14.458     | 3,50     | 3,27%      |
| 2026     | 8.355      | 3,58     | 3,30%      |

2021 + 2022 chiếm **54,2%** mẫu (tăng từ 47% ở mẻ 117.819 — mở rộng `sample_cap`
không rải đều theo năm, hai năm này vốn có nhiều review 4–5★ tồn kho nhất), và
riêng 2021 có tỉ lệ vượt SLA **16,75%** — giai đoạn giãn cách. Con số gộp 7,12% bị
kéo lên chủ yếu bởi năm này. Khi nói về hiện trạng, dùng số theo năm: 2024–2026 nằm
trong khoảng **1,4% – 3,3%**.

> Bảng trong bản trước bỏ sót hai dòng 2017 và 2018 dù chúng có mặt trong log gốc
> và vượt ngưỡng hiển thị `n ≥ 200`. Đã bổ sung. Bảng hiện tại tính trên mẻ 203.510.

## 3. Bảng chéo quyết định

Trên 15.816 review có nhãn khách tự báo (mẻ 117.819 trước đây: 11.391):

|                         | Khách: đúng hẹn | Khách: trễ hẹn |
| ----------------------- | --------------- | -------------- |
| **Trong SLA** (≤5 ngày) | 15.160          | **252**        |
| **Vượt SLA** (>5 ngày)  | **339**         | 65             |

Hai ô in đậm là hai kiểu sai của SLA: **bỏ sót** (khách bất mãn mà SLA báo đạt) và
**báo động giả** (SLA báo vi phạm mà khách hài lòng). Cả bốn ô tăng tỉ lệ thuận với
cỡ mẫu có nhãn (~+39%) so với mẻ trước — không có ô nào lệch dạng bất thường.

Ghi nhận lịch sử (đo trên mẻ 117.819 lúc phát hiện lỗi múi giờ): sau khi sửa múi
giờ, nhóm "vượt SLA" trong bảng này co từ 442 xuống 325 dòng (**−26,5%**); trên
toàn bộ 114.672 dòng phân tích được thì từ 11.927 xuống 9.912 (**−16,9%**). Nghĩa
là **một phần sáu số ca từng bị coi là vi phạm SLA thực ra chưa bao giờ vi phạm** —
tỉ lệ này cao hơn trong nhóm có nhãn vì nhóm đó lệch về các đơn gần ngưỡng. Cả hai
ô lệch vẫn khác 0. Mẻ 203.510 đo trên đồng hồ đã sửa ngay từ đầu nên không có phép
so sánh trước/sau tương ứng.

## 4. Kết luận chính

> **Nhãn khách hàng phân biệt mức hài lòng tốt hơn chỉ báo SLA — chênh lệch +0,2082 điểm rating, KTC 95% [0,097 · 0,353].** //QA cập nhật chênh lệch (ngày sửa đổi 26/09/2026)
>
> `n = 15.816 · n_eff = 4.689 · 1.916 cụm sản phẩm`

Khoảng tin cậy **không chứa 0** → có ý nghĩa thống kê.

Diễn giải: nếu SLA đo đúng thứ khách quan tâm, hai chỉ báo phải giải thích mức hài
lòng ngang nhau. Thực tế nhãn khách mạnh hơn rõ rệt. **SLA đang đo sai thứ.**

### Vì sao tin được kết luận này mà không tin hai kết luận kia

Đây là ước lượng sống sót qua **ba** phép thử độc lập liên tiếp:

|                          | Mẫu 66.851    | Mẫu 117.819     | + sửa múi giờ   | Mẫu 203.510         |
| ------------------------ | ------------- | --------------- | --------------- | ------------------- |
| Chênh lệch sức phân biệt | +0,209        | +0,241          | +0,234          | \*\*+0,2082         |
| KTC 95%                  | [0,06 · 0,44] | [0,129 · 0,398] | [0,113 · 0,393] | **[0,097 · 0,353]** |
| `n_eff`                  | 1.019         | 1.960           | 1.960           | **4.689**           |

Mẫu tăng 76% rồi thang thời gian dịch 0,29 ngày rồi tăng tiếp 73% — điểm ước lượng
mới nhất vẫn nằm gọn trong mọi KTC trước đó, và `n_eff` tăng **139%** so với lần
trước. Đó là hành vi của một hiệu ứng thật, không phải của nhiễu.

### Hai kết quả phụ — trình bày dưới dạng khoảng, không phải điểm

|                                               | Ước lượng | KTC 95%       | n_eff   | Trạng thái       |
| --------------------------------------------- | --------- | ------------- | ------- | ---------------- |
| % than phiền trễ hẹn đến từ đơn **trong SLA** | 83,5%     | [75,2 · 89,8] | **85**  | ⚠️ mong manh     |
| % đơn **vượt SLA** mà khách nói đúng hẹn      | 83,8%     | [76,8 · 90,0] | **130** | ✅ hết mong manh |

(mẻ 117.819: 86,3% `n_eff=48` và 88,4% `n_eff=99` — cả hai điểm mới đều **nằm
trong** hai KTC đó.)

Đáng chú ý: sau khi sửa múi giờ (mẻ 117.819), ước lượng thứ hai tụt từ `n_eff = 129`
xuống **99** và **tự động bật cờ ⚠️ MONG MANH**; nay ở mẻ 203.510 nó vượt lại ngưỡng
100 (`n_eff = 130`) và cờ **tự động tắt** — cơ chế ở §5 hoạt động đúng như thiết kế
theo cả hai chiều, không cần ai nhớ ra phải cảnh báo hay gỡ cảnh báo. Ước lượng đầu
vẫn dưới ngưỡng (`n_eff = 85`, tăng từ 48) — cần thêm dữ liệu ở đúng ô "khách nói
trễ hẹn × trong SLA" mới đủ.

## 5. Bài học phương pháp 1: số dòng ≠ lượng thông tin

### Chuyện đã xảy ra

Bản phân tích đầu tiên (66.851 review) báo cáo:

- 91,9% [85,1 · 95,7] — than phiền trễ hẹn đến từ đơn trong SLA
- 94,3% [91,2 · 96,2] — đơn vượt SLA mà khách nói đúng hẹn
- +0,209 [0,06 · 0,44] — chênh lệch sức phân biệt

Sau khi mở rộng mẫu lên 117.819 review (+76%), chạy lại **đúng cùng một đoạn code**:

|                                | Mẫu cũ | Mẫu mới    |                      |
| ------------------------------ | ------ | ---------- | -------------------- |
| Than phiền từ đơn trong SLA    | 91,9%  | **81,8%**  | ❌ lệch 10 điểm      |
| Vượt SLA mà khách nói đúng hẹn | 94,3%  | **88,9%**  | ❌ lệch 5,4 điểm     |
| Chênh lệch sức phân biệt       | +0,209 | **+0,241** | ✅ vững, KTC hẹp hơn |

Ước lượng mới **nằm ngoài khoảng tin cậy cũ** ở cả hai dòng đầu. Khoảng tin cậy cũ
đã hứa một độ chính xác không có thật.

> Cả hai cột trên đều tính bằng đồng hồ _chưa sửa_ (§6.1). Lỗi múi giờ là một hằng
> số chung cho cả hai, nên nó **triệt tiêu trong phép so sánh này** — bài học dưới
> đây thuần tuý là hiệu ứng cỡ mẫu. Giá trị hiện hành nằm ở §4.

### Nguyên nhân

Cỡ mẫu hiệu dụng Kish: `n_eff = (Σw)² / Σw²` — số quan sát _thực sự_ đóng góp
thông tin sau khi nhân trọng số.

| Nhóm ước lượng      | n dòng (66.851) | **n_eff (66.851)** | n_eff (117.819) | n_eff (203.510) |
| ------------------- | --------------- | ------------------ | --------------- | --------------- |
| Khách nói "trễ hẹn" | 182             | **25,4**           | 48,1            | **85**          |
| Đơn vượt SLA        | 279             | 84,8               | 129,0           | **130**         |
| Toàn bộ có nhãn     | 7.516           | **1.019**          | 1.960           | **4.689**       |

Xu hướng giữ nguyên qua lần mở rộng thứ hai: `n_eff` tiếp tục tăng ở cả ba nhóm,
và nhóm "Đơn vượt SLA" giờ đã **vượt ngưỡng 100** — ước lượng phụ tương ứng hết bị
gắn cờ mong manh (§4).

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

| Trường                         | Kiểu                        | Đồng hồ                  |
| ------------------------------ | --------------------------- | ------------------------ |
| `created_by.purchased_at`      | unix epoch (giây)           | UTC theo định nghĩa      |
| `timeline.delivery_date`       | chuỗi `YYYY-MM-DD HH:MM:SS` | **giờ Việt Nam (UTC+7)** |
| `timeline.review_created_date` | chuỗi                       | **giờ Việt Nam (UTC+7)** |

Code cũ đọc epoch bằng `pd.to_datetime(..., unit="s")` — cho ra giờ UTC — rồi trừ
thẳng vào chuỗi giờ Việt Nam. **Mọi `lead_days` vì thế cộng dư đúng 7 giờ.**

Đây không phải suy đoán. Review có **cả hai** dạng cho _cùng một sự kiện_:
`created_at` (epoch) và `review_created_date` (chuỗi) đều là lúc viết review. Hiệu
của chúng đo trực tiếp độ lệch:

```
n = 115.948 review (mẻ 117.819) · trung bình 7,0000 h · độ lệch chuẩn 0,000000 h · min = max = 7,0000 h
n = 200.290 review (mẻ 203.510) · trung bình 7,0000 h · độ lệch chuẩn 0,000000 h · min = max = 7,0000 h
```

Đo lại độc lập trên mẫu lớn hơn 73% cho **đúng cùng một hằng số tới bốn chữ số
thập phân** — không phải trùng hợp của một mẻ dữ liệu cụ thể.

**Phương sai bằng 0.** Đây là quy ước múi giờ, không phải nhiễu — và nó khoá chặt
hằng số `VN_UTC_OFFSET = 7h` trong `src/clean/reviews.py`.

Phép đo này trải **2014–2026** mà độ lệch chuẩn vẫn bằng 0, nên một hằng số là đủ:
Việt Nam không có giờ mùa hè (DST) trong toàn bộ khoảng thời gian của dữ liệu. Nếu
sau này mở rộng sang sàn có DST thì phải đổi sang chuyển đổi theo múi giờ thật, chứ
không cộng hằng số.

Hai kiểm chứng độc lập cùng chỉ một hướng:

- **Giờ trong ngày.** Đơn hàng dồn vào 9–14 giờ nếu đọc theo giờ VN; đọc theo UTC
  thì thành 2–7 giờ sáng — không ai đi chợ mạng lúc đó. Ngày giao dồn vào 9–11 giờ
  sáng theo giờ VN, đúng khung shipper hoạt động.
- **`timeline.current_date`** — dấu thời gian máy chủ đóng vào mỗi response — bám
  đúng đồng hồ Việt Nam của máy chạy cào. Mẻ v2 chạy từ `12:25:41` đến `14:02:11`
  giờ máy (UTC+7, xem `logs_crawl_v2.txt`); `current_date` của các response tương
  ứng trải từ `12:xx` đến `14:02:03`. Khớp tới từng giây, lệch 0 giờ chứ không lệch 7.

**Tác động:** trung vị 1,59 → **1,30** ngày · p90 4,64 → **4,35** · vượt SLA
8,54% → **7,27%** · nhóm "vượt SLA" trong bảng chéo co từ 442 → 325 dòng.

Kết luận chính không đổi (§4). **Luận điểm trung tâm thì mạnh thêm:** SLA công bố
1–5 ngày còn lỏng hơn ta tưởng — 92,7% đơn nằm trong ngưỡng, và trung vị chỉ bằng
**26% của trần SLA**. "Tỉ lệ đạt SLA" gần như không có sức phân biệt nào.

Ba test hồi quy khoá hành vi này lại (`tests/test_clean.py`); ca gắt nhất là _đặt
và giao cùng một thời điểm thì `lead_days` phải bằng 0_ — bản cũ trả về 0,2917.

### 6.2 1.799 dòng `lead_days` âm — đã truy ra nguyên nhân

(mẻ 117.819 trước đây: 1.181 dòng · 1,00%. Mẻ 203.510: 1.799 dòng · **0,88%** —
tỉ lệ giảm nhẹ, cơ chế bên dưới không đổi.)

Giả thuyết cũ ghi trong tài liệu là "đơn đổi/trả". **Sai.** Bằng chứng
(`python3 -m src.evaluate.lead_anomaly`):

**Mốc bị lệch chỗ là `purchased_at`, không phải `delivery_date`.**

| Dấu hiệu                              | Dòng âm   | Dòng bình thường |
| ------------------------------------- | --------- | ---------------- |
| Ngày đặt rơi **sau** ngày viết review | **93,4%** | 0,00%            |
| Ngày giao trước ngày viết review      | 100,0%    | —                |
| Trễ viết review (trung vị)            | 3,06 ngày | 4,31 ngày        |

(mẻ 117.819: 92,7% / 2,79 ngày / 4,05 ngày — cùng dạng, chênh lệch nhỏ do mẫu lớn hơn.)

Quan hệ **giao → review vẫn còn nguyên** ở các dòng này (Tiki tự tính "đã dùng N
ngày" từ chính hai mốc đó). Chỉ có ngày đặt rơi ra ngoài — trung vị **muộn hơn
ngày viết review 49,4 ngày**. Một đơn hàng không thể được đặt sau khi review của
nó đã được viết.

**Nguyên nhân:** `purchased_at` nằm trong `created_by` — đối tượng _người viết_,
không phải `timeline` — đối tượng _đơn hàng_. Nó ghi lần mua **gần nhất** của
khách với sản phẩm đó. Khách mua lại sau khi đã review thì trường này bị đẩy tới,
còn `delivery_date` vẫn thuộc đơn cũ, và hai mốc không còn cùng một đơn.

Ba dự đoán rơi ra từ giả thuyết đó — hai dự đoán đầu **tái xác nhận** trên mẻ
203.510, dự đoán thứ ba chưa đo lại (script hiện không còn tính breakdown này):

1. **Review càng cũ, tỉ lệ càng cao** (càng nhiều thời gian để mua lại) — **0,32%**
   ở review dưới 1 năm tăng đều lên **1,22%** ở review trên 6 năm, **gấp 3,8 lần**,
   tương quan hạng **+0,966** theo nhóm tuổi (mẻ 117.819: 0,36%→1,47%, gấp 4,1×,
   tương quan +0,95 — cùng hình dạng đơn điệu tăng, hệ số gần như không đổi):

   | Tuổi review | n      | bất thường | %     |
   | ----------- | ------ | ---------- | ----- |
   | <1 năm      | 12.623 | 40         | 0,32% |
   | 1–2 năm     | 16.382 | 71         | 0,43% |
   | 2–3 năm     | 18.536 | 110        | 0,59% |
   | 3–4 năm     | 36.143 | 258        | 0,71% |
   | 4–5 năm     | 61.067 | 701        | 1,15% |
   | 5–6 năm     | 41.330 | 445        | 1,08% |
   | 6+ năm      | 14.209 | 174        | 1,22% |

2. **Tập trung ở ngành hàng mua lặp** — bảng đầy đủ 10 ngành (mẻ 117.819 chỉ nêu 4):

   | Ngành hàng                 | n      | bất thường | %     | Bội số    |
   | -------------------------- | ------ | ---------- | ----- | --------- |
   | Thể thao - Dã ngoại        | 1.099  | 34         | 3,09% | **3,50×** |
   | Thời trang nữ              | 34.918 | 625        | 1,79% | **2,02×** |
   | Mẹ và bé                   | 34.634 | 474        | 1,37% | **1,55×** |
   | Điện thoại - Máy tính bảng | 544    | 5          | 0,92% | 1,04×     |
   | Sách tiếng Việt            | 22.623 | 177        | 0,78% | 0,89×     |
   | Điện gia dụng              | 9.791  | 70         | 0,71% | 0,81×     |
   | Làm đẹp - Sức khỏe         | 19.982 | 105        | 0,53% | 0,59×     |
   | Nhà cửa - Đời sống         | 68.003 | 288        | 0,42% | 0,48×     |
   | Nhà sách tiki              | 11.894 | 21         | 0,18% | 0,20×     |
   | Bách hóa online            | 22     | 0          | 0,00% | 0,00×     |

   Thứ tự ngành hàng và hướng lệch **giữ nguyên** so với mẻ 117.819 (Thể thao ·
   Thời trang nữ · Mẹ và bé dẫn đầu, Nhà sách Tiki thấp nhất — sách hiếm khi mua
   lại đúng cuốn đó).

3. **Khách mua nhiều thì tỉ lệ cao hơn** — 0,94% ở khách có 1 review, 1,34% ở
   khách trên 10 review (đo trên mẻ 117.819; `src/evaluate/lead_anomaly.py`
   hiện không còn in lại breakdown này nên **chưa tái xác nhận** ở mẻ 203.510).

Không phải lỗi cào: trải trên 560 sản phẩm và 120 nhà bán (mẻ 117.819: 511 sản
phẩm, 116 nhà bán).

### 6.3 Loại 1.799 dòng đó ra có làm lệch kết luận không? — Không

(mẻ 117.819 trước đây: 1.181 dòng.)

Đây là câu hỏi thật sự quan trọng, và **đếm thô trả lời sai nó**:

|                                 | Đếm thô ❌ | Có trọng số ✅ |
| ------------------------------- | ---------- | -------------- |
| rating trung bình, dòng âm      | 4,479      | 4,770          |
| rating trung bình, dòng còn lại | 4,560      | **4,781**      |
| % 1 sao, dòng âm                | 6,95%      | 2,95%          |
| % 1 sao, dòng còn lại           | 3,14%      | **1,46%**      |
| % ≤3 sao, dòng âm               | —          | 5,11%          |
| % ≤3 sao, dòng còn lại          | —          | **3,97%**      |

(mẻ 117.819, có trọng số: rating 4,854 vs 4,780 · %1 sao 1,87% vs 1,47% — cùng
hướng, biên độ hẹp lại một chút ở mẻ mới nhưng **kết luận không đổi**.)

Đếm thô nói nhóm bị loại **tệ hơn gấp đôi** về tỉ lệ 1 sao — đúng nỗi lo ghi trong
bản trước, rằng ta đang âm thầm gạt một nhóm khách bất mãn ra khỏi phân tích.
Nhưng mẫu được lấy **phân tầng theo sao**, nên đếm thô phóng đại sao thấp trong
_mọi_ nhóm con. Nhân trọng số thì khoảng cách co lại rất nhiều: rating trung bình
gần như bằng nhau (4,770 so với 4,781 — chênh 0,011, coi như bằng nhau trong sai
số) và tỉ lệ 1 sao đơn lẻ vẫn cao hơn ở nhóm bị loại (2,95% so với 1,46%), nhưng
**không còn gấp đôi** như đếm thô gợi ý (6,95% so với 3,14%).

> **Khác với mẻ 117.819:** ở đó nhóm bị loại có trọng số hoá ra **hài lòng hơn**
> phần còn lại (rating 4,854 vs 4,780, ≤3 sao 3,24% vs 3,99%). Ở mẻ 203.510, tỉ lệ
> ≤3 sao của nhóm bị loại lại **cao hơn nhẹ** (5,11% vs 3,97%) — hướng ngược lại,
> tuy biên độ nhỏ (≈1,1 điểm phần trăm) trên một nhóm chỉ chiếm 0,97% quần thể.
> Đây không phải mâu thuẫn nghiêm trọng — cả hai mẻ đều thống nhất ở kết luận
> _quan trọng hơn_: đếm thô phóng đại mức độ bất mãn của nhóm này rất nhiều lần,
> và sau khi nhân trọng số thì tác động của việc loại 1.799 dòng lên ước lượng
> quần thể là **không đáng kể**. Không nên diễn giải quá tay chiều dương/âm nhỏ ở
> một nhóm phụ 0,97% dân số thành một xu hướng chắc chắn.

Điều đó vẫn nhất quán với cơ chế đã tìm ra ở §6.2: phần lớn đây là **khách mua
lại**, không phải một nhóm bất mãn bị che giấu có hệ thống.

Nhóm này chiếm **0,97% quần thể có trọng số** (mẻ 117.819: 1,52%). Loại nó ra
không làm lệch đáng kể ước lượng quần thể ở §4. Quyết định giữ nguyên cách xử lý —
gắn cờ và loại khỏi phân tích lead time — nay có bằng chứng ở hai mẻ độc lập,
không còn là mặc định.

> Đây là **cùng một cái bẫy** với §5, ở dạng khác: một lần nữa, con số thô từ mẫu
> phân tầng dẫn tới kết luận ngược. `tests/test_lead_anomaly.py` có một test dựng
> đúng tình huống đó để khoá lại.

### 6.4 Phần chưa nhìn thấy được

Cơ chế ở §6.2 chỉ **lộ ra** khi lần mua lại xảy ra _sau_ ngày giao của đơn cũ.
Nếu khách mua lại trong khoảng giữa lúc đặt và lúc nhận đơn đầu, `lead_days` vẫn
dương nhưng đã sai. Cửa sổ đó rộng trung vị **1,26 ngày** (mẻ 117.819: 1,30), so
với nhiều năm phơi nhiễm sau đó — nên phần ẩn là bậc nhỏ hơn nhiều so với 0,88%
đã đo được (mẻ 117.819: 1,00%). Ghi lại để không ai đọc "0,88%" thành "đã sạch
tuyệt đối".

## 7. Giới hạn phải khai báo

**Nhãn khách hàng là MNAR (missing not at random).** Chỉ 7,9% review có nhãn (mẻ
117.819: 9,8% — độ phủ tiếp tục giảm vì mẫu mở rộng nghiêng thêm về review cũ/5★,
vốn ít được gắn nhãn hơn; vẫn trên ngưỡng cảnh báo 5% của data contract). Trường
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
3. ~~Tăng `n_eff` cho hai ước lượng phụ~~ — **một nửa xong**: sau khi mở rộng mẫu
   lên 203.510 (nới `sample_cap`, không cào sản phẩm mới), "% đơn vượt SLA mà
   khách nói đúng hẹn" đã qua ngưỡng 100 (`n_eff = 130`). "% than phiền trễ hẹn từ
   đơn trong SLA" vẫn dưới ngưỡng (`n_eff = 85`) — nhóm này hiếm hơn (khách vừa
   phàn nàn trễ vừa đơn nằm trong SLA), cần cào thêm sản phẩm sau 2023 (nơi nhãn
   tồn tại) mới đủ, tăng `sample_cap` thêm không giúp nhiều vì đã gần vét cạn tầng
   1–3★ ở các sản phẩm hiện có.
4. **Phân rã theo ngành hàng và nhà bán** — bẫy SLA có đồng đều không?
5. **Modeling** — dự đoán `is_low_rating`, thang baseline, ablation trên nhóm biến giao hàng.
6. **Đối chiếu Olist** làm tham chiếu quốc tế (chỉ so sánh, không phải nguồn phân tích).

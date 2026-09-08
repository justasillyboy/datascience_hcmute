# BẺ GÃY BẪY SLA TRONG THƯƠNG MẠI ĐIỆN TỬ

**Phân tích độ lệch giao hàng và tác động đến sự hài lòng khách hàng**

Đồ án môn Python for Data Science · HCMUTE · HK1 2025–2026 · PGS.TS Nguyễn Mạnh Hùng

| STT | Họ và tên | MSSV |
|---|---|---|
| 1 | Phan Ngô Quốc An | 24139002 |
| 2 | Đỗ Minh Hiển | 24139013 |
| 3 | Nguyễn Ngọc Ngân | 24139031 |
| 4 | Hồ Ngọc Sỹ | 24139047 |

---

## Câu hỏi nghiên cứu

Doanh nghiệp thương mại điện tử đo chất lượng giao hàng bằng **tỉ lệ đạt SLA** (giao trong thời hạn cam kết). Câu hỏi của đồ án:

> **SLA có thật sự phản ánh trải nghiệm khách hàng không?**

Câu trả lời từ dữ liệu: **gần như không.**

> **Nhãn khách hàng phân biệt mức hài lòng tốt hơn chỉ báo SLA — chênh lệch +0,234 điểm rating, KTC 95% [0,113 · 0,393].**
>
> `n = 11.391 · n_eff = 1.960 · 1.914 cụm sản phẩm` · bootstrap theo cụm sản phẩm

Khoảng tin cậy không chứa 0. Nếu SLA đo đúng thứ khách quan tâm, hai chỉ báo phải
giải thích mức hài lòng ngang nhau — thực tế nhãn khách mạnh hơn rõ rệt. **SLA đang
đo sai thứ.**

Hai kết quả phụ, trình bày dưới dạng khoảng vì cỡ mẫu hiệu dụng thấp:

| | Ước lượng | KTC 95% | n_eff |
|---|---|---|---|
| % than phiền trễ hẹn đến từ đơn **trong SLA** | 86,3% | [77,3 · 93,0] | ⚠️ 48 |
| % đơn **vượt SLA** mà khách nói đúng hẹn | 88,4% | [80,9 · 94,0] | ⚠️ 99 |

SLA hỏng theo cả hai chiều: **bỏ sót** bất mãn thật, và **báo động giả**.

## Hai sai lầm tự bắt được — phần đáng giá nhất của đồ án

### 1. Số dòng không phải lượng thông tin

Bản phân tích đầu (66.851 review) báo cáo **91,9%** và **94,3%** cho hai dòng trên.
Sau khi mở rộng mẫu lên 117.819 review và chạy lại đúng cùng code, chúng tụt xuống
**81,8%** và **88,9%** — **nằm ngoài khoảng tin cậy cũ**.

Nguyên nhân: cỡ mẫu hiệu dụng Kish `n_eff = (Σw)²/Σw²`. Con số 91,9% trông như dựa
trên 182 quan sát, nhưng sau khi nhân trọng số phân tầng thì **n_eff chỉ bằng 25**.
Vài review trọng số cao chi phối toàn bộ ước lượng.

Đã sửa **trong code chứ không chỉ trong tài liệu**: `Estimate` giờ luôn mang theo
`n_eff` và tự in `⚠️ MONG MANH` khi `n_eff < 100`.

### 2. Một API, hai đồng hồ

Tiki trả `purchased_at` là **unix epoch (UTC)** nhưng `delivery_date` là **chuỗi giờ
Việt Nam (UTC+7)**. Trừ thẳng hai thứ đó vào nhau không làm gãy gì cả — nó chỉ cộng
dư đúng 7 giờ vào **mọi** lead time. Lỗi im lặng, tồn tại từ ngày đầu.

Bắt được bằng một phép đo, không phải bằng linh cảm: review có *cả hai* dạng cho
cùng một sự kiện (`created_at` epoch và `review_created_date` chuỗi). Hiệu của chúng
trên 115.948 review là **7,0000 giờ, độ lệch chuẩn 0,000000** — phương sai bằng 0
nghĩa là quy ước múi giờ, không phải nhiễu.

Hậu quả: trung vị giao hàng 1,59 → **1,30** ngày · vượt SLA 8,54% → **7,27%** ·
11.927 → 9.912 dòng vi phạm, tức **một phần sáu số ca từng bị coi là vi phạm SLA
thực ra chưa bao giờ vi phạm**.

**Kết luận chính sống sót qua cả hai lần** — mẫu tăng 76%, rồi thang thời gian dịch
0,29 ngày, mà điểm ước lượng vẫn nằm gọn trong KTC cũ:

| | Mẫu 66.851 | Mẫu 117.819 | + sửa múi giờ |
|---|---|---|---|
| Chênh lệch sức phân biệt | +0,209 | +0,241 | **+0,234** |
| KTC 95% | [0,06 · 0,44] | [0,129 · 0,398] | [0,113 · 0,393] |

Chi tiết cả hai: [`docs/FINDINGS.md`](docs/FINDINGS.md) §5 và §6.

## Dữ liệu — tự thu thập, không dùng dataset có sẵn

**117.819 review · 2.438 sản phẩm · 366 nhà bán · 10 ngành hàng**, tự cào từ Tiki public API.
Mẫu này đại diện cho quần thể **438.248 review**.

Dataset Olist (Kaggle) chỉ dùng làm **tham chiếu schema** và đối chứng quốc tế, không phải nguồn phân tích.

Hai đặc điểm phương pháp đáng chú ý:

1. **Lấy mẫu phân tầng theo sao.** Cào theo cách thông thường cho ra mẫu 96% năm sao — biến mục tiêu không còn phương sai. Ta đọc histogram sao ở cấp quần thể (đếm thật, lấy được với 1 request), vét cạn các tầng hiếm, lấy mẫu tầng đông, rồi gán trọng số `w = N_h / n_h`. Kết quả: mẫu có **14,8% review ≤3 sao** trong khi quần thể chỉ **4,0%** — đủ tín hiệu để mô hình học, mà ước lượng quần thể vẫn không chệch.

   Kiểm chứng độc lập: `Σ weight` = 438.248 khớp với `Σ review_count` = 438.295 lấy từ **endpoint khác** — lệch 0,0108%.

2. **Nhãn khách tự báo.** Tiki công khai một trường Olist không có: bộ câu hỏi `delivery_rating`, trong đó `"Thời gian giao hàng?" → "Giao đúng hẹn"/"Giao trễ hẹn"` là **nhãn đúng-hẹn do chính khách hàng báo cáo**. Nhờ đó ta có *cả* đại lượng suy ra *lẫn* nhãn quan sát được, và kiểm chứng chéo được chúng.

## Chạy lại

```bash
pip install -r requirements.txt

# Bước 1 — Thu thập (có cache; lần chạy thứ hai gần như không gọi mạng)
python3 -m src.collect.run_crawl --max-products 300 --max-pages 10

# Bước 2 — Làm sạch + data contract (2 tầng: bảng thô và bảng đã sạch)
python3 -m src.clean.run_clean

# Bước 3 — Sinh toàn bộ kết quả trong docs/FINDINGS.md §1–§4
python3 -m src.evaluate.run_analysis

# Bước 4 — Điều tra lead_days âm (FINDINGS.md §6)
python3 -m src.evaluate.lead_anomaly

# Kiểm chứng vì sao chọn Tiki (dò 4 sàn TMĐT)
python3 scripts/probe_platforms.py

# Test
python3 -m pytest
```

## Cấu trúc

```
src/collect/    BaseScraper (trừu tượng) → TikiListingScraper, TikiReviewScraper
src/clean/      chuẩn hoá, gắn cờ chất lượng, tính Delivery Promise Gap
src/validate/   data contract — 22 phép kiểm tự động, xếp theo tầng thô/sạch
src/evaluate/   thống kê có trọng số, bootstrap theo cụm sản phẩm
tests/          39 test
src/clean/      run_clean.py     — raw → processed bằng 1 lệnh
src/evaluate/   run_analysis.py  — sinh mọi con số trong FINDINGS.md bằng 1 lệnh
                lead_anomaly.py  — điều tra lead_days âm, 5 phép kiểm
scripts/        probe_platforms.py — bằng chứng vì sao chọn Tiki
docs/           FEASIBILITY.md (nguồn dữ liệu + §6 vì sao Tiki) · FINDINGS.md (kết quả)
docs/evidence/  log thô của các phép kiểm chứng
CLAUDE.md       quy ước làm việc, cạm bẫy đã biết, ràng buộc đạo đức
```

## Lộ trình đồ án

Workflow bắt buộc của môn: **Business → Data Collection → Cleaning → EDA → Modeling**
(bỏ Deployment). Thang điểm **5đ đủ bước · 3đ chứng minh đúng · 2đ code** — 5đ ai
cũng lấy được, nên trọng tâm của repo này là **3đ chứng minh đúng**.

| # | Bước | Trạng thái | Sản phẩm |
|---|---|---|---|
| 1 | **Business** — định nghĩa Delivery Promise Gap, chốt phương án đo | ✅ xong | `CLAUDE.md` §4 · `FEASIBILITY.md` §5 |
| 2 | **Data Collection** — scraper phân tầng theo sao + cache | ✅ xong | `src/collect/` · 117.819 review |
| 3 | **Cleaning** — gắn cờ chất lượng, sửa múi giờ, data contract | ✅ xong | `src/clean/` · `src/validate/` · 22/22 kiểm |
| 4 | **EDA — số** — thống kê có trọng số, bootstrap cụm, `n_eff` | ✅ xong | `src/evaluate/` · `FINDINGS.md` §1–§6 |
| 5 | **EDA — hình** — biểu đồ cho báo cáo và slide | ⬜ chưa | `notebooks/` |
| 6 | **Độ nhạy MNAR** — chặn trên/dưới cho 90% review không nhãn | 🔶 đang dở | `src/evaluate/mnar_sensitivity.py` · kế hoạch ở `CLAUDE.md` §9.1 |
| 7 | **Phân rã** — bẫy SLA theo ngành hàng và nhà bán | ⬜ chưa | |
| 8 | **Modeling** — dự đoán `is_low_rating`, thang baseline, ablation | ⬜ chưa | `src/models/` |
| 9 | **Đối chiếu Olist** — tham chiếu quốc tế, *không* phải nguồn phân tích | ⬜ chưa | |
| 10 | **Báo cáo + slide** | ⬜ chưa | |

Việc kế tiếp: **bước 6**, bắt đầu từ mục 1 trong `CLAUDE.md` §9.1.

### Nguyên tắc xuyên suốt

- **Mọi con số trong tài liệu phải sinh ra bằng một lệnh.** Không chép tay — đã hai
  lần tài liệu âm thầm sai khi dữ liệu đổi.
- **Mọi ước lượng quần thể phải nhân trọng số phân tầng**, và phải kèm `n_eff`.
- **Không xoá dòng lỗi trong im lặng** — gắn cờ, đếm được, báo cáo được.
- Notebook chỉ kể chuyện; logic nằm trong `src/` và có test.

## Đạo đức thu thập dữ liệu

Tôn trọng `robots.txt` của Tiki (không đụng `/api/v2/me/`, `/v1/private/`, `/order/tracking`, `/customer/*`) · 1 request/giây, không chạy song song · cache xuống đĩa, cào đúng một lần · **không public dữ liệu thô** (`data/` nằm trong `.gitignore`) · User-Agent khai rõ mục đích học thuật.

## Giới hạn đã biết

- Nhãn `delivery_rating` là **MNAR**: chỉ tồn tại từ 2023 và nhóm có nhãn giao nhanh hơn nhóm không nhãn. Kết quả đã kiểm tra độ bền khi giới hạn 2023+ nhưng vẫn phải hiểu là ước lượng **trên nhóm có nhãn**.
- Không có địa lý khách hàng và phí vận chuyển → không tái tạo được phân tích vùng miền của Olist.
- "Lời hứa" là SLA site-wide (1–5 ngày), không phải cam kết theo từng đơn.
- 1.181 dòng (1,00%) có `lead_days` âm — **đã truy ra nguyên nhân**: `purchased_at` ghi lần mua
  *gần nhất* của khách với sản phẩm đó, nên khi khách mua lại sau khi đã review thì nó không còn
  cùng đơn với `delivery_date`. Đang gắn cờ và loại khỏi phân tích, chưa xoá; đã kiểm chứng bằng
  trọng số rằng việc loại chúng **không** cắt mất nhóm khách bất mãn (`FINDINGS.md §6.2–6.3`).
- Mẫu trải 12 năm và **không đồng nhất**: 2021+2022 chiếm 47%, riêng 2021 có tỉ lệ vượt SLA 17,5% (giãn cách).
  Không được trích tỉ lệ gộp 7,27% như tình hình hiện tại — hiện trạng 2024–2026 là 1,6%–4,0%,
  dùng bảng theo năm trong `FINDINGS.md §2`.
- Tiki không đại diện toàn thị trường (giao hàng tốt hơn mặt bằng chung) → kết quả là **chặn dưới**.

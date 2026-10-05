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

> **Nhãn khách hàng phân biệt mức hài lòng tốt hơn chỉ báo SLA — chênh lệch +0,208 điểm rating, KTC 95% [0,097 · 0,353].**
>
> `n = 15.816 · n_eff = 4.689 · 1.916 cụm sản phẩm` · bootstrap theo cụm sản phẩm ·
> **sống sót qua 3 lần mở rộng mẫu độc lập** (66.851 → 117.819 → 203.510, bảng bên dưới)

Khoảng tin cậy không chứa 0. Nếu SLA đo đúng thứ khách quan tâm, hai chỉ báo phải
giải thích mức hài lòng ngang nhau — thực tế nhãn khách mạnh hơn rõ rệt. **SLA đang
đo sai thứ.**

Hai kết quả phụ, trình bày dưới dạng khoảng vì cỡ mẫu hiệu dụng thấp:

| | Ước lượng | KTC 95% | n_eff |
|---|---|---|---|
| % than phiền trễ hẹn đến từ đơn **trong SLA** | 83,5% | [75,2 · 89,8] | ⚠️ 85 |
| % đơn **vượt SLA** mà khách nói đúng hẹn | 83,8% | [76,8 · 90,0] | **130** ✅ hết mong manh |

SLA hỏng theo cả hai chiều: **bỏ sót** bất mãn thật, và **báo động giả**.

## Hai sai lầm tự bắt được — phần đáng giá nhất của đồ án

### 1. Số dòng không phải lượng thông tin

Bản phân tích đầu (66.851 review) báo cáo **91,9%** và **94,3%** cho hai dòng trên.
Sau khi mở rộng mẫu lên 117.819 review và chạy lại đúng cùng code, chúng tụt xuống
**81,8%** và **88,9%** — **nằm ngoài khoảng tin cậy cũ**.

Mở rộng tiếp lên 203.510 review (mẻ thứ ba, **cùng đồng hồ đã sửa múi giờ**) thì
hai số này ổn định lại — **83,5%** và **83,8%** — cả hai đều **nằm trong** KTC của
mẻ 117.819 sau khi sửa múi giờ (`[77,3·93,0]` và `[80,9·94,0]`, xem bảng ở §"Kết
luận chính" phía trên). Đúng như dự đoán của cơ chế `n_eff`: một khi `n_eff` đã đủ
lớn để hết bị vài quan sát trọng số cao chi phối, ước lượng ngừng nhảy lung tung.

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
trên 115.948 review (nay đo lại trên 200.290 review ở mẻ 203.510: **vẫn 7,0000 giờ,
độ lệch chuẩn 0,000000**) — phương sai bằng 0 nghĩa là quy ước múi giờ, không phải
nhiễu.

Hậu quả (đo trên mẻ 117.819 lúc phát hiện lỗi): trung vị giao hàng 1,59 → **1,30**
ngày · vượt SLA 8,54% → **7,27%** · 11.927 → 9.912 dòng vi phạm, tức **một phần sáu
số ca từng bị coi là vi phạm SLA thực ra chưa bao giờ vi phạm**.

**Kết luận chính sống sót qua ba lần mở rộng liên tiếp** — mẫu tăng 76% rồi 73%
nữa, cộng thêm một lần thang thời gian dịch 0,29 ngày, mà điểm ước lượng vẫn nằm
gọn trong các KTC trước đó:

| | Mẫu 66.851 | Mẫu 117.819 | + sửa múi giờ | Mẫu 203.510 |
|---|---|---|---|---|
| Chênh lệch sức phân biệt | +0,209 | +0,241 | +0,234 | **+0,208** |
| KTC 95% | [0,06 · 0,44] | [0,129 · 0,398] | [0,113 · 0,393] | **[0,097 · 0,353]** |
| `n_eff` | 1.019 | 1.960 | 1.960 | **4.689** |

Chi tiết cả hai: [`docs/FINDINGS.md`](docs/FINDINGS.md) §5 và §6.

## Dữ liệu — tự thu thập, không dùng dataset có sẵn

**203.510 review · 2.438 sản phẩm · 366 nhà bán · 10 ngành hàng**, tự cào từ Tiki public API
(mẻ trước 117.819 review lưu ở `data/archive_v2/`, mẻ đầu 66.851 review ở `data/archive_v1/`).
Mẫu này đại diện cho quần thể **438.255 review** — phủ **46,4%** quần thể, tăng từ 26,9% ở mẻ trước.

Dataset Olist (Kaggle) chỉ dùng làm **tham chiếu schema** và đối chứng quốc tế, không phải nguồn phân tích.

Hai đặc điểm phương pháp đáng chú ý:

1. **Lấy mẫu phân tầng theo sao.** Cào theo cách thông thường cho ra mẫu 96% năm sao — biến mục tiêu không còn phương sai. Ta đọc histogram sao ở cấp quần thể (đếm thật, lấy được với 1 request), vét cạn các tầng hiếm, lấy mẫu tầng đông, rồi gán trọng số `w = N_h / n_h`. Kết quả: mẫu có **8,6% review ≤3 sao** trong khi quần thể chỉ **4,0%** — đủ tín hiệu để mô hình học, mà ước lượng quần thể vẫn không chệch. (Tỉ lệ này tụt so với 14,8% ở mẻ 117.819 vì `sample_cap` tăng làm tầng 4–5★ phình to hơn tầng hiếm vốn đã vét cạn từ trước — không ảnh hưởng ước lượng có trọng số.)

   Kiểm chứng độc lập: `Σ weight` = 438.255 khớp với `Σ review_count` = 438.295 lấy từ **endpoint khác** — lệch 0,0091%.

2. **Nhãn khách tự báo.** Tiki công khai một trường Olist không có: bộ câu hỏi `delivery_rating`, trong đó `"Thời gian giao hàng?" → "Giao đúng hẹn"/"Giao trễ hẹn"` là **nhãn đúng-hẹn do chính khách hàng báo cáo**. Nhờ đó ta có *cả* đại lượng suy ra *lẫn* nhãn quan sát được, và kiểm chứng chéo được chúng.

## Chạy lại

**Pipeline mô hình bằng một lệnh** (cần sẵn `data/` — xem gói nộp bài bên dưới):

```bash
pip install -r requirements.txt
python main.py                  # ~1–2 phút: feature → chia theo năm → Pipeline(prep + model) → chấm test → outputs/
python main.py --describe       # chỉ in mô tả pipeline: input/output, các bước, siêu tham số
python main.py --search quick   # GridSearchCV lưới rút gọn; --search full = lưới đầy đủ 120 cấu hình (~20 phút)
python main.py --help           # mọi tham số (chọn mô hình, ghi đè siêu tham số, năm chia test…)
```

Notebook EDA cho báo cáo: [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb) — tổng quan, chất lượng dữ liệu, mất cân
bằng lớp, feature vs nhãn, tương quan, leakage, trôi theo thời gian, luận điểm SLA, và bảng "phát hiện EDA → quyết
định trong pipeline".

**Gói nộp cho giảng viên** (code + notebook + dữ liệu rút gọn, giải nén là chạy được `python main.py`):
`python scripts/make_submission.py` → `dist/nop_bai_SLA_tiki.zip`. Gói chứa dữ liệu cào → **chỉ nộp riêng, không
đưa lên GitHub** (`dist/` đã gitignore).

Toàn bộ quy trình từng bước:

```bash
pip install -r requirements.txt

# Bước 1 — Thu thập (có cache; lần chạy thứ hai gần như không gọi mạng)
python3 -m src.collect.run_crawl --max-products 500 --max-pages 20 --sample-cap 200

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
main.py         pipeline mô hình chạy bằng 1 lệnh, có argparse (python main.py --help)
notebooks/      EDA.ipynb (EDA nộp bài) · 01–04 (EDA hình, chứng minh, thang mô hình, GridSearchCV)
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
| 2 | **Data Collection** — scraper phân tầng theo sao + cache | ✅ xong | `src/collect/` · 203.510 review |
| 3 | **Cleaning** — gắn cờ chất lượng, sửa múi giờ, data contract | ✅ xong | `src/clean/` · `src/validate/` · 22/22 kiểm |
| 4 | **EDA — số** — thống kê có trọng số, bootstrap cụm, `n_eff` | ✅ xong | `src/evaluate/` · `FINDINGS.md` §1–§6 |
| 5 | **EDA — hình** — biểu đồ cho báo cáo và slide | ✅ xong | `notebooks/EDA.ipynb` · `notebooks/01_eda.ipynb` |
| 6 | **Độ nhạy MNAR** — chặn trên/dưới cho 90% review không nhãn | 🔶 đang dở | `src/evaluate/mnar_sensitivity.py` · kế hoạch ở `CLAUDE.md` §9.1 |
| 7 | **Phân rã** — bẫy SLA theo ngành hàng và nhà bán | ⬜ chưa | |
| 8 | **Modeling** — dự đoán `is_low_rating`, thang baseline, ablation | ✅ xong | `src/models/` · `main.py` · notebook 03, 04 |
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
- 1.799 dòng (0,88%, mẻ 203.510 — trước là 1.181/1,00% ở mẻ 117.819) có `lead_days` âm — **đã truy ra
  nguyên nhân**: `purchased_at` ghi lần mua *gần nhất* của khách với sản phẩm đó, nên khi khách mua
  lại sau khi đã review thì nó không còn cùng đơn với `delivery_date`. Đang gắn cờ và loại khỏi phân
  tích, chưa xoá; đã kiểm chứng bằng trọng số rằng việc loại chúng **không** cắt mất nhóm khách bất
  mãn (`FINDINGS.md §6.2–6.3`).
- Mẫu trải 12 năm và **không đồng nhất**: 2021+2022 chiếm **54,2%** (tăng từ 47% ở mẻ trước — mẫu mở
  rộng không rải đều theo năm), riêng 2021 có tỉ lệ vượt SLA 16,75% (giãn cách). Không được trích tỉ
  lệ gộp 7,12% như tình hình hiện tại — hiện trạng 2024–2026 là **1,4%–3,3%**, dùng bảng theo năm
  trong `FINDINGS.md §2`.
- Tiki không đại diện toàn thị trường (giao hàng tốt hơn mặt bằng chung) → kết quả là **chặn dưới**.

# CLAUDE.md — Đồ án Khoa học Dữ liệu HCMUTE HK1 2025–2026

## 1. Đề tài

**BẺ GÃY BẪY SLA TRONG THƯƠNG MẠI ĐIỆN TỬ: Phân tích độ lệch giao hàng và tác động đến sự hài lòng khách hàng**

Nhóm 4 người: Phan Ngô Quốc An (24139002) · Đỗ Minh Hiển (24139013) · Nguyễn Ngọc Ngân (24139031) · Hồ Ngọc Sỹ (24139047)

Môn: Python for Data Science — PGS.TS Nguyễn Mạnh Hùng.
Workflow bắt buộc: **Business → Data Collection → Cleaning → EDA → Modeling** (bỏ Deployment).
Thang điểm: **5đ đủ bước · 3đ chứng minh đúng · 2đ code**.

> **Chiến lược trung tâm:** 5đ ai cũng lấy được. **3đ "chứng minh đúng" là toàn bộ khoảng trống để thắng.**
> Mọi quyết định kỹ thuật trong repo này phải phục vụ việc *chứng minh kết luận là đúng*, không phải khoe metric.

## 2. Ràng buộc bất di bất dịch từ giảng viên

**Tự cào dữ liệu từ nền tảng Việt Nam. KHÔNG dùng dataset có sẵn làm nguồn chính.**

- Olist (Kaggle Brazilian E-Commerce) chỉ được dùng làm **tham chiếu schema** và **đối chứng đối sánh quốc tế** — ghi rõ trong báo cáo là dữ liệu tham khảo, không phải dữ liệu phân tích.
- Nguồn chính = Tiki public API, tự cào, tự làm sạch.

## 3. Sự thật về dữ liệu — ĐÃ KIỂM CHỨNG BẰNG THỰC NGHIỆM

Ngày kiểm chứng: **2026-09-07**. Mẫu thử: 3.811 review / 120 sản phẩm / 10 ngành hàng.
Chi tiết đầy đủ: [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md). **Đừng thiết kế lại dựa trên phỏng đoán — các con số dưới đây là đo thật.**

### 3.1 Endpoint dùng được

| Endpoint | Dùng để | Trạng thái |
|---|---|---|
| `GET tiki.vn/api/personalish/v1/blocks/listings?category={id}&page={n}&limit=40&sort=top_seller` | Liệt kê sản phẩm | ✅ 200 |
| `GET tiki.vn/api/v2/products/{pid}` | Chi tiết sản phẩm | ✅ 200 |
| `GET tiki.vn/api/v2/reviews?product_id={pid}&seller_id={sid}&limit=20&page={n}&sort={sort}` | Review + timeline giao hàng | ✅ 200 |

`robots.txt` (đã đọc) **cho phép** mọi endpoint trên. Cấm: `/api/v2/me/`, `/v1/private/`, `/api/v2/reviews/writable`, `/order/tracking`, `/customer/*` → **tuyệt đối không đụng**.

⚠️ `curl` bị CDN chặn 403. **Phải dùng `requests.Session()` của Python.**

### 3.2 Trường dữ liệu và độ phủ (n=500 đo trực tiếp)

| Trường Tiki | Độ phủ | Tương đương Olist |
|---|---|---|
| `created_by.purchased_at` (unix) | **100%** | `order_purchase_timestamp` |
| `timeline.delivery_date` | **98.0%** | **`order_delivered_customer_date`** ⭐ |
| `timeline.review_created_date` | 98.0% | `review_creation_date` |
| `rating` (1–5) | 100% | `review_score` |
| `delivery_rating[]` (bộ 4 câu hỏi) | **18.1%** | *không có trong Olist* ⭐ |
| `customer_id` | 100% | `customer_id` |
| `seller.id` / `seller.name` | 100% | `sellers` |
| `product_id`, `spid`, giá, brand, category | 100% | `products`, `order_items` |
| `created_by.region` | **0%** | ❌ không có địa lý khách hàng |

**Lead time thực tế** `delivery_date − purchased_at` — số chính thức từ mẻ đầy đủ:
median có trọng số **1.30 ngày**, p90 4.35, vượt SLA **7.27%** (`docs/FINDINGS.md` §2).

> ⚠️ Số pilot cũ (median 1.07) tính bằng **đồng hồ chưa sửa** và phồng lên 0.2917
> ngày. `purchased_at` là epoch (UTC), còn `delivery_date`/`review_created_date` là
> chuỗi **giờ Việt Nam (UTC+7)** — phải quy về cùng một đồng hồ trước khi trừ. Hằng
> số `VN_UTC_OFFSET` trong `src/clean/reviews.py` đo được từ dữ liệu: 7,0000 h,
> phương sai 0, trên 115.948 review. Xem `docs/FINDINGS.md` §6.1.

1.00% giá trị âm → **là lỗi dữ liệu phải xử lý trong cleaning, không được im lặng bỏ đi**.
Nguyên nhân **đã truy ra** (2026-09-09): `purchased_at` ghi lần mua *gần nhất* của
khách với sản phẩm đó, nên khi khách mua lại sau khi đã review thì nó không còn cùng
đơn với `delivery_date`. Giả thuyết "đơn đổi/trả" đã bị bác bỏ — `docs/FINDINGS.md` §6.2.

`purchased_at` đã xác minh là **cấp đơn hàng** (cùng một `customer_id` cho ra các mốc khác nhau ở các review khác nhau; hai review cùng `purchased_at` = 2 món trong cùng 1 đơn — đúng cấu trúc `order_items`).

### 3.3 `delivery_rating` — nhãn SLA do khách tự báo (n=689)

```
Thời gian giao hàng?      Giao đúng hẹn 98.7%  |  Giao trễ hẹn 1.3%
Thái độ của shipper?      Lịch sự       99.6%  |  Thô lỗ        0.4%
Giờ giao hàng?            Có hẹn trước  93.6%  |  Không hẹn     6.4%
Cách đóng gói sản phẩm?   Cẩn thận      98.4%  |  Cẩu thả       1.6%
```

Đây là **ground truth trực tiếp cho "đúng hẹn / trễ hẹn"** — thứ Olist không có. Dùng nó để *kiểm chứng* Gap tính toán được, chứ không chỉ để mô tả.

### 3.4 Lời hứa giao hàng (promise)

Tiki **không** công bố ngày giao dự kiến theo từng đơn trong quá khứ. Cái có được là **SLA site-wide** nhúng trong schema.org của trang sản phẩm (`#TikiShippingPolicy`):

```
handlingTime 0–1 ngày  +  transitTime 1–4 ngày   →  SLA công bố = 1–5 ngày
```

→ **Promise là hằng số, không biến thiên theo đơn.** Đây là khác biệt cốt lõi so với Olist và là lý do đề tài phải được định nghĩa lại (xem §4).

### 3.5 ⚠️ Thiên lệch chọn mẫu — và cách khắc phục

Mẫu thô cực lệch: **5★ chiếm 95.9%** (3655/3811), "Giao trễ hẹn" chỉ 1.3%.
Nếu cào ngây thơ thì biến mục tiêu gần như không có phương sai → **đề tài chết**.

**Lối thoát đã kiểm chứng:** API nhận `sort=stars|N`:

```
sort=stars|1      → chỉ review 1 sao   (đã test: total=1, ratings=[1] ✅)
sort=stars|1|2|3  → "Chưa hài lòng"
sort=stars|5      → chỉ review 5 sao
sort=id|desc      → mới nhất
```

Và response luôn kèm **histogram `stars` ở cấp quần thể** (`stars["1"]["count"]`…) — **đếm thật, không thiên lệch**.

→ **Thiết kế lấy mẫu bắt buộc: phân tầng theo sao (stratified sampling) với kích thước tầng đã biết.**
1. Đọc histogram `stars` của mỗi sản phẩm → biết đúng `N_h` của từng tầng.
2. Cào **vét cạn** các tầng hiếm (1★, 2★, 3★), **lấy mẫu** tầng 5★.
3. Gắn trọng số `w_h = N_h / n_h` → mọi ước lượng quần thể phải dùng trọng số này.

Đây không phải mẹo vặt — đây là **survey sampling đúng bài** và là một phần lớn của 3đ "chứng minh đúng". Mọi con số quần thể báo cáo mà **không** kèm trọng số đều là sai.

## 4. Định nghĩa Delivery Promise Gap — ĐÃ CHỐT (phương án C)

Đề cương gốc định nghĩa `Gap = ngày giao thực tế − ngày giao dự kiến`, lấy từ Olist. Trên dữ liệu Tiki, **ngày giao dự kiến theo từng đơn không tồn tại công khai** (§3.4). Nhóm đã chốt **phương án C** (2026-09-07):

> **Bước 1 — Gap suy ra:** `gap_days = lead_time_thực_tế − 5` (SLA công bố 1–5 ngày).
> **Bước 2 — Đối chiếu chéo:** so `gap_days > 0` với nhãn khách tự báo `dr_thoi_gian` ∈ {Giao đúng hẹn, Giao trễ hẹn}.

Bảng chéo 2×2 này là **trục chính của cả đồ án**. Hai ô lệch nhau chính là "bẫy SLA":

| | Khách: đúng hẹn | Khách: trễ hẹn |
|---|---|---|
| **Trong SLA** | đồng thuận | **BẪY 1** — SLA "đạt" mà khách bất mãn |
| **Vượt SLA** | **BẪY 2** — SLA "trượt" mà khách hài lòng | đồng thuận |

Đã xác nhận trên pilot (n=113): cả hai ô lệch **đều khác 0** (2 và 3 ca) → phép đo khả thi, chỉ cần quy mô.

Ưu thế phương pháp: ta có **cả đại lượng suy ra lẫn nhãn quan sát được** → kiểm chứng chéo được, đúng tầng 9 "external sanity check" của khung §7. Olist không làm được điều này vì không có nhãn khách tự báo.

**Không được âm thầm đổi định nghĩa Gap.** Nếu đổi, cập nhật mục này + `docs/FEASIBILITY.md` §5 và nói rõ với nhóm.

### 4.1 Luận điểm trung tâm — CÒN ĐỂ MỞ

Pilot gợi ý hướng này; **mẻ đầy đủ 117.819 review đã xác nhận** (2026-09-09):

> Trung vị có trọng số **1.30 ngày** so với trần SLA **5 ngày** — chỉ bằng **26%**
> của trần. **92,73% đơn đạt SLA.** "Tỉ lệ đạt SLA" gần như không có sức phân biệt
> nào để làm chỉ số vận hành.

Con số này **mạnh hơn** ước lượng pilot sau khi sửa lỗi múi giờ (§3.2): SLA còn lỏng
hơn ta tưởng. Hai ô lệch của bảng chéo vẫn khác 0 sau khi sửa (`FINDINGS.md` §3).

→ Luận điểm đã đủ căn cứ để đưa vào báo cáo/slide, **kèm ba cảnh báo ở §9**. Riêng
tỉ lệ vượt SLA thì phải trích **theo năm**, không được trích số gộp.

## 5. Kiến trúc code (2đ)

Nộp **một repo**, không phải một notebook 800 cell.

```
src/collect/    # BaseScraper (abstract) → TikiListingScraper, TikiReviewScraper
src/clean/      # chuẩn hoá, xử lý lead time âm, khử trùng lặp
src/validate/   # data contract — bộ assert chạy sau mỗi bước
src/features/   # gap, phân tầng trọng số, feature engineering
src/models/     # BaseModel với fit/predict chung — baseline.py, ml.py
src/evaluate/   # backtest, bootstrap CI, calibration, ablation
notebooks/      # CHỈ kể chuyện — gọi hàm từ src/, không chứa logic
tests/          # pytest cho clean & features
data/raw/       # BẤT BIẾN. Không bao giờ sửa file trong này.
```

**Quy ước bắt buộc:**
- **OOP theo bài giảng Buổi 1**: `BaseScraper` abstract → các scraper con. `BaseModel` chung interface để so sánh mô hình công bằng.
- **NumPy vectorization theo bài giảng Buổi 2**: dùng `np.searchsorted`/broadcasting, kèm `%timeit` đối chiếu vòng lặp thuần → đưa thẳng vào slide.
- Notebook không chứa logic. Logic nằm trong `src/`, có test.
- `random_state` cố định, `requirements.txt` ghim version.
- Không hardcode đường dẫn/tham số → `config.yaml`.
- File 200–400 dòng là vừa, **tối đa 800**.

## 6. Đạo đức cào dữ liệu — KHÔNG THƯƠNG LƯỢNG

- `time.sleep(1)` giữa các request. **Không chạy song song nhiều luồng.**
- User-Agent khai rõ là đồ án môn học.
- **Cache xuống đĩa, cào đúng một lần.** Chạy lại phải đọc cache, không đập vào server.
- Tôn trọng mọi `Disallow` trong §3.1.
- **Không public raw dump.** `data/raw/` phải nằm trong `.gitignore`.
- Trích dẫn nguồn đầy đủ trong báo cáo.

## 7. Khung "chứng minh đúng" — đây là 3đ

Mọi kết luận phải đi qua khung này:

| # | Tầng | Công cụ |
|---|---|---|
| 1 | **Data contract** | Bộ `assert`: kiểu, khoảng giá trị, khoá duy nhất, tỉ lệ null, số dòng |
| 2 | **Reproducibility** | seed cố định, ghim version, checksum file raw |
| 3 | **Leakage audit** | Chia theo thời gian; scaler/encoder `fit` **bên trong** fold |
| 4 | **Baseline ladder** | naive → tuyến tính → cây → tuned. **Không baseline thì metric vô nghĩa** |
| 5 | **Significance** | Paired bootstrap / permutation test → **CI 95% cho *chênh lệch* metric** |
| 6 | **Uncertainty & calibration** | Kiểm tra **độ phủ thực tế** của khoảng dự báo |
| 7 | **Error analysis & ablation** | Lỗi theo phân khúc; ΔMetric khi bỏ từng nhóm feature |
| 8 | **Robustness & sensitivity** | Đổi ngưỡng, đổi cách impute → kết luận có đổi không? |
| 9 | **External sanity check** | **Đối chiếu Gap tính được với nhãn `delivery_rating` khách tự báo** |

Riêng đề tài này còn thêm một tầng đặc thù: **mọi ước lượng quần thể phải dùng trọng số phân tầng §3.5.**

Câu nói khi bảo vệ:
> "Nhóm em không hỏi *mô hình đạt bao nhiêu điểm*, nhóm em hỏi *bao nhiêu phần trăm con số đó là thật*."

## 8. Cạm bẫy đã biết — đọc trước khi code

1. **Đừng dùng `curl`** với tiki.vn → 403. Dùng `requests.Session()`.
2. **Đừng cào ngây thơ theo `sort` mặc định** → mẫu 96% 5 sao, vô dụng. Phải phân tầng (§3.5).
3. **Đừng `dropna()` cho 1.00% lead time âm.** Đã điều tra xong (`FINDINGS.md` §6.2) — giữ nguyên cách gắn cờ, và §6.3 là bằng chứng loại chúng ra không làm lệch kết luận.
4. **Đừng dùng KFold ngẫu nhiên** trên dữ liệu có thời gian → leakage. Chia theo thời gian.
5. **Đừng báo cáo tỉ lệ thô từ mẫu phân tầng** mà không nhân trọng số → sai hệ thống.
6. **Không có địa lý khách hàng** (`region` = 0%) → không tái tạo được phân tích vùng miền của Olist. Ghi rõ là *limitation*, đừng bịa.
7. `delivery_rating` chỉ có ở 18.1% review → mọi phân tích dùng nó phải nêu rõ cỡ mẫu và kiểm tra xem nhóm có nhãn có khác nhóm không có nhãn không (**missing not at random**).
8. **⚠️ Tiki trả về hai đồng hồ.** Epoch (`purchased_at`, `created_at`) là UTC; chuỗi trong `timeline` là **giờ Việt Nam**. Trừ thẳng hai loại này vào nhau là lỗi im lặng — nó không làm gãy gì cả, chỉ cộng dư đúng 7 giờ vào mọi lead time. Luôn đi qua `parse_timestamps()`, đừng tự parse. `FINDINGS.md` §6.1.
9. **Đừng so hai nhóm con bằng đếm thô.** Mẫu phân tầng theo sao phóng đại tỉ lệ sao thấp trong *mọi* nhóm con — đã hai lần dẫn tới kết luận ngược (`FINDINGS.md` §5 và §6.3). Nhân trọng số, rồi mới so.

## 9. Trạng thái hiện tại (cập nhật 2026-09-09, sau khi sửa lỗi múi giờ)

- [x] Đọc đề tài, kiểm chứng khả thi dữ liệu bằng thực nghiệm → `docs/FEASIBILITY.md`
- [x] Dựng khung repo + CLAUDE.md
- [x] Chốt định nghĩa Gap = phương án C (§4)
- [x] Scraper phân tầng + cache → `src/collect/`
- [x] **Mẻ dữ liệu chính: 117.819 review · 2.438 sản phẩm · 366 nhà bán · 10 ngành**
      (mẻ cũ 66.851 review lưu ở `data/archive_v1/`)
- [x] Data contract → `src/validate/contract.py` (22/22 đạt, đã xếp lại đúng tầng)
- [x] Cleaning → `src/clean/reviews.py` (97,33% dòng phân tích được)
- [x] **Pipeline làm sạch tái lập bằng 1 lệnh** → `python3 -m src.clean.run_clean`
- [x] Thống kê có trọng số + bootstrap theo cụm + **n_eff** → `src/evaluate/stats.py`
- [x] **Phân tích tái lập bằng 1 lệnh** → `python3 -m src.evaluate.run_analysis`
- [x] **Bằng chứng "vì sao Tiki"** → `scripts/probe_platforms.py` + `FEASIBILITY.md §6`
- [x] **Sửa lỗi múi giờ UTC/UTC+7** — mọi lead time từng phồng 7 giờ (`FINDINGS.md` §6.1)
- [x] **Điều tra 1.181 dòng `lead_days` âm** → `src/evaluate/lead_anomaly.py` + `FINDINGS.md` §6.2–6.4
- [x] Kết quả + hai bài học phương pháp → `docs/FINDINGS.md`
- [x] Test: 39/39 đạt (`python3 -m pytest`)
- [ ] Phân tích độ nhạy chặn trên/dưới cho thiên lệch MNAR ← **việc đáng làm nhất còn lại**
- [ ] Cào thêm sản phẩm sau 2023 để nâng `n_eff` của hai ước lượng phụ (cả hai giờ < 100)
- [ ] Phân rã theo ngành hàng / nhà bán
- [ ] Modeling + thang baseline + ablation
- [ ] EDA hình ảnh + báo cáo + slide

### Lệnh tái lập toàn bộ

```bash
python3 -m src.clean.run_clean        # raw -> processed, chạy contract 2 tầng
python3 -m src.evaluate.run_analysis  # mọi con số trong FINDINGS.md §1-§4
python3 -m src.evaluate.lead_anomaly  # mọi con số trong FINDINGS.md §6
python3 -m pytest                     # 39 test
```

### Kết quả đã xác lập (chi tiết: `docs/FINDINGS.md`)

**Kết luận chính** — chênh lệch sức phân biệt rating (nhãn khách − chỉ báo SLA):

> **+0,234 · KTC 95% [0,113 · 0,393] · n = 11.391 · n_eff = 1.960 · 1.914 cụm**

KTC không chứa 0. Đây là ước lượng duy nhất sống sót qua **cả hai** phép thử độc
lập — mẫu tăng 76%, rồi thang thời gian dịch 0,29 ngày vì sửa múi giờ:

| | Mẫu 66.851 | Mẫu 117.819 | + sửa múi giờ |
|---|---|---|---|
| Chênh lệch sức phân biệt | +0,209 | +0,241 | **+0,234** |
| KTC 95% | [0,06 · 0,44] | [0,129 · 0,398] | [0,113 · 0,393] |

**Luận điểm trung tâm (đã chốt):** trung vị giao hàng **1,30 ngày** chỉ bằng 26%
trần SLA 5 ngày; **92,73% đơn đạt SLA** → "tỉ lệ đạt SLA" gần như vô dụng làm chỉ
số vận hành.

Hai ước lượng phụ, dùng dưới dạng khoảng — **không** làm kết luận chính:

| | Ước lượng | KTC 95% | n_eff |
|---|---|---|---|
| % than phiền trễ hẹn đến từ đơn trong SLA | 86,3% | [77,3 · 93,0] | ⚠️ 48 |
| % đơn vượt SLA mà khách nói đúng hẹn | 88,4% | [80,9 · 94,0] | ⚠️ 99 |

### ⚠️ Bốn cảnh báo bắt buộc khi trích dẫn

1. **Mọi con số thời gian giao hàng in trước 2026-09-09 đều phồng lên 7 giờ.**
   Vượt SLA **7,27%** (không phải 8,54%) · trung vị **1,30** (không phải 1,59) ·
   p90 **4,35** (không phải 4,64). Thấy số cũ ở slide/báo cáo → sửa. `FINDINGS.md` §6.1.

2. **Con số 91,9% / 94,3% (mẫu cũ) và 81,8% (mẫu mới, đồng hồ cũ) đã bị bác bỏ.**
   Giá trị hiện hành là 86,3% và 88,4%, cả hai đều `n_eff < 100` → **mong manh**,
   chỉ trích dưới dạng khoảng.

3. **Nhãn `delivery_rating` là MNAR** — chỉ có từ 2023, độ phủ 9,8%, nhóm có nhãn
   giao nhanh hơn nhóm không nhãn. Mọi ước lượng ở trên là **trên nhóm có nhãn**,
   không phải toàn quần thể.

4. **Tỉ lệ vượt SLA gộp 7,27% KHÔNG phải tình hình hiện tại.** Mẫu trải 12 năm,
   2021+2022 chiếm 47% và riêng 2021 là 17,48% (giãn cách). Hiện trạng 2024–2026
   nằm trong khoảng **1,6% – 4,0%**. Dùng bảng theo năm ở `FINDINGS.md` §2.

### Hai nguyên tắc rút ra — áp dụng cho mọi ước lượng sau này

**1. Số dòng không phải lượng thông tin.** Với mẫu phân tầng, luôn báo cáo `n_eff`
cùng ước lượng. `Estimate` đã tự làm việc này và tự gắn cờ `⚠️ MONG MANH` khi
`n_eff < 100` — đừng gỡ cảnh báo đó, hãy đi lấy thêm dữ liệu. (Cơ chế này vừa tự
bắt được ước lượng thứ hai khi nó tụt từ 129 xuống 99.)

**2. Đếm thô từ mẫu phân tầng dẫn tới kết luận ngược.** Đã xảy ra hai lần: một lần
với `n_eff` (§5), một lần khi hỏi "loại dòng lead âm ra có cắt mất khách bất mãn
không" — đếm thô nói *có* (10,58% vs 5,43% một sao), nhân trọng số nói *không*
(1,87% vs 1,47%, và rating trung bình còn **cao hơn**). Nhân trọng số trước, rồi
mới so sánh.

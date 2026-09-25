# Đánh giá bản kế hoạch + Phân chia công việc

> Đối chiếu `Plan data science.docx` với trạng thái thật của repo (mẻ 203.510 review),
> rồi chia việc còn lại cho 3 người. Mọi con số dưới đây đã kiểm lại trực tiếp trên
> `data/processed/reviews_clean.parquet`, không chép từ tài liệu.
>
> **Ngày:** 2026-09-22 · **Căn cứ:** `docs/FINDINGS.md`, `CLAUDE.md` §9–§9.1, `README.md`

---

# PHẦN 1 — ĐÁNH GIÁ BẢN NHẬN XÉT

## Kết luận một dòng

**Hướng nhìn đúng, nhưng viết trên mẻ dữ liệu đã bị thay, và hai đề xuất lớn nhất
sẽ làm hỏng phần mạnh nhất của đồ án nếu làm đúng nguyên văn.**

## 1.0 Lỗi hệ thống: toàn bộ số trong file là của mẻ 117.819

Đây không phải lỗi rải rác — nó là **một lỗi duy nhất lặp lại ở mọi mục**:

| Nơi | File .docx ghi | Giá trị hiện hành (mẻ 203.510) |
|---|---|---|
| Bảng chéo (Biểu đồ 8) | 10.859 · 207 · 274 · 51 → **tổng 11.391** | 15.160 · 252 · 339 · 65 → **tổng 15.816** |
| Độ phủ `delivery_rating` | "khoảng 9,8%" | **7,9%** |

Con số 11.391 chính là `n` có nhãn của **mẻ 117.819** (`FINDINGS.md` §3 ghi rõ
"mẻ 117.819 trước đây: 11.391"). Nghĩa là file này được viết trước khi mẻ 203.510
ra đời, và chưa ai chạy lại.

> **Hệ quả thực tế:** nếu bê bảng 10.859/207/274/51 lên slide thì đó là bảng của một
> mẻ dữ liệu không còn tồn tại trong repo. Đây đúng là cái bẫy mà `README.md` đã đặt
> ra nguyên tắc để chặn: *"Mọi con số trong tài liệu phải sinh ra bằng một lệnh."*
>
> **Việc phải làm trước mọi việc khác:** chạy `python3 -m src.evaluate.run_analysis`
> rồi cập nhật lại file kế hoạch. Task **A0**.

---

## 1.1 "Chỉ nên lấy 2022–2026 vì 2021 bị giãn cách" → ⚠️ **Chẩn đoán đúng, phương thuốc sai**

### Đúng ở chỗ nào

2021 đúng là bất thường và đúng là đang bóp méo con số gộp:

* 2021 chiếm 52.044/198.355 = **26,2%** mẫu phân tích được
* Tỉ lệ vượt SLA 2021 là **16,75%**, trong khi 2024–2026 chỉ **1,4%–3,3%**
* Con số gộp 7,12% **bị 2021 kéo lên** — trích nó như "tình hình hiện nay" là sai

Nhận xét này bắt đúng vấn đề. Nhưng repo **đã xử lý xong vấn đề này** theo cách
khác: `CLAUDE.md` §9 cảnh báo #4 và `FINDINGS.md` §2 đã cấm trích số gộp và bắt
buộc dùng bảng theo năm.

### Sai ở chỗ nào — ba lý do, đã đo

**Lý do 1 — Cắt 2021 không cứu được kết luận chính, vì kết luận chính chưa bao giờ
dùng 2021.**

Nhãn `delivery_rating` gần như không tồn tại trước 2023. Đếm thật:

| Năm | n | có nhãn | % |
|---|---|---|---|
| 2014–2018 | 1.697 | **0** | 0,00% |
| 2019 | 4.254 | 1 | 0,02% |
| 2020 | 20.311 | 3 | 0,01% |
| 2021 | 52.044 | **0** | 0,00% |
| 2022 | 55.274 | **0** | 0,00% |
| 2023 | 25.456 | 4.542 | 17,84% |
| 2024 | 17.896 | 5.108 | 28,54% |
| 2025 | 14.075 | 3.938 | 27,98% |
| 2026 | 7.348 | 2.224 | 30,27% |

15.812/15.816 review có nhãn nằm ở **2023+**. Kết luận chính (+0,208 điểm rating,
`FINDINGS.md` §4) và bảng chéo 2×2 **đã là phân tích 2023–2026 rồi**. Cắt xuống
2022 không đổi một con số nào trong đó — nó chỉ xoá 2021 và 2022, hai năm **không
đóng góp một dòng có nhãn nào**.

**Lý do 2 — Cắt 2021 xoá mất 61% toàn bộ bằng chứng về vượt SLA.**

Đây là con số quyết định, đo trực tiếp:

| | Tổng | Riêng 2021 | Tỉ trọng 2021 |
|---|---|---|---|
| Ca vượt SLA (đếm thô) | 16.317 | 9.531 | **58,4%** |
| Ca vượt SLA (có trọng số) | 30.443 | 18.618 | **61,2%** |

Cắt còn 2022–2026 thì:
* mất **78.306/198.355 dòng = 39,5%** dữ liệu
* ca vượt SLA còn lại **4.693/16.317 = 28,8%** — tức **mất 71,2%**

Biến `sla_breach` là biến trung tâm của cả đồ án, và lớp dương của nó chỉ chiếm
7,12%. Cắt đi 71% lớp dương đó thì:
- phần Modeling gần như không còn ví dụ "giao trễ" để học
- phần phân rã theo ngành/seller (chính mục 5 trong file kế hoạch) sập luôn
- mọi khoảng tin cậy nở rộng ra

**Lý do 3 — Xoá dữ liệu cho số đẹp đi ngược nguyên tắc đang là điểm mạnh nhất của repo.**

`CLAUDE.md` §8 cạm bẫy #3 và `README.md` đều ghi: *"Không xoá dòng lỗi trong im
lặng — gắn cờ, đếm được, báo cáo được."* Repo đã hai lần được cứu nhờ nguyên tắc
này (§6.2, §6.3). Cắt 2021 vì "nó xấu" là đúng cái việc đó, chỉ khác quy mô.

### Nên làm gì thay thế

Biến câu hỏi này thành **tầng 8 của khung chứng minh đúng** (`CLAUDE.md` §7 —
Robustness & sensitivity), tức là biến một quyết định cảm tính thành một bằng chứng:

> Chạy lại **đúng kết luận chính** trên 4 lát cắt: **toàn mẫu · bỏ 2021 · 2022+ ·
> 2023+**, rồi in 4 ước lượng cạnh nhau kèm KTC và `n_eff`.

Nếu 4 con số nằm chồng lên nhau → "kết luận không phụ thuộc vào việc có giữ 2021
hay không", và đó là một câu **mạnh hơn nhiều** so với việc im lặng cắt dữ liệu.
Nếu chúng lệch nhau → ta vừa tìm ra một phát hiện thật.

Chi phí: ~1 buổi. Đây là task **A7**.

Và thêm một cơ hội mà cách cắt bỏ sẽ đánh mất: **2021 là chế độ vận hành duy nhất
trong dữ liệu có tỉ lệ vượt SLA hai chữ số.** `rating` thì có đủ cả 12 năm. Nên
2021 trả lời được một câu mà không năm nào khác trả lời được:

> *Khi giao hàng thật sự trễ trên diện rộng, `lead_days` có ăn vào rating không?*

Nếu 2021 cho thấy lead_days **có** ảnh hưởng rating rõ rệt, còn 2024–2026 thì
không, thì luận điểm sắc hơn hẳn: **SLA không vô dụng — nó chỉ vô dụng khi ngưỡng
đặt quá lỏng so với thực tế vận hành.** Đó là task **C13**.

---

## 1.2 "SLA không cụ thể theo từng đơn, tạm để vậy" → ✅ **Lo ngại đúng, nhưng lời giải tốt hơn đang nằm sẵn trong code**

### Đúng ở chỗ nào

Hoàn toàn đúng, và repo đã biết: `CLAUDE.md` §4 (phương án C), `FEASIBILITY.md` §5,
`FINDINGS.md` §7 đều khai báo rõ "lời hứa là SLA site-wide 1–5 ngày, không phải cam
kết theo từng đơn". Kết luận "cần tài khoản order/seller → không khả thi" cũng đúng
với ràng buộc đạo đức ở `CLAUDE.md` §6 (cấm đụng `/api/v2/me/`, `/order/tracking`).

### Thiếu ở chỗ nào

"Tạm để vậy" là bỏ lỡ hai thứ:

**Thứ nhất — đây không phải điểm yếu, đây là đề tài.** Luận điểm của cả đồ án là
*"doanh nghiệp đo chất lượng giao hàng bằng tỉ lệ đạt SLA site-wide, và nó đo sai
thứ."* Nếu có SLA theo từng đơn thì đó là **một đề tài khác**. Nên đừng xin lỗi vì
nó — hãy khai báo nó là phạm vi nghiên cứu.

**Thứ hai — `compute_gap()` đã nhận `sla_days` làm tham số rồi.**

```python
def compute_gap(df, sla_days: float = SLA_MAX_DAYS)   # src/clean/reviews.py
```

Nghĩa là câu hỏi *"nếu ngưỡng 5 ngày là tuỳ tiện thì sao?"* trả lời được bằng
khoảng 20 dòng code: chạy lại toàn bộ kết luận ở **sla_days ∈ {3, 4, 5, 6, 7}** và
in ra một bảng. Kết quả sẽ có dạng:

> *"Kết luận không đổi dấu ở bất kỳ ngưỡng SLA nào từ 3 đến 7 ngày."*

Đó là câu chặn đứng câu hỏi hiển nhiên nhất mà hội đồng sẽ hỏi. Rẻ, và là tầng 8
của khung chứng minh. Task **A6**.

**Thứ ba — còn đúng một phép dò chưa làm.** Trước khi đóng vĩnh viễn câu hỏi này,
bỏ 1 giờ dò endpoint `product detail` xem có trường `delivery_estimate` /
`handling_time` cấp sản phẩm không (endpoint công khai, không cần tài khoản). Nếu
có → SLA theo sản phẩm, tốt hơn site-wide nhiều. Nếu không → ghi bằng chứng vào
`FEASIBILITY.md` và **đóng câu hỏi này lại có căn cứ**, thay vì để lửng "chưa biết
cách nào ok hơn". Task **A5**.

---

## 1.3 Phân tích MNAR → ✅ **Hai câu hỏi đặt rất trúng, nhưng thiếu một phân biệt sẽ làm hỏng kết quả**

### Đúng ở chỗ nào

Hai câu hỏi trong file:
1. *Nếu nhóm không nhãn hành vi khác nhóm có nhãn thì kết luận chính còn đúng không?*
2. *Trong những giả định hợp lý, "SLA không phản ánh trải nghiệm khách hàng" có đứng vững không?*

Đây **chính xác** là câu hỏi đúng, và nó trùng với việc repo đã chọn là "đáng làm
nhất còn lại" (`FINDINGS.md` §8 mục 2). Không có gì phải sửa về hướng.

### Thiếu ở chỗ nào — và đây là chỗ dễ mất công vô ích nhất

File kế hoạch nói "MNAR" như **một** cơ chế. `CLAUDE.md` §9.1 đã đo ra **hai**:

| Cơ chế | Quy mô | Chặn được? |
|---|---|---|
| **Thiếu do thiết kế** — trường chưa tồn tại trước 2023, độ phủ đúng 0,00% | **66% quần thể** | ❌ Không bao giờ |
| **Chọn lọc thật** — trong 2023+, 25,7% (có trọng số) có nhãn | 34% quần thể | ✅ Chỉ ở đây |

**Hệ quả nếu bỏ qua phân biệt này:** chạy chặn Manski trên toàn mẫu 12 năm thì 74%
số dòng không có nhãn → chặn sẽ rộng đến mức chứa cả 0 → đọc nhầm thành *"kết luận
chính sụp đổ"*. Nó không sụp — chỉ là đặt sai câu hỏi cho 66% số dòng chưa bao giờ
có cơ hội mang nhãn.

**Chặn phải chạy trong kỷ nguyên 2023+, không phải toàn mẫu.**

Kế hoạch chi tiết 4 bước (Manski → quét θ → bootstrap → test) đã viết sẵn ở
`CLAUDE.md` §9.1, kèm một phép **tự kiểm bắt buộc**: hàm quét θ chạy trên riêng
nhóm có nhãn phải tái lập đúng **0,2082**; không khớp thì hàm sai. Đừng viết lại
kế hoạch — cứ theo nó. Task **A1–A4**.

---

## 1.4 Danh sách 8 biểu đồ EDA → 🔶 **Khung tốt, nhưng thiếu trọng số và thiếu 4 hình mạnh nhất**

### Đúng ở chỗ nào

* Chia 4 nhóm theo chủ đề, mỗi hình có một câu "mục tiêu" — đúng nguyên tắc
  *"mỗi slide một câu kết luận, biểu đồ là bằng chứng"*.
* Ưu tiên hình sớm là quyết định đúng — khớp với ghi chú của Hiển trong `CLAUDE.md`
  §9 rằng giảng viên thích trình bày bằng notebook.
* Đánh dấu Biểu đồ 8 (bảng chéo) là quan trọng nhất — đúng, đó là trục của đồ án.

### Sai ở chỗ nào — lỗi nặng nhất

**Cả 8 biểu đồ đều là biểu đồ đếm thô, và mẫu này là mẫu phân tầng theo sao.**

`CLAUDE.md` §8 cạm bẫy #5 và #9 ghi thẳng: *"Đừng báo cáo tỉ lệ thô từ mẫu phân
tầng"* và *"Đếm thô từ mẫu phân tầng đã hai lần dẫn tới kết luận ngược."*

Cụ thể với Biểu đồ 6 (phân phối rating): vẽ thô sẽ ra **8,6% review ≤3★**, trong
khi quần thể thật chỉ **4,0%**. Tức là hình đó nói dối gấp hơn hai lần, và nó nói
dối theo đúng chiều có lợi cho luận điểm của nhóm — giám khảo mà bắt được thì mất
nhiều hơn cả 3 điểm chứng minh.

Histogram/boxplot `lead_days` cũng vậy: review 1★ đang bị lấy mẫu vét cạn, mà
review 1★ giao chậm hơn → phân phối thô lệch phải hơn thực tế.

→ **Quy tắc bắt buộc cho mọi hình: hoặc vẽ có trọng số, hoặc ghi rõ trên hình
"mẫu — chưa trọng số, chỉ để xem hình dạng".** Không có lựa chọn thứ ba. Task **B1**.

### Thiếu ở chỗ nào — 4 hình mạnh nhất của đồ án không có trong danh sách

Repo đã có sẵn bằng chứng cho 4 hình này, nhưng file kế hoạch không liệt kê hình nào:

| Hình còn thiếu | Vì sao nó là hình mạnh nhất |
|---|---|
| **Forest plot: kết luận chính qua 3 mẻ** (+0,209 → +0,241 → +0,234 → +0,208 với KTC) | Đây là hình trả lời "sao tin được?" — 4 chấm nằm chồng lên nhau qua 2 lần mở rộng mẫu và 1 lần sửa thang đo. Không nhóm nào trong lớp có hình này. |
| **Trước/sau sửa múi giờ** — phân phối `lead_days` dịch 0,2917 ngày, kèm phép đo 7,0000h ± 0,000000 | Câu chuyện "một API, hai đồng hồ" là phần hay nhất của đồ án và hiện chưa có hình nào. |
| **`n` so với `n_eff`** — cột 182 cạnh cột 25 | Bài học "số dòng ≠ lượng thông tin" gói gọn trong một hình. |
| **Độ phủ nhãn theo năm** — vách 0% dựng đứng trước 2023 | Toàn bộ câu chuyện MNAR trong một hình, và nó giải thích luôn vì sao không cắt 2021 (§1.1). |

### Sai nhỏ hơn

* **Biểu đồ 7 "Rating vs lead_days" chưa định nghĩa được dạng hình.** Scatter 198k
  điểm là một vệt mực. Phải là: chia bin `lead_days` (0–1, 1–2, 2–3, 3–5, 5–10, >10)
  × rating trung bình **có trọng số** × dải KTC bootstrap. Task **B8**.
* **Biểu đồ 5 "SLA breach theo năm"** phải chú thích `n` từng năm ngay trên hình,
  nếu không người đọc không biết 2017 chỉ có 541 dòng còn 2021 có 52.044.

---

## 1.5 "Phân rã bẫy SLA theo ngành hàng / seller" → ❌ **Theo cách đang viết thì không chạy được — đã đo**

Đây là mục sai nặng nhất, và sai theo đúng cái bẫy repo đã dính hai lần.

### Vấn đề: hai ô lệch chỉ có 591 dòng cho toàn bộ 10 ngành

Hai ô "bẫy SLA" cộng lại chỉ có **252 + 339 = 591** dòng. Chia cho 10 ngành:

| Ngành hàng | Số ca lệch |
|---|---|
| Nhà cửa - Đời sống | 327 |
| Làm đẹp - Sức khỏe | 73 |
| Sách tiếng Việt | 56 |
| Mẹ và bé | 55 |
| Thời trang nữ | 35 |
| Điện gia dụng | 23 |
| Nhà sách tiki | 15 |
| Thể thao - Dã ngoại | **5** |
| Điện thoại - Máy tính bảng | **2** |
| Bách hóa online | **0** |

Đây là **đếm thô**. Nhân trọng số vào thì `n_eff` còn thấp hơn nữa — và `Estimate`
sẽ tự bật `⚠️ MONG MANH` cho gần hết bảng. Xếp hạng "ngành nào SLA đáng tin hơn"
trên 2 hoặc 5 quan sát là xếp hạng nhiễu.

### Vấn đề với seller còn nặng hơn

File kế hoạch viết "Top sellers có SLA mismatch cao / Bottom sellers". Đo thật:

* 591 ca lệch trải trên **105 seller**
* **seller_id = 1 (Tiki Trading) một mình chiếm 402/591 = 68%**
* 104 seller còn lại chia nhau 189 ca — seller đông nhất trong số đó có **10 ca**

Bảng xếp hạng top/bottom seller trên 1–10 quan sát mỗi seller là **bảo đảm ra
nhiễu**. Và nó sẽ ra một bảng trông rất thuyết phục — đó mới là chỗ nguy hiểm.

### Nên làm gì thay thế — và hoá ra lại hay hơn

`seller_id = 1` là **Tiki Trading**, tức gian hàng tự vận hành của chính sàn. Vậy
phép tách đúng không phải 366 seller, mà là **2 nhóm**, và cả hai nhóm đều đủ dữ liệu:

| | Trong SLA / khách đúng hẹn | Trong SLA / khách **trễ** | Vượt SLA / khách **đúng hẹn** | Vượt SLA / khách trễ |
|---|---|---|---|---|
| **Tiki Trading** (n=11.573) | 11.127 | **199** | **203** | 44 |
| **Bên thứ ba** (n=4.243) | 4.033 | **53** | **136** | 21 |

Câu hỏi nghiên cứu trở thành:

> **Bẫy SLA có giống nhau giữa hàng sàn tự vận hành và hàng nhà bán bên thứ ba không?**

Nhìn sơ bộ đã thấy hình dạng khác nhau: tỉ lệ "báo động giả" (vượt SLA mà khách nói
đúng hẹn) ở bên thứ ba là 136/157 = 86,6%, còn Tiki Trading là 203/247 = 82,2%. Cần
nhân trọng số và bootstrap mới kết luận được — nhưng đây là một câu hỏi **trả lời
được**, khác hẳn bảng xếp hạng 366 seller.

Với ngành hàng: gộp còn **3 nhóm** (Nhà cửa-Đời sống · Sách · Còn lại) hoặc chỉ báo
cáo 2–3 ngành đủ `n_eff`, phần còn lại ghi thẳng "không đủ dữ liệu để kết luận".
Task **C14, C15**.

---

## 1.6 Feature Engineering → ⚠️ **Có một lỗi cộng tuyến hoàn hảo, và bỏ sót phần thú vị nhất**

### Lỗi phải sửa: `lead_days` + `gap_days` + `sla_breach` không thể đứng chung một mô hình

Theo `src/clean/reviews.py`:

```
gap_days   = lead_days - 5.0
sla_breach = gap_days > 0
```

`gap_days` là **hàm affine của `lead_days`** → tương quan đúng bằng 1,0. Đưa cả hai
vào Logistic Regression thì hệ số chia nhau tuỳ tiện và **không diễn giải được** —
mà mục đích file kế hoạch ghi cho Model 1 lại chính là "đây là model giải thích
chính". `sla_breach` thì là bản nhị phân hoá của cùng một biến.

### Nhưng đúng chỗ này lại là cơ hội — biến lỗi thành phép chứng minh

Ba biến đó không nên đứng chung **một** mô hình. Chúng nên là **ba mô hình riêng**,
rồi đem so:

| Mô hình | Chỉ dùng | Nó đại diện cho |
|---|---|---|
| M1a | `sla_breach` (nhị phân) | **Cách doanh nghiệp đang đo** |
| M1b | `lead_days` (liên tục) | Thông tin đầy đủ về thời gian |
| M1c | nhãn khách `customer_says_late` | Thứ khách thật sự cảm nhận |

So sức dự đoán 3 mô hình này bằng **paired bootstrap theo cụm sản phẩm** chính là
kết luận §4 phát biểu lại dưới dạng dự đoán — và lần này có thể đo bằng PR-AUC thay
vì chênh lệch rating. Nếu M1a gần như không thắng baseline trong khi M1b và M1c
thắng rõ → **"tỉ lệ đạt SLA vứt mất thông tin"** được chứng minh hai lần bằng hai
phương pháp độc lập.

Đó là ablation ở tầng 7 của khung chứng minh, và nó là phần đáng giá nhất của
Modeling. Task **C9**.

### Bỏ sót: các biến đã có sẵn trong dữ liệu mà không ai dùng

Danh sách feature trong file chỉ có 8 biến. Bảng đã sạch còn sẵn:

* `review_lag_days` — khách viết review sau bao lâu (đã tính trong `compute_durations`)
* `dr_shipper`, `dr_dong_goi`, `dr_gio_giao` — 3 nhãn khách tự báo khác, cùng độ phủ
* `n_images`, `is_photo`, `thank_count` — cường độ đầu tư của review
* `customer_purchased_flag`, `customer_joined` — thâm niên khách
* `la_tiki_trading` — biến nhị phân từ §1.5, gần như chắc chắn có tín hiệu

### Ba cảnh báo kỹ thuật

1. **`seller_id` 366 mức không đưa thô vào model.** Target-encoding phải `fit`
   **bên trong** fold, nếu không là leakage (`CLAUDE.md` §7 tầng 3).
2. **Chưa ai quyết định có train có trọng số hay không.** Phải chốt và ghi lý do:
   train không trọng số (học tốt hơn trên lớp hiếm) nhưng **đánh giá phải có trọng
   số** (ước lượng quần thể). Task **C2**.
3. **`year` làm feature + chia theo thời gian**: nếu train ≤2024 thì model chưa bao
   giờ thấy `year=2025`. Dùng biến phái sinh (khoảng cách tới mốc) chứ đừng dùng năm thô.

---

## 1.7 Modeling → 🔶 **Mới có 2/5 bậc thang, và `is_low_rating` mất cân bằng nặng**

### Thiếu bậc thang

`CLAUDE.md` §7 tầng 4 đòi **naive → tuyến tính → cây → tuned**. File kế hoạch mới có
Model 0 (baseline) và Model 1 (Logistic). Thiếu cây/GBM và thiếu bản tuned.

### Vấn đề chưa ai nêu: lớp dương chỉ ~4%

`is_low_rating = rating <= 3`, và quần thể có trọng số chỉ **4,0%** ≤3★. Nghĩa là:

* **Accuracy vô nghĩa** — đoán bừa "luôn luôn không phải low rating" đã được 96%.
  Baseline Model 0 phải là đúng cái này, và phải nói thẳng nó được 96%.
* Metric phải là **PR-AUC** và **Brier score**, không phải accuracy, cũng không nên
  chỉ ROC-AUC (ROC đẹp giả trên dữ liệu lệch).
* Cần **calibration curve** — tầng 6 của khung chứng minh, và gần như không nhóm nào làm.

### Thiếu toàn bộ tầng đánh giá

Không thấy nhắc: chia theo thời gian (cạm bẫy #4), paired bootstrap cho **chênh
lệch** metric (tầng 5), calibration (tầng 6), error analysis + ablation (tầng 7).
Đây đúng là 3 điểm của đề bài — đừng để Modeling dừng ở `model.fit()` rồi in metric.

---

## 1.8 Không hề nhắc tới — 5 mục đang trống

| Mục | Trạng thái |
|---|---|
| **Đối chiếu Olist** (bước 9 trong `README.md`) | ⬜ chưa, và không có trong file kế hoạch |
| **Báo cáo + slide** (bước 10) | ⬜ chưa, và không có trong file kế hoạch |
| **Nâng `n_eff` = 85** cho ước lượng phụ còn mong manh | cần cào thêm sản phẩm sau 2023 |
| **Test cho code mới** | `mnar_sensitivity.py` hiện **0 test**; `src/features/` và `src/models/` rỗng |
| **Vệ sinh git** | 5 commit, message là `nghịch`, `add new sth hêhhehe`, `add something hâhhaa` |

Mục cuối nghe như chuyện nhỏ nhưng nó nằm trong 2 điểm code, và sắp tới 3 người
cùng đẩy code lên một repo — không đặt quy ước bây giờ thì tuần sau sẽ mất buổi để
gỡ conflict.

---

## 1.9 Bảng tổng kết đánh giá

| # | Mục trong .docx | Đánh giá | Việc phải làm |
|---|---|---|---|
| 0 | Mọi con số | ❌ Của mẻ 117.819, đã bị thay | Chạy lại, cập nhật (**A0**) |
| 1 | Cắt còn 2022–2026 | ⚠️ Chẩn đoán đúng, phương thuốc sai | Đổi thành phân tích độ nhạy 4 lát (**A7**) |
| 2 | SLA không theo từng đơn | ✅ Đúng, nhưng dừng quá sớm | Quét ngưỡng SLA 3–7 ngày (**A6**) + 1 giờ dò endpoint (**A5**) |
| 3 | Phân tích MNAR | ✅ Trúng, thiếu 1 phân biệt | Chặn chỉ chạy trong 2023+ (**A1–A3**, phương án B) |
| 4 | 8 biểu đồ EDA | 🔶 Khung tốt, thiếu trọng số + 4 hình | Bắt buộc trọng số (**B1**) + thêm 4 hình (**B11–B14**) |
| 5 | Phân rã ngành/seller | ❌ Không chạy được như đang viết | Đổi sang Tiki Trading vs bên thứ ba (**C14** · **C15**→Track A) |
| 6 | Feature Engineering | ⚠️ Cộng tuyến hoàn hảo | Tách thành 3 model để làm ablation (**C9**) |
| 7 | Modeling | 🔶 Mới 2/5 bậc, sai metric | Đủ bậc thang + PR-AUC + calibration (**C4–C13**) |

---

# PHẦN 2 — PHÂN CHIA CÔNG VIỆC CHO 3 NGƯỜI

## 2.0 Quyết định đã chốt — 2026-09-22

Mười một quyết định đã duyệt. Mọi task bên dưới **đã cập nhật theo bảng này**.
Đổi một dòng ở đây thì phải sửa lại task tương ứng.

| # | Quyết định | Đã chốt | Ảnh hưởng |
|---|---|---|---|
| **D1** | Phạm vi năm | ✅ **Giữ toàn mẫu**, báo cáo theo năm, chạy A7 làm bằng chứng | Không cắt 2021 |
| **D2** | Biến mục tiêu | ✅ **`is_low_rating`** (rating ≤ 3) | C1, C5 |
| **D3** | Chia thời gian | ✅ **train ≤2023 · test 2024–2026** | C5 viết lại |
| **D4** | MNAR | ✅ **Phương án B** — bỏ bootstrap, vẫn ra θ\* | **A4 bị cắt** |
| **D5** | Phân rã | ✅ **Tiki Trading vs bên thứ ba**, bỏ xếp hạng 366 seller | C14, C15 |
| **D6** | Nhân sự | ⚠️ **3 track, chưa gán tên** — xem §2.6 | cần bạn điền |
| **D7** | Deadline | ⚠️ **Giả định 3 tuần → 2026-10-13** — xem §2.6 | cần bạn xác nhận |
| **D8** | Cào mẻ v4 | ✅ **Bỏ.** Báo cáo `n_eff=85` dưới dạng khoảng + cờ mong manh | **A11, A12 bị cắt** |
| **D9** | Trọng số | ✅ **Train KHÔNG trọng số · Đánh giá CÓ trọng số** | C2 |
| **D10** | Olist | ✅ **Ưu tiên thấp nhất**, cắt đầu tiên nếu thiếu giờ | A13 tuỳ chọn |
| **D11** | Trình bày | ✅ **Notebook là bản chính**, slide 12 trang chỉ dẫn chuyện | B15 ↑ · B18 ↓ |

### Căn cứ cho D2 (đo trực tiếp, không chép)

| Ứng viên | Có trên | Lớp dương (thô) | Lớp dương (**có trọng số**) |
|---|---|---|---|
| **`is_low_rating`** ✅ | 198.355 dòng | 16.597 (8,37%) | **3,88%** |
| `customer_says_late` ❌ | 15.816 dòng | ~316 (2,00%) | 1,87% |

`customer_says_late` chỉ có **~316 ca dương** — quá mỏng làm biến mục tiêu chính.
Nó vẫn được dùng làm **feature** trong M1c và làm nhãn đối chiếu ở bảng chéo 2×2.

### Căn cứ cho D3 (đo trực tiếp)

| Lát cắt | n | ca dương `is_low_rating` |
|---|---|---|
| ~~train ≤2024 · val 2025 · test 2026~~ ❌ | test chỉ 7.348 | **268** — quá ít để bootstrap |
| **train ≤2023 · test 2024–2026** ✅ | test 39.881 | **1.078** |

Lý do phụ: train ≤2024 chứa 2021 nên tỉ lệ lớp dương train 8,95% so với test 3,65%
— lệch 2,5 lần. Mốc 2023 giảm bớt độ lệch đó.

---

## 2.1 Luật chung — đọc trước khi mở máy

### Quyền sở hữu file (để không ai đè code ai)

| Track | Được sửa | Tuyệt đối không sửa |
|---|---|---|
| **A** | `src/evaluate/*` (trừ `stats.py`), `src/viz/` **không**, `tests/test_mnar.py`, `tests/test_sensitivity.py`, `Makefile` | `src/features/`, `src/models/`, `src/clean/` |
| **B** | `src/viz/` (mới), `notebooks/`, `reports/figures/` | `src/evaluate/`, `src/models/`, `src/clean/` |
| **C** | `src/features/`, `src/models/`, `tests/test_features.py`, `tests/test_models.py` | `src/clean/`, `src/evaluate/stats.py`, `notebooks/` |

**`src/clean/reviews.py` và `src/evaluate/stats.py` là vùng đông lạnh.** Ai cần sửa
phải báo cả nhóm trước — mọi con số trong `FINDINGS.md` phụ thuộc vào hai file này.

### Chia mục `docs/FINDINGS.md`

* **A** sở hữu §7 (giới hạn) · §8 (việc tiếp) · §9 mới (độ nhạy MNAR + ngưỡng SLA + phân rã ngành)
* **B** sở hữu §10 mới (thư viện hình) — mỗi hình một dòng: tên file → kết luận nó chứng minh
* **C** sở hữu §11 mới (modeling)
* **§1–§6 là vùng đông lạnh.** Chỉ sửa khi cả nhóm đồng ý.

### Git

```
feat: thêm quét ngưỡng SLA 3-7 ngày
fix:  sửa trọng số trong histogram lead_days
docs: cập nhật FINDINGS §10
test: thêm ca tái lập 0.2082 cho mnar
```

* Mỗi người một nhánh: `track-a-mnar`, `track-b-eda`, `track-c-model`
* Không commit vào `main` trực tiếp · không commit `data/` (đã trong `.gitignore`)
* Pull `main` **trước** mỗi phiên làm việc

### Định nghĩa "xong" — áp cho mọi task

1. Chạy được bằng **một lệnh**, không phải copy-paste từng cell
2. Số liệu **sinh ra từ lệnh đó**, không chép tay vào tài liệu
3. Mọi ước lượng quần thể **có trọng số**, và **in kèm `n_eff`**
4. Có test nếu là logic mới trong `src/`
5. `python3 -m pytest` vẫn **xanh toàn bộ**
6. Log thô lưu vào `docs/evidence/<tên>_<ngày>.txt`

---

## 2.2 TRACK A — Chứng minh & Độ tin cậy · ~44h

**Mục tiêu:** khoá lại 3 điểm "chứng minh đúng". Khó nhất về tư duy, nhẹ nhất về số dòng code.
**Đã có sẵn:** `src/evaluate/stats.py`, `mnar_sensitivity.py` (xong phần mô tả), kế hoạch chi tiết `CLAUDE.md` §9.1.

| ID | Việc | Đầu ra | Giờ |
|---|---|---|---|
| **A0** | Chạy lại `run_clean` → `run_analysis` → `lead_anomaly`. Cập nhật mọi số trong `.docx` sang mẻ 203.510 | log mới trong `docs/evidence/` | 1 |
| **A0b** | `scripts/check_stale_numbers.py` — grep `docs/` + `README.md` tìm hằng số đã bị bác bỏ (`8.54`, `1.59`, `4.64`, `91.9`, `94.3`, `81.8`, `88.9`, `9.8%`, `11.391`, `10.859`) và **thoát mã lỗi** nếu thấy | script | 2 |
| **A7** | ⭐ **Độ nhạy kỷ nguyên** — chạy lại kết luận chính trên 4 lát: toàn mẫu · bỏ 2021 · 2022+ · 2023+. **Đây là bằng chứng cho D1** | bảng 4 dòng + KTC + `n_eff` | 3 |
| **A6** | **Quét ngưỡng SLA** ở `sla_days ∈ {3,4,5,6,7}` (dùng tham số sẵn có của `compute_gap`) | `src/evaluate/sensitivity.py` + bảng | 4 |
| **A5** | Dò endpoint `product detail` tìm `delivery_estimate`/`handling_time` cấp sản phẩm. **Chốt cứng 1 giờ.** Có → báo nhóm. Không → ghi bằng chứng vào `FEASIBILITY.md`, đóng câu hỏi | mục mới FEASIBILITY | 1 |
| **A1** | Chặn Manski **chỉ trong kỷ nguyên 2023+**. Dự đoán trước: chặn sẽ rộng. Báo cáo kể cả khi vô dụng | FINDINGS §9 | 4 |
| **A2** | Quét điểm gãy θ theo `CLAUDE.md` §9.1 mục 2, dùng **trọng số phân đoạn** | `theta_scan()` | 6 |
| **A3** | 🔒 **Tự kiểm bắt buộc**: `theta_scan` trên riêng nhóm có nhãn với `p` cứng phải ra **đúng 0,2082**. Không khớp → hàm sai, dừng sửa | test đỏ→xanh | 1 |
| ~~A4~~ | ~~Bootstrap KTC tại 4 giá trị θ~~ | **CẮT — D4 chọn phương án B** | ~~3~~ |
| **A8** | Viết lại `FINDINGS.md` §7 theo phát hiện hai cơ chế. Câu "nhóm có nhãn giao nhanh hơn" đang phóng đại 4,0× trong khi cùng kỷ nguyên chỉ 1,3× | §7 viết lại | 2 |
| **A9** | Test cho `mnar_sensitivity.py` — hiện **0 dòng test**. Tối thiểu: ca A3, ca trọng số phân đoạn cộng đúng, ca θ=1 bằng MAR | `tests/test_mnar.py` | 3 |
| **A10** | Test cho `sensitivity.py` — ca `sla_days=5` phải tái lập đúng số hiện hành `FINDINGS.md` §4 | `tests/test_sensitivity.py` | 2 |
| ~~A11~~ | ~~Cào mẻ v4 nâng `n_eff` 85→100~~ | **CẮT — D8.** Báo cáo dưới dạng khoảng + `⚠️ MONG MANH` | ~~5~~ |
| ~~A12~~ | ~~Kiểm chứng lại trọng số sau A11~~ | **CẮT — kéo theo A11** | ~~1~~ |
| **A13** | 🔻 **Đối chiếu Olist** — tham chiếu quốc tế, **không** phải nguồn phân tích. **D10: cắt đầu tiên nếu thiếu giờ** | mục FINDINGS | 4 |
| **C10** | ⬅️ *chuyển từ Track C.* **Paired bootstrap theo cụm sản phẩm** cho **chênh lệch** metric giữa các model. Dùng `src/evaluate/stats.py` — vùng của A. `n_boot=1000` | KTC cho ΔPR-AUC | 5 |
| **C15** | ⬅️ *chuyển từ Track C.* **Phân rã theo ngành, có cổng `n_eff`.** Gộp còn 3 nhóm, hoặc chỉ báo cáo ngành đủ `n_eff`, phần còn lại ghi thẳng *"không đủ dữ liệu"*. Bách hóa online có **0** ca lệch, Điện thoại có **2** | bảng có cờ mong manh | 4 |
| **B16** | ⬅️ *chuyển từ Track B.* `notebooks/00_data_quality.ipynb` — bảng cờ chất lượng, 22/22 contract, kiểm chứng `Σw` vs `Σreview_count`. Nội dung thuộc chuyên môn A | notebook | 3 |
| **S4** | `Makefile`: `make clean / analysis / mnar / sensitivity / figures / model / test / all` | Makefile | 3 |

**Thứ tự:** A0 → A0b → **A7** → A6 → A5 → A1 → A2 → **A3** → A9 → A8 → A10 → C15 → B16 → C10 → S4 → A13

> **A7 lên vị trí thứ ba** vì Track C bị chặn bởi nó. Phải xong **trong tuần 1**.
> **A3 là cổng khoá** — chưa tái lập được 0,2082 thì không được đi tiếp sang A9.

---

## 2.3 TRACK B — EDA hình ảnh & Notebook · ~59h

**Mục tiêu:** biến mọi con số trong `FINDINGS.md` thành hình. **D11 đã chốt notebook là bản chính** — đây là track đổi công thành điểm nhanh nhất.
**Ràng buộc tuyệt đối:** notebook **không chứa logic**. Mọi phép tính gọi từ `src/`.

### B1–B3 · hạ tầng (làm trước, đừng vẽ vội)

| ID | Việc | Giờ |
|---|---|---|
| **B1** | 🔒 `src/viz/weighted.py` — `weighted_hist()`, `weighted_ecdf()`, `weighted_bar_ci()`, `weighted_box()`. **Task này chặn mọi task vẽ** | 5 |
| **B2** | `src/viz/theme.py` — bảng màu **an toàn cho người mù màu**, font, nhãn tiếng Việt, `save_fig()` xuất 300dpi vào `reports/figures/` | 3 |
| **B3** | Test `src/viz/weighted.py`: ca "trọng số đều = hàm thô" và ca "trọng số lệch đổi kết quả đúng chiều" | 2 |

### B4–B9 · 6 hình từ danh sách gốc (đã sửa)

| ID | Hình | Sửa gì so với `.docx` | Giờ |
|---|---|---|---|
| **B4** | Histogram `lead_days` | **Có trọng số** + vẽ đè bản thô để lộ vì sao trọng số quan trọng. Kẻ vạch SLA=5 | 2 |
| **B5** | Boxplot `lead_days` **theo năm** | Boxplot gộp không nói được gì; tách theo năm thì 2021 tự nhảy ra | 2 |
| **B6** | Phân vị p10→p99 | Có trọng số, kẻ vạch SLA=5. Nêu rõ trung vị 1,26 = **25% của trần SLA** | 2 |
| **B7** | SLA đạt/vượt **theo năm** | **Chú thích `n` từng năm ngay trên hình** — 2017 có 541 dòng, 2021 có 52.044 | 3 |
| **B8** | Phân phối rating 1★→5★ | **Có trọng số** + đè bản thô. Hình dễ nói dối nhất: thô 8,37% ≤3★ vs quần thể **3,88%** | 2 |
| **B9** | Rating theo bin `lead_days` | Đổi từ scatter sang: bin (0–1,1–2,2–3,3–5,5–10,>10) × rating TB **có trọng số** × dải KTC | 4 |

### B10–B14 · 5 hình mạnh nhất (không có trong `.docx`)

| ID | Hình | Vì sao nó ăn điểm | Giờ |
|---|---|---|---|
| **B10** | ⭐ **Heatmap bảng chéo 2×2**, 2 ô bẫy tô nổi — **số mới 15.160 / 252 / 339 / 65** | Trục của cả đồ án. Số trong `.docx` là của mẻ đã chết | 3 |
| **B11** | ⭐ **Forest plot: kết luận chính qua 3 mẻ** — 4 điểm ước lượng + KTC (`FINDINGS.md` §4) | Hình trả lời "sao tin được?". Không nhóm nào trong lớp có | 4 |
| **B12** | ⭐ **Trước/sau sửa múi giờ** — phân phối dịch 0,2917 ngày + phép đo 7,0000h ± 0,000000 | "Một API, hai đồng hồ" là phần hay nhất, hiện chưa có hình | 4 |
| **B13** | **`n` so với `n_eff`** qua 3 mẻ — cột 182 cạnh cột 25 | "Số dòng ≠ lượng thông tin" gói trong một hình | 3 |
| **B14** | **Độ phủ nhãn theo năm** — vách 0% dựng đứng trước 2023 | Toàn bộ câu chuyện MNAR trong một hình, và giải thích luôn **vì sao D1 giữ 2021** | 2 |

### B15–B19 · notebook và slide

| ID | Việc | Giờ |
|---|---|---|
| **B15** | ⭐ `notebooks/01_eda.ipynb` — **bản chính theo D11.** Kể chuyện qua B4→B14, mỗi hình kèm **một câu kết luận**, không phải một đoạn mô tả. Chỉ gọi hàm từ `src/` | **8** |
| ~~B16~~ | ~~`notebooks/00_data_quality.ipynb`~~ | ➡️ **chuyển sang Track A** |
| **B17** | `FINDINGS.md` §10 — thư viện hình: mỗi hình một dòng *"tên file → kết luận nó chứng minh"* | 2 |
| **B19** | Hình phân rã cho Track A/C (Tiki Trading vs bên thứ ba) — **chờ C14 và C15** | 3 |
| **B18** | 🔻 Slide 12 trang. **D11: chỉ để dẫn chuyện**, nội dung thật nằm ở notebook — đừng làm hai bản rồi lệch số | **5** |

**Thứ tự:** B1 → B2 → B3 → **B10** → **B14** → B4 → B8 → B7 → B6 → B5 → B9 → B11 → B12 → B13 → **B15** → B17 → B19 → B18

> **B10 và B14 lên sớm** vì Track A và C đều cần chúng để nói chuyện với nhau.
> **B18 để cuối** — làm slide trước khi có kết quả là làm lại hai lần.

---

## 2.4 TRACK C — Feature & Modeling · ~59h

**Mục tiêu:** biến kết luận §4 từ phép so sánh trung bình thành phép so sánh **sức dự đoán**.
**Bị chặn bởi A7** (tuần 1). Trong lúc chờ: làm C3 → C2 → C1 → C4.
**Nền tảng:** `src/features/` và `src/models/` hiện **rỗng hoàn toàn** — xây từ đầu.

### C1–C3 · nền

| ID | Việc | Giờ |
|---|---|---|
| **C3** | 🔒 **Danh sách biến cấm (leakage).** `gap_days = lead_days − 5` → tương quan **đúng 1,0**; `sla_breach` là bản nhị phân của cùng biến. Ghi rõ biến nào không được đứng chung mô hình nào | 2 |
| **C2** | Ghi lại **D9 đã chốt** vào `CLAUDE.md`: **train KHÔNG trọng số** (học tốt hơn lớp hiếm) · **đánh giá CÓ trọng số** (ước lượng quần thể) + lý do | 2 |
| **C1** | `src/features/build.py`. Target = **`is_low_rating`** (D2). Gồm cả biến đang bị bỏ quên: `review_lag_days`, `dr_shipper`, `dr_dong_goi`, `n_images`, `thank_count`, `customer_purchased_flag`, **`la_tiki_trading`** | 6 |

### C4–C8 · bậc thang mô hình

| ID | Việc | Giờ |
|---|---|---|
| **C4** | `src/models/base.py` — `BaseModel` trừu tượng, interface `fit`/`predict_proba` chung. **Yêu cầu OOP `CLAUDE.md` §5, tính vào 2 điểm code** | 4 |
| **C5** | **Chia theo D3: train ≤2023 · test 2024–2026** (test 39.881 dòng, 1.078 ca dương). Không bao giờ KFold ngẫu nhiên (cạm bẫy #4). Encoder/scaler `fit` **bên trong** fold | 4 |
| **C6** | **M0 — baseline đa số.** Luôn đoán "không phải low rating". **Sẽ được ~96%** — đó chính là lý do accuracy vô dụng, và phải nói thẳng ra | 2 |
| **C7** | **M1a / M1b / M1c** — ba Logistic riêng: chỉ `sla_breach` · chỉ `lead_days` · chỉ nhãn khách. **Không gộp** (xem C3) | 5 |
| **C8** | 🔻 **M2 cây/GBM** đầy đủ feature + **M3 tuned**. Hoàn tất bậc thang naive→tuyến tính→cây→tuned | 6 |

### C9–C14 · tầng chứng minh — đây là 3 điểm

| ID | Việc | Giờ |
|---|---|---|
| **C9** | ⭐⭐ **Ablation trung tâm** — so sức dự đoán M1a vs M1b vs M1c. Nếu "cách doanh nghiệp đang đo" (M1a) gần như không thắng baseline trong khi hai cái kia thắng rõ → **kết luận §4 được chứng minh lần hai bằng phương pháp độc lập** | 6 |
| ~~C10~~ | ~~Paired bootstrap cho chênh lệch metric~~ | ➡️ **chuyển sang Track A** (dùng `stats.py`, vùng của A) |
| **C11** | **Metric đúng cho dữ liệu lệch 3,88%**: PR-AUC + Brier. Không accuracy. ROC-AUC chỉ ghi kèm | 3 |
| **C12** | 🔻 **Calibration curve** + độ phủ thực tế. Tầng 6 khung chứng minh — gần như không nhóm nào làm | 4 |
| **C13** | **Phân tích lỗi theo phân khúc** — sai nhiều nhất ở đâu? Theo năm (**2021 vs 2024–2026**), theo ngành, theo Tiki vs bên thứ ba. **2021 là chế độ duy nhất có vượt SLA hai chữ số** → trả lời được *"khi giao thật sự trễ thì lead_days có ăn vào rating không?"* | 5 |
| **C14** | ⭐⭐ **Tiki Trading vs bên thứ ba** (D5) — thay cho xếp hạng 366 seller vốn không chạy được. Cả hai nhóm đủ dữ liệu: Tiki Trading n=11.573 (199/203 ở hai ô lệch), bên thứ ba n=4.243 (53/136). Bảng chéo **có trọng số** + bootstrap cho **chênh lệch** | 5 |
| ~~C15~~ | ~~Phân rã theo ngành có cổng `n_eff`~~ | ➡️ **chuyển sang Track A** |
| **C16** | Test `src/features/` và `src/models/`: ca không-leakage, ca chia thời gian đúng, ca `BaseModel` giữ đúng interface | 5 |

**Thứ tự:** C3 → C2 → C1 → C4 → C5 → C6 → C7 → **C9** → C11 → **C14** → C13 → C16 → C8 → C12

> **C9 và C14 là hai task đáng giá nhất.** Nếu hết giờ, bỏ **C8 và C12 trước** — đừng bao giờ bỏ C9/C14.

---

## 2.5 Lịch 3 tuần

Giả định deadline **2026-10-13** (xem D7 ở §2.6).

| Tuần | Track A | Track B | Track C |
|---|---|---|---|
| **1**<br>22–29/9 | A0 · A0b · **A7** · A6 · A5 | B1 · B2 · B3 · **B10** · B14 | C3 · C2 · C1 · C4 |
| **2**<br>30/9–6/10 | A1 · A2 · **A3** · A9 · A8 | B4–B9 · B11 · B12 · B13 | C5 · C6 · C7 · **C9** · C11 |
| **3**<br>7–13/10 | A10 · C15 · B16 · C10 · S4 · A13 | **B15** · B17 · B19 · B18 | **C14** · C13 · C16 · C8 · C12 |

### Bốn điểm chặn — biết trước để không ngồi chờ

1. **A7 chặn C5** — có dùng 2021 hay không quyết định cách chia train/test. A7 phải xong **trong tuần 1**.
2. **B1 chặn B4–B14** — không có hàm vẽ có trọng số thì mọi hình vẽ ra đều phải vẽ lại.
3. **C14 + C15 chặn B19** — B vẽ hình phân rã sau khi có số.
4. **Tất cả chặn B18** — slide làm sau cùng.

---

## 2.6 Hai ô còn trống — cần bạn điền

### D6 · Ai làm track nào?

`README.md` ghi **4 thành viên** (An · Hiển · Ngân · Sỹ) nhưng đang chia 3 track.

| Track | Giờ | Tính chất | Người |
|---|---|---|---|
| **A** — Chứng minh & Độ tin cậy | ~44h | **Khó nhất về tư duy** — thống kê, chặn, `n_eff`. Ít code nhất | _______ |
| **B** — EDA hình & Notebook | ~59h | Nhiều việc tay nhất, rủi ro thấp nhất, **ra điểm nhanh nhất** | _______ |
| **C** — Feature & Modeling | ~59h | Nhiều code nhất, **bị chặn bởi A7** tuần 1 | _______ |

**Nếu đủ 4 người:** người thứ tư nhận **Track D — reviewer chéo + báo cáo**
(S1 viết báo cáo 10h · S2 kiểm chéo 4h · S3 cập nhật README 2h · S5 `CONTRIBUTING.md` 2h ·
S6 tổng duyệt 4h). Đề cương gốc ghi vai này *"quan trọng nhất"*.

**Nếu chỉ 3 người:** chia S1/S2/S3/S5/S6 đều ba người — mỗi người viết phần track
mình, một người ráp; kiểm chéo xoay vòng **A kiểm C · B kiểm A · C kiểm B**, kiểm
đúng 3 thứ: (1) có trọng số không, (2) có `n_eff` không, (3) số có sinh từ lệnh không.

### D7 · Deadline thật?

Đang giả định **3 tuần → 2026-10-13**. Nếu ngắn hơn, **thứ tự cắt đã định sẵn**:

1. **A13** Olist (D10 đã chốt là ưu tiên thấp nhất) — tiết kiệm 4h
2. **C8** cây/GBM + tuned — 6h
3. **C12** calibration — 4h
4. **B5, B6** hai hình phân vị/boxplot — 4h
5. **B18** slide, trình bày thẳng từ notebook theo D11 — 5h

**Không bao giờ cắt:** A3 · A7 · B1 · B10 · C9 · C14.

---

## 2.7 Ba điều dễ làm hỏng nhất — dán lên tường

1. **Không nhân trọng số = sai hệ thống.** Đã hai lần dẫn tới kết luận ngược
   (`FINDINGS.md` §5, §6.3). Mọi tỉ lệ, mọi hình, mọi metric đánh giá.
2. **`n` lớn không cứu được `n_eff` nhỏ.** 182 dòng có thể chỉ mang thông tin của 25.
   Thấy `⚠️ MONG MANH` thì báo cáo dưới dạng khoảng — **đừng gỡ cảnh báo** (D8 đã
   chốt là không cào thêm).
3. **Không chép số bằng tay.** Mọi con số trong báo cáo/slide phải sinh từ một lệnh.
   Tài liệu này tồn tại chính vì `.docx` đã chép tay số của mẻ 117.819.

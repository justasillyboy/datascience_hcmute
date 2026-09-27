"""Feature lịch sử "tính tới thời điểm dự báo" (as-of / point-in-time).

Câu hỏi mà feature này trả lời: *tại lúc đơn được giao, những review đã đăng
trước đó của cùng sản phẩm / nhà bán / khách hàng nói gì?* Đây là thông tin Tiki
thật sự có trong tay tại thời điểm dự báo, nên không phải leakage — **với điều
kiện** chỉ cộng những review có `review_ts` **nhỏ hơn hẳn** mốc dự báo, và không
bao giờ cộng chính dòng đang dự báo.

Hai lựa chọn phương pháp cần giải thích được khi bảo vệ:

1. **Cộng có trọng số khảo sát** `w = N_h/n_h`. Mẫu vét cạn review 1–3★ nhưng chỉ
   lấy mẫu 5★, nên đếm thô sẽ phóng đại tỉ lệ sao thấp của mọi sản phẩm. Cộng có
   trọng số là ước lượng Horvitz–Thompson của tổng quần thể.
2. **Làm trơn Bayes thực nghiệm** (m-estimate, tương đương prior Beta):
   `rate = (Σw·y + m·p0) / (Σw + m)`. Sản phẩm mới chỉ có 1 review xấu sẽ không bị
   gán tỉ lệ 100%; nó bị kéo về tỉ lệ chung `p0`, và càng nhiều review thì càng tự
   đứng được. `m` là "số review ảo" của prior.

Cài đặt **vector hoá hoàn toàn**: một lần sắp xếp + một lần `np.searchsorted`
cho mọi nhóm cùng lúc, thay vì lặp Python qua 146.241 khách hàng.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_NS_PER_SECOND = 1_000_000_000


def _to_seconds(ts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Datetime → số giây nguyên (int64) + mặt nạ hợp lệ. Số nguyên để so sánh chính xác."""
    s = pd.to_datetime(pd.Series(ts).reset_index(drop=True))
    valid = s.notna().to_numpy()
    sec = np.zeros(len(s), dtype=np.int64)
    sec[valid] = s[valid].to_numpy("datetime64[ns]").astype(np.int64) // _NS_PER_SECOND
    return sec, valid


def asof_group_stats(
    groups: pd.Series,
    event_ts: pd.Series,
    cutoff_ts: pd.Series,
    y: pd.Series,
    w: pd.Series,
    prior_rate: float,
    prior_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Tổng trọng số và tỉ lệ nhãn dương đã làm trơn của các sự kiện **trước** mốc dự báo.

    Mỗi dòng vừa là một *sự kiện* (review đăng lúc `event_ts`, nhãn `y`, trọng số
    `w`) vừa là một *truy vấn* (dự báo lúc `cutoff_ts`). Kết quả dòng i chỉ gồm
    các sự kiện j cùng nhóm với `event_ts[j] < cutoff_ts[i]`, và luôn loại j = i.

    Thuật toán: mã hoá khoá `nhóm·K + giây` (số nguyên, không mất chính xác) để
    mọi nhóm nằm trên một trục duy nhất đã sắp xếp; tổng tích luỹ trên trục đó
    cho phép lấy tổng của bất kỳ đoạn "đầu nhóm → trước mốc" bằng hai phép trừ.

    Returns:
        `(sum_w, rate)` — `rate` là NaN khi không có lịch sử và `prior_strength=0`,
        hoặc khi mốc dự báo bị thiếu.
    """
    g = pd.factorize(pd.Series(groups).reset_index(drop=True))[0].astype(np.int64)
    ev, ev_ok = _to_seconds(event_ts)
    cut, cut_ok = _to_seconds(cutoff_ts)
    yv = np.asarray(y, dtype=float)
    wv = np.asarray(w, dtype=float)

    known = np.concatenate([ev[ev_ok], cut[cut_ok]])
    base = known.min() if known.size else 0
    span = (known.max() - base + 2) if known.size else 2
    # Sự kiện thiếu mốc thời gian được đẩy về cuối nhóm (span-1) nên không bao giờ
    # đứng trước một mốc dự báo hợp lệ (tối đa span-2).
    ev_rel = np.where(ev_ok, ev - base, span - 1)
    ev_key = g * span + ev_rel

    order = np.argsort(ev_key, kind="stable")
    sorted_keys = ev_key[order]
    cum_w = np.concatenate([[0.0], np.cumsum(wv[order])])
    cum_wy = np.concatenate([[0.0], np.cumsum((wv * yv)[order])])

    query = g * span + np.where(cut_ok, cut - base, 0)
    end = np.searchsorted(sorted_keys, query, side="left")        # sự kiện có khoá < truy vấn
    start = np.searchsorted(sorted_keys, g * span, side="left")   # vị trí đầu nhóm
    sum_w = cum_w[end] - cum_w[start]
    sum_wy = cum_wy[end] - cum_wy[start]

    # Dữ liệu lỗi (review đăng trước lúc giao) sẽ khiến dòng tự đếm nhãn của mình.
    self_counted = ev_ok & cut_ok & (ev < cut)
    sum_w = np.clip(sum_w - np.where(self_counted, wv, 0.0), 0.0, None)
    sum_wy = np.clip(sum_wy - np.where(self_counted, wv * yv, 0.0), 0.0, None)

    denom = sum_w + prior_strength
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(denom > 0, (sum_wy + prior_strength * prior_rate) / denom, np.nan)
    rate = np.where(cut_ok, rate, np.nan)
    sum_w = np.where(cut_ok, sum_w, np.nan)
    return sum_w, rate


def asof_group_stats_loop(
    groups: pd.Series,
    event_ts: pd.Series,
    cutoff_ts: pd.Series,
    y: pd.Series,
    w: pd.Series,
    prior_rate: float,
    prior_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cài đặt tham chiếu bằng vòng lặp Python thuần — O(n²) trong mỗi nhóm.

    Chỉ dùng để (1) kiểm chứng bản vector hoá cho kết quả **giống hệt** (test),
    và (2) đo `%timeit` trong notebook. Không dùng trong pipeline.
    """
    g = pd.Series(groups).to_numpy()
    ev = pd.to_datetime(pd.Series(event_ts)).to_numpy()
    cut = pd.to_datetime(pd.Series(cutoff_ts)).to_numpy()
    yv, wv = np.asarray(y, dtype=float), np.asarray(w, dtype=float)
    n = len(g)
    sum_w, rate = np.full(n, np.nan), np.full(n, np.nan)
    members: dict = {}
    for j in range(n):
        members.setdefault(g[j], []).append(j)
    for i in range(n):
        if np.isnat(cut[i]):
            continue
        s_w = s_wy = 0.0
        for j in members[g[i]]:
            if j != i and not np.isnat(ev[j]) and ev[j] < cut[i]:
                s_w += wv[j]
                s_wy += wv[j] * yv[j]
        sum_w[i] = s_w
        denom = s_w + prior_strength
        rate[i] = (s_wy + prior_strength * prior_rate) / denom if denom > 0 else np.nan
    return sum_w, rate

from pathlib import Path
import json
import requests


PRODUCT_IDS = [
    "278630241",
    "278628866",
    "278628812",
    "277619147",
    "201067649",
]

OUTPUT_PATH = Path(
    "docs/evidence/a5_product_delivery_endpoint.txt"
)

BASE_URL = (
    "https://tiki.vn/api/v2/products/"
    "widget/delivery_info/"
)


def find_fields(obj, target_fields, path=""):
    """
    Tìm đệ quy các field cần kiểm tra trong JSON response.
    """
    found = []

    if isinstance(obj, dict):
        for key, value in obj.items():

            current_path = (
                f"{path}.{key}"
                if path
                else key
            )

            if key.lower() in target_fields:
                found.append(
                    (
                        current_path,
                        value
                    )
                )

            found.extend(
                find_fields(
                    value,
                    target_fields,
                    current_path,
                )
            )

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            current_path = f"{path}[{i}]"

            found.extend(
                find_fields(
                    item,
                    target_fields,
                    current_path,
                )
            )

    return found


def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_fields = {
        "delivery_estimate",
        "delivery_estimate_time",
        "handling_time",
        "time_estimation",
        "delivery_options",
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as log:

        def write(text=""):
            print(text)
            log.write(text + "\n")

        write("=" * 70)
        write("A5 - TIKI PRODUCT DELIVERY ENDPOINT CHECK")
        write("=" * 70)
        write()

        write(
            "Endpoint:"
        )
        write(
            BASE_URL + "{product_id}"
        )
        write()

        write(
            "Product IDs được lấy trực tiếp từ "
            "reviews_clean.parquet:"
        )

        for product_id in PRODUCT_IDS:
            write(f"  - {product_id}")

        write()

        for product_id in PRODUCT_IDS:

            url = BASE_URL + product_id

            write("=" * 70)
            write(
                f"PRODUCT ID: {product_id}"
            )
            write(
                f"URL: {url}"
            )
            write("=" * 70)

            try:

                response = requests.get(
                    url,
                    headers={
                        "User-Agent":
                            "Mozilla/5.0"
                    },
                    timeout=20,
                )

                write(
                    f"HTTP status: "
                    f"{response.status_code}"
                )

                write(
                    f"Content-Type: "
                    f"{response.headers.get('Content-Type')}"
                )

                try:
                    data = response.json()

                except ValueError:

                    write(
                        "Response không phải JSON."
                    )

                    write(
                        response.text[:5000]
                    )

                    write()
                    continue

                # ------------------------------------------------
                # RAW JSON
                # ------------------------------------------------

                write()
                write("RAW JSON RESPONSE:")

                pretty_json = json.dumps(
                    data,
                    ensure_ascii=False,
                    indent=2,
                )

                write(pretty_json)

                # ------------------------------------------------
                # FIELD CHECK
                # ------------------------------------------------

                write()
                write("-" * 70)
                write("FIELD CHECK")
                write("-" * 70)

                found = find_fields(
                    data,
                    target_fields,
                )

                if not found:

                    write(
                        "Không tìm thấy các field mục tiêu:"
                    )

                    write(
                        ", ".join(
                            sorted(target_fields)
                        )
                    )

                else:

                    for path, value in found:

                        write(
                            f"{path} = "
                            f"{json.dumps(value, ensure_ascii=False)}"
                        )

                write()

            except requests.RequestException as exc:

                write(
                    f"REQUEST ERROR: {exc}"
                )

                write()

        write("=" * 70)
        write("KẾT THÚC A5")
        write("=" * 70)
        write(
            f"Evidence: {OUTPUT_PATH}"
        )


if __name__ == "__main__":
    main()
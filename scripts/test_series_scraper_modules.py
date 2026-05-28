from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scraper.dcd_scraper import DongchediScraper  # noqa: E402


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class FakeModuleScraper(DongchediScraper):
    def __init__(self, raw_dir: Path):
        super().__init__(min_interval_sec=0.0, max_retry=1)
        self.raw_dir = str(raw_dir)
        self.raw_image_dir = str(raw_dir / "images")
        Path(self.raw_dir).mkdir(parents=True, exist_ok=True)
        Path(self.raw_image_dir).mkdir(parents=True, exist_ok=True)

    def _fetch_page_next_data(self, url: str, html_path: str, json_path: str, timeout: int = 10):
        if url.endswith("/images-wg"):
            data = build_module_payload("wg", "外观", "正面前脸图", "https://example.com/wg-front.jpg")
        elif url.endswith("/images-ns"):
            data = build_module_payload("ns", "内饰", "正面主副驾拍摄图", "https://example.com/ns-cockpit.jpg")
        else:
            data = build_home_payload()
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        Path(html_path).write_text("<html></html>", encoding="utf-8")
        Path(json_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data, {"url": url, "retry_count": 0, "http_status": 200, "error": ""}

    def _request_binary_with_retry(self, url: str, timeout: int = 20):
        class Resp:
            status_code = 200
            headers = {"content-type": "image/jpeg"}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size: int = 8192):
                yield b"binary-image"

        return True, Resp(), 0, 200, ""


def build_home_payload() -> dict:
    return {
        "props": {
            "pageProps": {
                "seriesId": "25634",
                "seriesName": "长安启源Q05",
                "seriesHomeHead": {
                    "series_id": 25634,
                    "series_name": "长安启源Q05",
                    "brand_name": "长安启源",
                    "cover_url": "https://example.com/cover.jpg",
                    "official_price": "7.99-10.99万",
                    "dealer_price": "6.99-9.99万",
                    "series_image_info_list": [{"image_url": "https://example.com/legacy.jpg"}],
                    "pics_summary_info": [{"CoverPicUrl": "https://example.com/legacy-summary.jpg"}],
                },
                "scoreSimpleInfo": {"score": 4.2, "total_review_count": 123},
                "carModelsData": {"tab_list": []},
                "overviewData": {"space": []},
                "newestStaticNews": [{"title": "不应保留的新闻"}],
                "guideStaticNews": [{"title": "不应保留的导购"}],
            }
        }
    }


def build_module_payload(module_key: str, category_name: str, required_view: str, image_url: str) -> dict:
    return {
        "props": {
            "pageProps": {
                "pictureInfo": {
                    "name": module_key,
                    "series_name": "长安启源Q05",
                    "series_id": 25634,
                    "picture_list": [
                        {
                            "car_id": 254067,
                            "car_name": "405 Air",
                            "sale_status": 0,
                            "pic_url": [image_url, f"{image_url}?alt=1"],
                        }
                    ],
                },
                "head": {
                    "category_list": [
                        {
                            "key": module_key,
                            "text": category_name,
                            "filter": {
                                "color": [
                                    {
                                        "color": "#ffffff",
                                        "color_name": "云锦白",
                                        "key": "EEE5E5_",
                                        "car_ids": [254067],
                                    }
                                ],
                                "car": [
                                    {
                                        "is_all_car": False,
                                        "car_list": [
                                            {
                                                "car_id": 254067,
                                                "name": "405 Air",
                                                "car_text": "2026款 405 Air",
                                                "color_keys": ["EEE5E5_"],
                                            }
                                        ],
                                    }
                                ],
                            },
                        }
                    ]
                },
                "query": {"seriesId": "25634", "category": module_key},
            }
        }
    }


def main() -> int:
    artifact_dir = REPO_ROOT / "data" / "test_runs" / "series_scraper_modules"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    try:
        scraper = FakeModuleScraper(raw_dir=artifact_dir / "raw")
        data, meta = scraper.fetch_series_data("25634")
        page_props = ((data.get("props") or {}).get("pageProps") or {})
        selected = page_props.get("selectedImageModules") or {}

        assert_true("newestStaticNews" not in page_props, "主页新闻字段未被移除")
        assert_true("guideStaticNews" not in page_props, "主页导购字段未被移除")
        assert_true(set(selected.keys()) == {"wg", "ns"}, "selectedImageModules 未正确写入 wg/ns")
        assert_true(page_props["seriesHomeHead"]["series_image_info_list"] == [], "旧版 series_image_info_list 未被清空")
        assert_true(meta["image_success_count"] == 3, "图片保存数量不正确")
        assert_true(selected["wg"]["selected_images"][0]["image_url"] == "https://example.com/wg-front.jpg", "外观图首图选择错误")
        assert_true(selected["ns"]["selected_images"][0]["image_url"] == "https://example.com/ns-cockpit.jpg", "内饰图首图选择错误")

        combined_json = artifact_dir / "raw" / "series_25634.json"
        assert_true(combined_json.exists(), "聚合后的 series json 未保存")

        print("[PASS] 三模块车系抓取测试通过")
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"[FAIL] 三模块车系抓取测试失败: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

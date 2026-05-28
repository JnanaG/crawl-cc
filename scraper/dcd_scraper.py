import requests
import json
import re
import os
import random
import hashlib
import time
from copy import deepcopy
from urllib.parse import quote, urlparse
from loguru import logger
from bs4 import BeautifulSoup


class DongchediScraper:
    NEWS_KEYS = (
        "newestStaticNews",
        "guideStaticNews",
        "newcarStaticNews",
        "evaluatingStaticNews",
        "originalStaticNews",
    )

    IMAGE_MODULE_SPECS = {
        "wg": {
            "path": "images-wg",
            "category_name": "外观",
            "required_view": "正面前脸图",
            "source_section": "module_wg",
        },
        "ns": {
            "path": "images-ns",
            "category_name": "内饰",
            "required_view": "正面主副驾拍摄图",
            "source_section": "module_ns",
        },
    }

    def __init__(self, min_interval_sec: float = 0.8, max_retry: int = 3):
        self.base_url = "https://www.dongchedi.com"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9"
        })
        self.min_interval_sec = min_interval_sec
        self.max_retry = max_retry
        self.last_request_ts = 0.0
        # 确保原始数据存储目录存在
        self.raw_dir = os.path.join("data", "raw", "dongchedi")
        self.raw_image_dir = os.path.join(self.raw_dir, "images")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.raw_image_dir, exist_ok=True)

    def _throttle(self):
        now = time.monotonic()
        elapsed = now - self.last_request_ts
        if elapsed < self.min_interval_sec:
            sleep_sec = self.min_interval_sec - elapsed + random.uniform(0.05, 0.2)
            time.sleep(sleep_sec)
        self.last_request_ts = time.monotonic()

    def _request_with_retry(self, url: str, timeout: int = 10):
        last_error = ""
        last_status = None
        attempts = 0
        for attempt in range(1, self.max_retry + 1):
            attempts = attempt
            try:
                self._throttle()
                resp = self.session.get(url, timeout=timeout)
                last_status = resp.status_code
                # 4xx 通常是路径或权限问题，除 429 外不做重试，避免浪费请求预算。
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    return False, None, attempts - 1, last_status, f"HTTP {resp.status_code}"
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return True, resp, attempts - 1, last_status, ""
            except Exception as e:
                last_error = str(e)
                logger.warning(f"请求失败({attempt}/{self.max_retry}) url={url}, error={e}")
                if attempt < self.max_retry:
                    backoff = min(2 ** (attempt - 1), 8) + random.uniform(0.1, 0.4)
                    time.sleep(backoff)
        return False, None, attempts - 1, last_status, last_error

    def _request_binary_with_retry(self, url: str, timeout: int = 20):
        last_error = ""
        last_status = None
        attempts = 0
        for attempt in range(1, self.max_retry + 1):
            attempts = attempt
            try:
                self._throttle()
                resp = self.session.get(url, timeout=timeout, stream=True)
                last_status = resp.status_code
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    return False, None, attempts - 1, last_status, f"HTTP {resp.status_code}"
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return True, resp, attempts - 1, last_status, ""
            except Exception as e:
                last_error = str(e)
                logger.warning(f"图片请求失败({attempt}/{self.max_retry}) url={url}, error={e}")
                if attempt < self.max_retry:
                    backoff = min(2 ** (attempt - 1), 8) + random.uniform(0.1, 0.4)
                    time.sleep(backoff)
        return False, None, attempts - 1, last_status, last_error

    @staticmethod
    def _as_dict(value) -> dict:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _as_list(value) -> list:
        return value if isinstance(value, list) else []

    @staticmethod
    def _normalize_url(url: str) -> str:
        if not isinstance(url, str):
            return ""
        url = url.strip()
        if not url:
            return ""
        if url.startswith("//"):
            return f"https:{url}"
        return url

    @staticmethod
    def _infer_extension(url: str, content_type: str = "") -> str:
        content_type = (content_type or "").split(";")[0].strip().lower()
        if content_type == "image/jpeg":
            return ".jpg"
        if content_type == "image/png":
            return ".png"
        if content_type == "image/webp":
            return ".webp"
        if content_type == "image/gif":
            return ".gif"
        path = (urlparse(url).path or "").lower()
        for suffix in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
            if path.endswith(suffix):
                return suffix
        if ".image" in path:
            return ".jpg"
        return ".bin"

    @staticmethod
    def _make_image_asset_id(series_id: str, image_url: str, source_section: str, image_role: str, rank: int) -> str:
        raw = "|".join([str(series_id or ""), image_url, source_section, image_role, str(rank)])
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _append_image_task(
        self,
        tasks: list[dict],
        seen: set[tuple[str, str, str]],
        *,
        series_id: str,
        image_url,
        source_section: str,
        image_role: str,
        rank: int,
        category: str = "",
        category_name: str = "",
        raw_ref: str = "",
    ) -> None:
        normalized_url = self._normalize_url(image_url)
        if not normalized_url:
            return
        dedupe_key = (normalized_url, source_section, str(rank))
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        tasks.append(
            {
                "asset_id": self._make_image_asset_id(series_id, normalized_url, source_section, image_role, rank),
                "series_id": str(series_id or ""),
                "image_url": normalized_url,
                "source_section": source_section,
                "image_role": image_role,
                "rank": rank,
                "category": category or "",
                "category_name": category_name or "",
                "raw_ref": raw_ref,
            }
        )

    def _extract_image_download_tasks(self, raw_json: dict, include_contextual_images: bool = True) -> list[dict]:
        page_props = self._as_dict(self._as_dict(raw_json.get("props")).get("pageProps"))
        series_head = self._as_dict(page_props.get("seriesHomeHead"))
        series_id = str(page_props.get("seriesId") or series_head.get("series_id") or "")

        tasks = []
        seen: set[tuple[str, str, str]] = set()

        self._append_image_task(
            tasks,
            seen,
            series_id=series_id,
            image_url=series_head.get("cover_url"),
            source_section="series_cover",
            image_role="cover",
            rank=0,
            category="cover",
            category_name="封面",
            raw_ref="pageProps.seriesHomeHead.cover_url",
        )

        selected_modules = self._as_dict(page_props.get("selectedImageModules"))
        if selected_modules:
            for module_key, module in selected_modules.items():
                module = self._as_dict(module)
                source_section = self.IMAGE_MODULE_SPECS.get(module_key, {}).get("source_section", f"module_{module_key}")
                for idx, sample in enumerate(self._as_list(module.get("selected_images"))):
                    sample = self._as_dict(sample)
                    self._append_image_task(
                        tasks,
                        seen,
                        series_id=series_id,
                        image_url=sample.get("image_url"),
                        source_section=source_section,
                        image_role="module_sample",
                        rank=idx,
                        category=module_key,
                        category_name=module.get("category_name") or "",
                        raw_ref=f"pageProps.selectedImageModules.{module_key}.selected_images",
                    )
            return tasks

        image_floor_data = self._as_dict(page_props.get("imageFloorData"))
        for idx, item in enumerate(self._as_list(image_floor_data.get("floor_image_list"))):
            item = self._as_dict(item)
            category = str(item.get("category") or "")
            category_name = str(item.get("text") or "")
            for image_idx, image_info in enumerate(self._as_list(item.get("image_list"))):
                image_info = self._as_dict(image_info)
                self._append_image_task(
                    tasks,
                    seen,
                    series_id=series_id,
                    image_url=image_info.get("image_url") or image_info.get("url") or image_info.get("cover_url"),
                    source_section="image_floor",
                    image_role="gallery",
                    rank=(idx * 1000) + image_idx,
                    category=category,
                    category_name=category_name,
                    raw_ref="pageProps.imageFloorData.floor_image_list",
                )

        return tasks

    def _extract_next_data_from_html(self, html_text: str) -> dict:
        soup = BeautifulSoup(html_text, "html.parser")
        script_tag = soup.find("script", id="__NEXT_DATA__")
        if not script_tag:
            raise ValueError("__NEXT_DATA__ missing")
        return json.loads(script_tag.string)

    def _fetch_page_next_data(self, url: str, html_path: str, json_path: str, timeout: int = 10) -> tuple[dict, dict]:
        meta = {"url": url, "retry_count": 0, "http_status": None, "error": ""}
        ok, resp, retry_count, http_status, err = self._request_with_retry(url, timeout=timeout)
        meta["retry_count"] = retry_count
        meta["http_status"] = http_status
        if not ok or resp is None:
            meta["error"] = err or "请求失败"
            return {}, meta
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        try:
            data = self._extract_next_data_from_html(resp.text)
        except Exception as e:
            meta["error"] = str(e)
            return {}, meta
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data, meta

    def _strip_news_sections(self, data: dict) -> dict:
        sanitized = deepcopy(data)
        page_props = self._as_dict(self._as_dict(sanitized.get("props")).get("pageProps"))
        for key in self.NEWS_KEYS:
            page_props.pop(key, None)
        return sanitized

    def _build_selected_image_module(self, module_key: str, module_data: dict) -> dict:
        spec = self.IMAGE_MODULE_SPECS[module_key]
        page_props = self._as_dict(self._as_dict(module_data.get("props")).get("pageProps"))
        picture_info = self._as_dict(page_props.get("pictureInfo"))
        head = self._as_dict(page_props.get("head"))

        color_lookup = {}
        for category in self._as_list(head.get("category_list")):
            category = self._as_dict(category)
            if str(category.get("key") or "") != module_key:
                continue
            color_items = self._as_list(self._as_dict(category.get("filter")).get("color"))
            for color in color_items:
                color = self._as_dict(color)
                color_lookup[str(color.get("key") or "")] = {
                    "color_name": color.get("color_name") or "",
                    "sub_color_name": color.get("sub_color_name") or "",
                    "color": color.get("color") or "",
                    "car_ids": self._as_list(color.get("car_ids")),
                }

        car_lookup = {}
        for category in self._as_list(head.get("category_list")):
            category = self._as_dict(category)
            if str(category.get("key") or "") != module_key:
                continue
            car_groups = self._as_list(self._as_dict(category.get("filter")).get("car"))
            for car_group in car_groups:
                car_group = self._as_dict(car_group)
                for car in self._as_list(car_group.get("car_list")):
                    car = self._as_dict(car)
                    car_lookup[str(car.get("car_id") or "")] = car

        selected_images = []
        for idx, item in enumerate(self._as_list(picture_info.get("picture_list"))):
            item = self._as_dict(item)
            pic_urls = [self._normalize_url(url) for url in self._as_list(item.get("pic_url")) if self._normalize_url(url)]
            if not pic_urls:
                continue
            car_id = str(item.get("car_id") or "")
            car_meta = self._as_dict(car_lookup.get(car_id))
            color_keys = self._as_list(car_meta.get("color_keys"))
            color_names = []
            for key in color_keys:
                color_meta = self._as_dict(color_lookup.get(str(key)))
                color_name = color_meta.get("color_name") or ""
                if color_name and color_name not in color_names:
                    color_names.append(color_name)
            selected_images.append(
                {
                    "rank": idx,
                    "car_id": item.get("car_id"),
                    "car_name": item.get("car_name") or car_meta.get("car_text") or car_meta.get("name"),
                    "image_url": pic_urls[0],
                    "candidate_image_urls": pic_urls,
                    "sale_status": item.get("sale_status"),
                    "available_color_names": color_names,
                    "selection_rule": "first_picture",
                    "required_view": spec["required_view"],
                }
            )

        available_colors = []
        for color_meta in color_lookup.values():
            available_colors.append(
                {
                    "color_name": color_meta.get("color_name") or "",
                    "sub_color_name": color_meta.get("sub_color_name") or "",
                    "hex_color": color_meta.get("color") or "",
                }
            )

        return {
            "module_key": module_key,
            "category_name": spec["category_name"],
            "required_view": spec["required_view"],
            "source_url": f"{self.base_url}/auto/series/{picture_info.get('series_id') or page_props.get('query', {}).get('seriesId')}/{spec['path']}",
            "selected_images": selected_images,
            "available_colors": available_colors,
            "sample_count": len(selected_images),
        }

    def _build_compact_image_floor_data(self, selected_modules: dict) -> dict:
        floor_image_list = []
        for module_key in ("wg", "ns"):
            module = self._as_dict(selected_modules.get(module_key))
            image_list = []
            for sample in self._as_list(module.get("selected_images")):
                sample = self._as_dict(sample)
                image_list.append(
                    {
                        "image_url": sample.get("image_url"),
                        "car_id": sample.get("car_id"),
                        "car_name": sample.get("car_name"),
                        "color_name": ",".join(self._as_list(sample.get("available_color_names"))[:3]),
                        "required_view": sample.get("required_view") or module.get("required_view"),
                    }
                )
            floor_image_list.append(
                {
                    "category": module_key,
                    "text": module.get("category_name") or "",
                    "selection_rule": "first_picture",
                    "required_view": module.get("required_view") or "",
                    "image_list": image_list,
                    "pic_count": len(image_list),
                }
            )
        return {"floor_head_list": [], "floor_image_list": floor_image_list}

    def save_series_images(self, series_id: str, raw_json: dict, include_contextual_images: bool = True) -> dict:
        tasks = self._extract_image_download_tasks(raw_json, include_contextual_images=include_contextual_images)
        series_image_dir = os.path.join(self.raw_image_dir, f"series_{series_id}")
        os.makedirs(series_image_dir, exist_ok=True)
        manifest_path = os.path.join(self.raw_dir, f"series_{series_id}_images.json")

        manifest_items = []
        success_count = 0
        failed_count = 0

        for task in tasks:
            ok, resp, retry_count, http_status, err = self._request_binary_with_retry(task["image_url"], timeout=20)
            item = {
                **task,
                "status": "failed",
                "retry_count": retry_count,
                "http_status": http_status,
                "error": err,
                "local_path": "",
                "content_type": "",
                "bytes": 0,
            }
            if ok and resp is not None:
                try:
                    content_type = resp.headers.get("content-type", "")
                    ext = self._infer_extension(task["image_url"], content_type)
                    filename = f"{int(task['rank']):05d}_{task['source_section']}_{task['asset_id']}{ext}"
                    local_path = os.path.join(series_image_dir, filename)
                    total_bytes = 0
                    with open(local_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if not chunk:
                                continue
                            f.write(chunk)
                            total_bytes += len(chunk)
                    item["status"] = "success"
                    item["local_path"] = os.path.abspath(local_path)
                    item["content_type"] = content_type
                    item["bytes"] = total_bytes
                    item["error"] = ""
                    success_count += 1
                except Exception as e:
                    item["error"] = str(e)
                    failed_count += 1
            else:
                failed_count += 1
            manifest_items.append(item)

        manifest = {
            "series_id": str(series_id),
            "image_count": len(tasks),
            "success_count": success_count,
            "failed_count": failed_count,
            "items": manifest_items,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return {
            "image_manifest_path": os.path.abspath(manifest_path),
            "image_dir": os.path.abspath(series_image_dir),
            "image_count": len(tasks),
            "image_success_count": success_count,
            "image_failed_count": failed_count,
        }

    def get_homepage_series_ids(self, limit: int = 5) -> list[str]:
        """从首页获取热门车系的 ID 列表"""
        logger.info("正在从懂车帝首页获取车系 ID...")
        try:
            ok, resp, _, _, err = self._request_with_retry(self.base_url, timeout=10)
            if not ok or resp is None:
                raise RuntimeError(err or "请求首页失败")
            # 使用正则快速提取主页上的所有 /auto/series/xxxx 链接
            ids = list(set(re.findall(r'/auto/series/(\d+)', resp.text)))
            logger.info(f"首页共发现 {len(ids)} 个车系 ID")
            return ids[:limit]
        except Exception as e:
            logger.error(f"获取首页车系 ID 失败: {e}")
            return []

    def _extract_series_ids_from_html(self, html_text: str) -> list[str]:
        return list(set(re.findall(r"/auto/series/(\d+)", html_text or "")))

    def _extract_series_ids_from_local_cache(self) -> list[str]:
        """从本地 raw 目录回收历史抓取过的车系 ID，作为额外入口。"""
        if not os.path.isdir(self.raw_dir):
            return []
        ids = set()
        for name in os.listdir(self.raw_dir):
            m = re.match(r"series_(\d+)\.(json|html)$", name)
            if m:
                ids.add(m.group(1))
        return list(ids)

    def _collect_series_ids_from_entry_urls(self, entry_urls: list[str], max_pages_per_entry: int = 3) -> list[str]:
        """
        从多个公共入口页收集车系ID。
        - 如果 URL 包含 {page}，会按分页尝试。
        """
        seen = set()
        for template in entry_urls:
            if "{page}" in template:
                page_urls = [
                    template.format(page=page)
                    for page in range(1, max(1, max_pages_per_entry) + 1)
                ]
            else:
                page_urls = [template]
            for url in page_urls:
                ok, resp, _, _, err = self._request_with_retry(url, timeout=10)
                if not ok or resp is None:
                    logger.debug(f"入口页抓取失败: url={url}, error={err}")
                    continue
                for sid in self._extract_series_ids_from_html(resp.text):
                    seen.add(sid)
        return list(seen)

    def _collect_series_ids_from_search_keywords(self, keywords: list[str], max_requests: int = 20) -> list[str]:
        """
        通过站内搜索页扩展车系池。
        该入口天然会覆盖更多冷门车系，通常比首页更全。
        """
        seen = set()
        used = 0
        for kw in keywords:
            if used >= max_requests:
                break
            url = f"{self.base_url}/search?keyword={quote(kw)}"
            ok, resp, _, _, err = self._request_with_retry(url, timeout=10)
            used += 1
            if not ok or resp is None:
                logger.debug(f"搜索入口抓取失败: keyword={kw}, error={err}")
                continue
            for sid in self._extract_series_ids_from_html(resp.text):
                seen.add(sid)
        return list(seen)

    def collect_series_ids(self, target_count: int = 300, max_expand_requests: int = 30) -> list[str]:
        """多入口收集车系ID：本地缓存 + 首页 + 公共入口页 + 搜索页，再做关联扩展。"""
        seen = set()
        queue = []

        def _push(ids: list[str], source: str):
            before = len(seen)
            for sid in ids:
                if sid in seen:
                    continue
                seen.add(sid)
                queue.append(sid)
                if len(seen) >= target_count:
                    break
            added = len(seen) - before
            if added > 0:
                logger.info(f"{source} 新增车系ID: +{added}, 当前总数: {len(seen)}")

        # 入口1：本地历史抓取缓存（真实数据）
        _push(self._extract_series_ids_from_local_cache(), "本地缓存入口")

        # 入口2：首页
        _push(self.get_homepage_series_ids(limit=max(target_count, 200)), "首页入口")

        # 入口3：公共入口页（榜单/库页等）
        entry_urls = [
            f"{self.base_url}/auto/library/x-x-x-x-x-x-x-x-x-x",
            f"{self.base_url}/auto/library/x-x-x-x-x-x-x-x-x-x?page={{page}}",
        ]
        page_budget = max(2, min(8, max_expand_requests // 5))
        _push(
            self._collect_series_ids_from_entry_urls(
                entry_urls=entry_urls,
                max_pages_per_entry=page_budget,
            ),
            "公共入口页",
        )

        # 入口4：搜索页关键词（品牌+级别+热门词）
        keywords = [
            "比亚迪", "吉利", "长安", "奇瑞", "理想", "蔚来", "小鹏", "问界",
            "丰田", "本田", "大众", "宝马", "奔驰", "奥迪", "日产", "福特",
            "SUV", "轿车", "MPV", "混动", "纯电", "增程", "皮卡", "轻客",
        ]
        search_budget = max(8, min(30, max_expand_requests))
        _push(
            self._collect_series_ids_from_search_keywords(
                keywords=keywords,
                max_requests=search_budget,
            ),
            "搜索入口",
        )

        if not seen:
            return []

        expand_count = 0

        while queue and len(seen) < target_count and expand_count < max_expand_requests:
            current = queue.pop(0)
            url = f"{self.base_url}/auto/series/{current}"
            ok, resp, _, _, err = self._request_with_retry(url, timeout=10)
            if not ok or resp is None:
                logger.debug(f"扩展车系池失败: series_id={current}, error={err}")
                continue

            expand_count += 1
            discovered = self._extract_series_ids_from_html(resp.text)
            for sid in discovered:
                if sid in seen:
                    continue
                seen.add(sid)
                queue.append(sid)
                if len(seen) >= target_count:
                    break

        ids = list(seen)
        logger.info(
            f"车系池收集完成: 当前 {len(ids)} 个ID, 目标 {target_count}, 扩展请求 {expand_count}/{max_expand_requests}"
        )
        return ids[:target_count]

    def fetch_series_data(self, series_id: str) -> tuple[dict, dict]:
        """按固定 3 个模块抓取车系数据：首页 + images-wg + images-ns。"""
        url = f"{self.base_url}/auto/series/{series_id}"
        logger.info(f"正在抓取车系页: {url}")
        fetch_meta = {
            "url": url,
            "retry_count": 0,
            "http_status": None,
            "error": "",
            "module_meta": {},
            "image_manifest_path": "",
            "image_dir": "",
            "image_count": 0,
            "image_success_count": 0,
            "image_failed_count": 0,
        }
        try:
            home_html_path = os.path.join(self.raw_dir, f"series_{series_id}.html")
            home_json_path = os.path.join(self.raw_dir, f"series_{series_id}.json")
            home_data, home_meta = self._fetch_page_next_data(url, home_html_path, home_json_path, timeout=10)
            fetch_meta["retry_count"] = home_meta["retry_count"]
            fetch_meta["http_status"] = home_meta["http_status"]
            if not home_data:
                fetch_meta["error"] = home_meta["error"] or "主页请求失败"
                return {}, fetch_meta

            sanitized = self._strip_news_sections(home_data)
            page_props = self._as_dict(self._as_dict(sanitized.get("props")).get("pageProps"))
            selected_modules = {}
            for module_key, spec in self.IMAGE_MODULE_SPECS.items():
                module_url = f"{self.base_url}/auto/series/{series_id}/{spec['path']}"
                module_html_path = os.path.join(self.raw_dir, f"series_{series_id}_{spec['path']}.html")
                module_json_path = os.path.join(self.raw_dir, f"series_{series_id}_{spec['path']}.json")
                module_data, module_meta = self._fetch_page_next_data(
                    module_url,
                    module_html_path,
                    module_json_path,
                    timeout=10,
                )
                fetch_meta["module_meta"][module_key] = module_meta
                if not module_data:
                    logger.warning(f"车系 {series_id} 的 {module_key} 模块抓取失败: {module_meta.get('error')}")
                    continue
                selected_modules[module_key] = self._build_selected_image_module(module_key, module_data)

            page_props["selectedImageModules"] = selected_modules
            page_props["imageFloorData"] = self._build_compact_image_floor_data(selected_modules)
            page_props["seriesHomeHead"] = self._as_dict(page_props.get("seriesHomeHead"))
            page_props["seriesHomeHead"]["series_image_info_list"] = []
            page_props["seriesHomeHead"]["pics_summary_info"] = []

            with open(home_json_path, "w", encoding="utf-8") as f:
                json.dump(sanitized, f, ensure_ascii=False, indent=2)

            try:
                image_meta = self.save_series_images(series_id=str(series_id), raw_json=sanitized, include_contextual_images=False)
                fetch_meta.update(image_meta)
            except Exception as image_error:
                logger.warning(f"车系 {series_id} 原始图片保存失败: {image_error}")
                
            return sanitized, fetch_meta
            
        except Exception as e:
            logger.error(f"抓取车系 {series_id} 失败: {e}")
            fetch_meta["error"] = str(e)
            return {}, fetch_meta

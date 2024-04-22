import os
import time
import requests
from tqdm import tqdm
from urllib.parse import urlparse
from cgi import parse_header
import aria2p

def downloader(url, dir, filename=None, overwrite=False, timeout=30, max_retries=3, retry_interval=5):
    os.makedirs(dir, exist_ok=True)
    retries = 0
    download_path = None
    while retries < max_retries:
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                if filename is None:
                    content_disposition = r.headers.get('content-disposition')
                    if content_disposition:
                        _, params = parse_header(content_disposition)
                        filename = params.get('filename')
                    else:
                        filename = os.path.basename(urlparse(url).path)
                download_path = os.path.join(dir, filename)
                if os.path.exists(download_path) and overwrite == False:
                    print(f"🟡 Existed, skip: {filename} -> {dir}")
                    return None
                print(f"⬇️ Downloading: {filename} -> {dir}")
                download_size = int(r.headers.get('content-length', 0))
                progress_bar = tqdm(total=download_size,
                                    unit='iB', unit_scale=True)
                with open(download_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
                progress_bar.close()
                print(f"🟢 Downloaded: {filename} -> {dir}")
                return True
        except Exception as e:
            print(f"🔴 Error: {e}")
            retries += 1
            print(f"⏳ Retrying download, sleep {
                  retry_interval}s... ({retries}/{max_retries})")
            time.sleep(retry_interval)
    print(f"🔴 Failed to download: {url}")
    if download_path and os.path.exists(download_path):
        os.remove(download_path)
    return False

def downloader_aria2(url, dir, filename=None, timeout=30, max_retries=3, retry_interval=5):
    print(f"⬇️ Downloading: {url}")
    os.makedirs(dir, exist_ok=True)
    client = aria2p.Client(host="http://localhost", port=6800)
    api = aria2p.API(client)
    retries = 0
    while retries < max_retries:
        try:
            if filename is None:
                download = api.add_uri(
                    [url], options={"dir": dir, "continue": "true", "timeout": timeout})
                filename = download.name
            else:
                download = api.add_uri([url], options={
                    "dir": dir,
                    "out": filename,
                    "continue": "true",
                    "timeout": timeout
                })
            while not download.is_complete:
                download.update()
                if download.error_message:
                    raise Exception(download.error_message)
        except Exception as e:
            print(f"🟡 Error downloading file: {e}")
            retries += 1
            print(f"⏳ Retrying download, sleep {
                  retry_interval}s... ({retries}/{max_retries})")
            time.sleep(retry_interval)
    if retries >= max_retries:
        print(f"🔴 Failed to download: {url}")
        return False
    print(f"🟢 {filename} -> {dir}: downloaded.")
    return True

def download_models(models_dict, model_path, include_category=None, include_tags=None, downloader=downloader):
    print(f"⬇️ Start: downloads models to {model_path}")
    if include_category:
        print(f"🟡 Include categories: {
              include_category}, will download these model categories only.")
    total_list = []
    downloaded_list = []
    skipped_list = []
    error_list = []
    for type_name, type in models_dict.items():
        for model in type['items']:
            model_category = model.get('category')
            model_tags = model.get('tags')
            if include_category and model_category and model_category not in include_category:
                print(f"🟡 Skipped: {model['url']} not in category {
                      include_category}")
                skipped_list.append(model['url'])
                continue
            if include_tags and model_tags and not set(model_tags).isdisjoint(include_tags):
                print(f"🟡 Skipped: {model['url']} not in tags {include_tags}")
                skipped_list.append(model['url'])
                continue
            download_url = model['url']
            total_list.append(download_url)
            download_dir = os.path.join(model_path, model.get('dir') or type.get(
                'dir') or os.path.join(type_name, model.get('category') or type_name))
            download_filename = model.get('filename')
            download_status = downloader(
                download_url, download_dir, download_filename)
            if download_status:
                downloaded_list.append(download_url)
            elif download_status is None:
                skipped_list.append(download_url)
            else:
                error_list.append(download_url)
    total_counts = len(total_list)
    downloaded_counts = len(downloaded_list)
    skipped_counts = len(skipped_list)
    error_counts = len(error_list)
    print(f"⬇️ Finished: downloads models to {model_path}")
    print(f"📦 Total: {total_counts}"
          f"\n🟢 Downloaded: {downloaded_counts}"
          f"\n🟡 Skipped: {skipped_counts}"
          f"\n🔴 Error: {error_counts}")
    if error_counts > 0:
        for i in error_list:
            print(f"- {i}")
    return True

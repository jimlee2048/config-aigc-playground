import os
import subprocess
import shutil
import re
import time
from pathlib import Path
import tomllib
import urllib.parse
import json
import git
import giturlparse
from comfy_cli.config_manager import ConfigManager
import comfy_cli.constants as cli_constants
import logging
from rich.console import Console
from rich.logging import RichHandler
from collections import defaultdict
import aria2p


console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False, level="NOTSET")]
)
logger = logging.getLogger("boot")

def get_bool_env(var_name: str, default: bool = False) -> bool:
  value = os.environ.get(var_name)
  if value is None:
    return default
  value = value.lower()
  if value in ("true", "1", "t", "yes", "y"):
    return True
  elif value in ("false", "0", "f", "no", "n"):
    return False
  else:
    # raise ValueError(f"Invalid bool value for environment variable '{var_name}': '{value}'")
    return default

def compile_pattern(pattern_str: str) -> re.Pattern:
    if not pattern_str:
        return None
    try:
        return re.compile(pattern_str)
    except re.error as e:
        logger.error(f"❌ Invalid regex pattern: {pattern_str}\n{str(e)}")
        return None

def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError

def exec_command(command: list[str], cwd: str = None) -> int:
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd) as proc:
        for line in proc.stdout:
            logger.info(line.strip())
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, command)
        return proc.returncode

def exec_script(path: str) -> bool:
    script = Path(path)
    if not script.is_file():
        logger.warning(f"⚠️ Invalid script path: {path}")
        return False
    try:
        logger.info(f"🛠️ Executing: {path}")
        exec_command(["bash", str(script)])
        logger.info(f"✅ Script completed: {path}")
        return True
    except Exception as e:
        logger.error(f"❌ Script error: {path}\n{str(e)}")
        return False

class BootProgress:
    def __init__(self):
        self.total_steps = 0
        self.current_step = 0
        self.log_level_info = "info"
        self.log_level_warning = "warning"
        self.log_level_error = "error"

    def start(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0

    def advance(self, msg: str = None, style: str = None):
        self.current_step += 1
        if msg:
            self.log_progress(msg, style)

    def log_progress(self, msg: str = None, style: str = None):
        overall_progress = f"[{self.current_step}/{self.total_steps}]"
        if msg is None:
            logger.info(f"{overall_progress}")
            return
        if style == self.log_level_info:
            logger.info(f"{overall_progress}: {msg}")
        elif style == self.log_level_warning:
            logger.warning(f"{overall_progress}: {msg}")
        elif style == self.log_level_error:
            logger.error(f"{overall_progress}: {msg}")
        else:
            logger.info(f"{overall_progress}: {msg}")

class BootConfigManager:
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.node_exclude = ["ComfyUI-Manager", "comfyui-manager"]
        self.current_config = self.load_boot_config()
        self.prev_config = self.load_prev_config(BOOT_CONFIG_PREV_PATH)
        self.current_nodes = self.load_nodes_config(self.current_config)
        self.current_models = self.load_models_config(self.current_config)
        self.prev_nodes = self.load_nodes_config(self.prev_config)
        self.prev_models = self.load_models_config(self.prev_config)


    def _drop_duplicates_config(self, config: list[dict], cond_key: list[str]) -> tuple[list[dict], list[dict], int]:
        unique_items = []
        duplicate_items = []
        unique_kv_tracker = set()
        for item in config:
            unique_kv = tuple(item[key] for key in cond_key)
            if unique_kv not in unique_kv_tracker:
                unique_items.append(item)
                unique_kv_tracker.add(unique_kv)
            else:
                duplicate_items.append(item)
        return unique_items, duplicate_items

    def _preprocess_url(self, url: str) -> str:
        parsed_url = urllib.parse.urlparse(url)
        # check if url is valid
        if not parsed_url.netloc:
            raise Exception(f"Invalid URL: {url}")
        # chinese mainland network settings
        if BOOT_CN_NETWORK:
            fr_map = {
                'huggingface.co': 'hf-mirror.com',
                'civitai.com': 'civitai.work'
            }
            if parsed_url.netloc in fr_map:
                url = parsed_url._replace(netloc=fr_map[parsed_url.netloc]).geturl()
        return url

    def load_boot_config(self) -> dict:
        logger.info(f"📂 Loading boot config from {self.config_dir}")

        config_dir = Path(self.config_dir)
        if not config_dir.is_dir():
            if config_dir.is_file():
                logger.warning(f"⚠️ Invalid boot config detected, removing")
                shutil.rmtree(config_dir)
            logger.info(f"ℹ️ No boot config found, using default settings")
            return {}
        
        config_files = list(config_dir.rglob("*.toml"))
        
        if BOOT_CONFIG_INCLUDE or BOOT_CONFIG_EXCLUDE:
            include_pattern = compile_pattern(BOOT_CONFIG_INCLUDE)
            exclude_pattern = compile_pattern(BOOT_CONFIG_EXCLUDE)
            filtered_files = [
                f for f in config_files
                if (not include_pattern or include_pattern.search(f.name)) and
                (not exclude_pattern or not exclude_pattern.search(f.name))
            ]
            if include_pattern:
                logger.info(f"⚡ Include filter: {BOOT_CONFIG_INCLUDE}")
            if exclude_pattern:
                logger.info(f"⚡ Exclude filter: {BOOT_CONFIG_EXCLUDE}")
            config_files = filtered_files

        logger.info(f"📄 Found {len(config_files)} config files:")
        for file in config_files:
            logger.info(f"└─ {file}")
        
        boot_config = defaultdict(list)
        try:
            for file in config_files:
                config = tomllib.loads(file.read_text())
                for key, value in config.items():
                    boot_config[key].extend(value)
        except Exception as e:
            logger.error(f"❌ Failed to load boot config: {str(e)}")
            exit(1)
        return boot_config

    def load_prev_config(self, prev_path: Path) -> dict:
        if prev_path.is_file():
            logger.info(f"📂 Loading previous config: {prev_path}")
        elif prev_path.is_dir():
            logger.warning(f"⚠️ Invalid previous config detected, removing")
            shutil.rmtree(prev_path)
        else:
            logger.info(f"ℹ️ No previous config found")
            return {}
        return json.loads(prev_path.read_text())

    def write_config_cache(self, path: Path, config: dict) -> bool:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(config,default=json_default,indent=4))
            logger.info(f"✅ Current config saved to {path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save current config: {str(e)}")
            return False

    def load_nodes_config(self, boot_config: dict) -> list[dict]:
        nodes_config = boot_config.get('custom_nodes', [])
        if not nodes_config:
            return []

        for node in nodes_config.copy():
            try:
                node['url'] = self._preprocess_url(node['url'])
                node_repo = giturlparse.parse(node['url'])
                # validate git url
                if not node_repo.valid:
                    raise Exception(f"Invalid git URL: {node['url']}")
                # parse custom node name from URL
                node['name'] = node.get('name', node_repo.name)
                node['alt_name'] = node.get('alt_name', node['name'].lower())
                should_exclude = (
                    node['name'] in self.node_exclude
                    or node['alt_name'] in self.node_exclude
                )
                if should_exclude:
                    logger.warning(f"⚠️ Skip excluded node: {node['name']}")
                    nodes_config.remove(node)
                    continue
            except KeyError as e:
                logger.warning(f"⚠️ Invalid node config: {node}\n{str(e)}")
                continue

        # drop duplicates
        nodes_config, duplicates = self._drop_duplicates_config(nodes_config, ['name'])
        if duplicates:
            logger.warning(f"⚠️ Found {len(duplicates)} duplicate nodes:")
            for node in duplicates:
                logger.warning(f"└─ {node['name']}")

        return nodes_config

    def load_models_config(self, boot_config: dict) -> list[dict]:
        models_config = boot_config.get('models', [])
        if not models_config:
            return []

        for model in models_config.copy():
            try:
                model['url'] = self._preprocess_url(model['url'])
                model['path'] = str(COMFYUI_PATH / model['dir'] / model['filename'])
            except KeyError as e:
                logger.warning(f"⚠️ Invalid model config: {model}\n{str(e)}")
                continue

        # drop duplicates
        models_config, duplicates = self._drop_duplicates_config(models_config, ['path'])
        if duplicates:
            logger.warning(f"⚠️ Found {len(duplicates)} duplicate models:")
            for model in duplicates:
                logger.warning(f"└─ {model['filename']}")

        return models_config


class NodeManager:
    def __init__(self, comfyui_path: Path):
        self.comfyui_path = comfyui_path
        self.progress = BootProgress()
        self.node_exclude = ["ComfyUI-Manager", "comfyui-manager"]
        self.failed_list = []

    def _is_valid_git_repo(self, path: str) -> bool:        
        try:
            _ = git.Repo(path).git_dir
            return True
        except Exception as e:
            return False

    def is_node_exists(self, config: dict) -> bool:
        node_name = config['name']
        node_alt_name = config.get('alt_name', node_name.lower())
        possible_paths = { self.comfyui_path / "custom_nodes" / name for name in [node_name, node_alt_name] } 

        for p in possible_paths:
            if p.exists() and self._is_valid_git_repo(p):
                return True
            elif p.is_dir():
                logger.warning(f"⚠️ {node_name} invalid, removing: {p}")
                shutil.rmtree(p)
            elif p.is_file():
                logger.warning(f"⚠️ {node_name} invalid, removing: {p}")
                p.unlink()
        return False

    def install_node(self, config: dict) -> bool:
        try:
            node_name = config['name']
            node_url = config['url']
            if node_name in self.node_exclude:
                self.progress.advance(msg=f"⚠️ Not allowed to install node: {node_name}", style="warning")
                return False
            if self.is_node_exists(config):
                self.progress.advance(msg=f"ℹ️ {node_name} already exists, skip.", style="info")
                return True
            self.progress.advance(msg=f"📦 Installing node: {node_name}", style="info")
            exec_command(["python", str(COMFYUI_MN_PATH / "cm-cli.py"), "install", node_url])
            if 'script' in config:
                exec_script(config['script'])
            return True
        except Exception as e:
            logger.error(f"❌ Failed to install node {node_name}: {str(e)}")
            return False

    def uninstall_node(self, config: dict) -> bool:
        try:
            node_name = config['name']
            node_alt_name = config.get('alt_name', node_name.lower())
            if node_name in self.node_exclude or node_alt_name in self.node_exclude:
                self.progress.advance(msg=f"⚠️ Not allowed to uninstall node: {node_name}", style="warning")
                return False
            if not self.is_node_exists(config):
                self.progress.advance(msg=f"ℹ️ {node_name} not found, skip.", style="info")
                return True
            possible_paths = { self.comfyui_path / "custom_nodes" / name for name in [node_name, node_alt_name] }
            self.progress.advance(msg=f"🗑️ Uninstalling node: {node_name}", style="info")
            for node_path in possible_paths:
                if node_path.exists():
                    shutil.rmtree(node_path)
            logger.info(f"✅ Uninstalled node: {node_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to uninstall node {node_name}: {str(e)}")
            return False

    def init_nodes(self, current_config: list[dict], prev_config: list[dict] = None) -> bool:
        if not current_config:
            logger.info(f"📦 No nodes in config")
            return False

        if not prev_config:
            install_nodes = current_config
            uninstall_nodes = []
        else:
            install_nodes = [node for node in current_config if node not in prev_config]
            uninstall_nodes = [node for node in prev_config if node not in current_config]

        if not install_nodes and not uninstall_nodes:
            logger.info(f"ℹ️ No changes in nodes")
            return False
        if install_nodes:
            install_count = len(install_nodes)
            logger.info(f"📦 Installing {install_count} nodes:")
            for node in install_nodes:
                logger.info(f"└─ {node['url']}")
            self.progress.start(install_count)
            for node in install_nodes:
                if not self.install_node(node):
                    self.failed_list.append(node)
        if uninstall_nodes:
            uninstall_count = len(uninstall_nodes)
            logger.info(f"🗑️ Uninstalling {uninstall_count} nodes:")
            for node in uninstall_nodes:
                logger.info(f"└─ {node['name']}")
            self.progress.start(uninstall_count)
            for node in uninstall_nodes:
                if not self.uninstall_node(node):
                    self.failed_list.append(node)
        if self.failed_list:
            logger.error(f"❌ Failed to process {len(self.failed_list)} nodes:")
            for node in self.failed_list:
                logger.error(f"└─ {node['name']}")
            return False
        return True

class ModelManager:
    def __init__(self, comfyui_path: Path):
        self.comfyui_path = comfyui_path
        self.progress = BootProgress()
        self.failed_list = []
        self.aria2 = aria2p.API(
            aria2p.Client(
                host="http://localhost",
                port=6800,
                secret=""
            )
        )
        self._start_aria2c()

    def _start_aria2c(self):
        try:
            subprocess.run([
                "aria2c", 
                "--daemon=true",
                "--enable-rpc",
                "--rpc-listen-port=6800",
                "--max-concurrent-downloads=1",
                "--max-connection-per-server=16",
                "--split=16",
                "--continue=true",
                "--disable-ipv6=true",
            ], check=True)
            # sleep to ensure aria2c is ready
            time.sleep(1)
        except Exception as e:
            logger.error(f"❌ Failed to start aria2c: {str(e)}")

    def _is_huggingface_url(self, url: str) -> bool:
        parsed_url = urllib.parse.urlparse(url)
        return parsed_url.netloc in ["hf.co", "huggingface.co", "huggingface.com", "hf-mirror.com"]
    
    def _is_civilai_url(self, url: str) -> bool:
        parsed_url = urllib.parse.urlparse(url)
        return parsed_url.netloc in ["civitai.com", "civitai.work"]

    def is_model_exists(self, config: dict) -> bool:
        model_path = Path(config['path'])
        model_dir = model_path.parent
        model_filename = config['filename']

        # remove huggingface cache
        if (model_dir / ".cache").exists():
            logger.warning(f"⚠️ Found cache directory: {str(model_dir)}, removing...")
            shutil.rmtree(model_dir / ".cache")

        # check if previous download cache exists
        previous_download_cache = model_dir / (model_filename + ".aria2")
        previous_download_exists = previous_download_cache.exists()
        if previous_download_exists:
            logger.warning(f"⚠️ Found previous download cache: {str(previous_download_cache)}, removing...")
            previous_download_cache.unlink()

        if model_path.exists():
            if previous_download_exists:
                logger.warning(f"⚠️ {model_filename} download incomplete, removing...")
                model_path.unlink()
                return False
            if model_path.is_file():
                return True
            else:
                logger.warning(f"⚠️ {model_filename} invalid, removing: {model_path}")
                shutil.rmtree(model_path)
        return False

    def download_model(self, config: dict) -> bool:
        model_url = config['url']
        model_dir = config['dir']
        model_filename = config['filename']

        if self.is_model_exists(config):
            self.progress.advance(msg=f"ℹ️ {model_filename} already exists in {model_dir}, skip.", style="info")
            return True
    
        self.progress.advance(msg=f"⬇️ Downloading: {model_filename} -> {model_dir}", style="info")
        headers = defaultdict(str)
        if self._is_huggingface_url(model_url) and HF_API_TOKEN:
            headers['Content-Type'] = "application/json"
            headers['Authorization'] = f"Bearer {HF_API_TOKEN}"
        if self._is_civilai_url(model_url) and CIVITAI_API_TOKEN:
            headers['Content-Type'] = "application/json"
            headers['Authorization'] = f"Bearer {CIVITAI_API_TOKEN}"

        for attempt in range(1, 4):
            try:
                download = self.aria2.add_uris([model_url], {
                    "dir": str(self.comfyui_path / model_dir),
                    "out": model_filename,
                    "header": headers
                })
                while not download.is_complete:
                    download.update()
                    if download.status == "error":
                        raise Exception(f"{download.error_message}")
                    if download.status == "removed":
                        raise Exception(f"Download was removed")
                    self.progress.log_progress(f"{model_filename}: {download.progress_string()} | {download.completed_length_string()}/{download.total_length_string()} [{download.eta_string()}, {download.download_speed_string()}]", "info")
                    time.sleep(1)
                logger.info(f"✅ Downloaded: {model_filename} -> {model_dir}")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Download attempt {attempt} failed: {str(e)}")
        logger.error(f"❌ Exceeded max retries: {model_filename} -> {model_dir}")
        return False

    def move_model(self, src: Path, dst: Path) -> bool:
        try:
            self.progress.advance(msg=f"📦 Moving: {src} -> {dst}", style="info")
            src.rename(dst)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to move model: {src} -> {dst}\n{str(e)}")
            return False

    def remove_model(self, config: dict) -> bool:
        try:
            model_path = Path(config['path'])
            model_filename = config['filename']
            if not self.is_model_exists(config):
                self.progress.advance(msg=f"ℹ️ {model_filename} not found in path: {model_path}, skip.", style="info")
                return True
            self.progress.advance(msg=f"🗑️ Removing model: {model_filename}", style="info")
            model_path.unlink()
            logger.info(f"✅ Removed model: {model_filename}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to remove model {model_filename}: {str(e)}")
            return False

    def init_models(self, current_config: list, prev_config: list = None) -> bool:
        if not current_config:
            logger.info(f"📦 No models in config")
            return False
        
        models_to_download = []
        models_to_move = []
        models_to_remove = []
        if not prev_config:
            models_to_download = current_config
        else:
            for model in current_config:
                if model not in prev_config:
                    models_to_download.append(model)
                else:
                    prev_model = next((m for m in prev_config if m['url'] == model['url']), None)
                    prev_path = Path(prev_model['path'])
                    current_path = Path(model['path'])
                    if current_path != prev_path:
                        models_to_move.append({"src": prev_path, "dst": current_path})
            for prev_model in prev_config:
                if not any(model['url'] == prev_model['url'] for model in current_config):
                    models_to_remove.append(prev_model)

        if not models_to_download and not models_to_move and not models_to_remove:
            logger.info(f"ℹ️ No changes in models")
            return False
        if models_to_download:
            download_count = len(models_to_download)
            logger.info(f"⬇️ Downloading {download_count} models:")
            for model in models_to_download:
                logger.info(f"└─ {model['filename']}")
            self.progress.start(download_count)
            for model in models_to_download:
                if not self.download_model(model):
                    self.failed_list.append(model)
        if models_to_move:
            move_count = len(models_to_move)
            logger.info(f"📦 Moving {move_count} models:")
            for model in models_to_move:
                logger.info(f"└─ {model['src']} -> {model['dst']}")
            self.progress.start(move_count)
            for model in models_to_move:
                if not self.move_model(model['src'], model['dst']):
                    self.failed_list.append(model)
        if models_to_remove:
            remove_count = len(models_to_remove)
            logger.info(f"🗑️ Removing {remove_count} models:")
            for model in models_to_remove:
                logger.info(f"└─ {model['filename']}")
            self.progress.start(remove_count)
            for model in models_to_remove:
                if not self.remove_model(model):
                    self.failed_list.append(model)
        if self.failed_list:
            logger.error(f"❌ Failed to process {len(self.failed_list)} models:")
            for model in self.failed_list:
                logger.error(f"└─ {model['filename']}")
            return False
        return True


class ComfyUIInitializer:
    def __init__(self, config_dir: Path = None, comfyui_path: Path = None):
        self.config_loader = BootConfigManager(config_dir)
        self.current_config = self.config_loader.current_config
        self.prev_config = self.config_loader.prev_config
        self.current_nodes = self.config_loader.current_nodes
        self.current_models = self.config_loader.current_models
        self.prev_nodes = self.config_loader.prev_nodes
        self.prev_models = self.config_loader.prev_models
        self.node_manager = NodeManager(comfyui_path)
        self.model_manager = ModelManager(comfyui_path)

    def run(self):
        # init nodes and models
        failed_config = defaultdict(list)
        if self.current_config and BOOT_INIT_NODE:
            self.node_manager.init_nodes(self.current_nodes, self.prev_nodes)
            failed_config['nodes'] = self.node_manager.failed_list
        if self.current_config and BOOT_INIT_MODEL:
            self.model_manager.init_models(self.current_models, self.prev_models)
            failed_config['models'] = self.model_manager.failed_list
        self.config_loader.write_config_cache(BOOT_CONFIG_PREV_PATH, {k: v for k, v in self.current_config.items() if k not in failed_config})
        # launch comfyui
        logger.info(f"🚀 Launching ComfyUI...")
        launch_args_list = ["--listen", "0.0.0.0,::", "--port", "8188"] + (COMFYUI_EXTRA_ARGS.split() if COMFYUI_EXTRA_ARGS else [])
        launch_args_str = " ".join(launch_args_list).strip()
        cli_config_manager.set(cli_constants.CONFIG_KEY_DEFAULT_LAUNCH_EXTRAS, launch_args_str)
        subprocess.run(["comfy", "env"], check=True)
        subprocess.run(["comfy", "launch", "--"] + launch_args_list, check=True)

if __name__ == '__main__':
    logger.info(f"Starting boot process")

    # Environment variables
    HF_API_TOKEN = os.environ.get('HF_API_TOKEN', None)
    CIVITAI_API_TOKEN = os.environ.get('CIVITAI_API_TOKEN', None)
    COMFYUI_PATH = Path(os.environ.get('COMFYUI_PATH', "/workspace/ComfyUI"))
    COMFYUI_EXTRA_ARGS = os.environ.get('COMFYUI_EXTRA_ARGS', None)
    COMFYUI_MN_PATH = Path(os.environ.get('COMFYUI_MN_PATH', COMFYUI_PATH / "custom_nodes" / "comfyui-manager"))
    BOOT_CONFIG_DIR = Path(os.environ.get('BOOT_CONFIG_DIR', None))
    BOOT_CONFIG_PREV_PATH = Path.home() / ".cache" / "comfyui" / "boot_config.prev.json"
    BOOT_CONFIG_INCLUDE = os.environ.get('BOOT_CONFIG_INCLUDE', None)
    BOOT_CONFIG_EXCLUDE = os.environ.get('BOOT_CONFIG_EXCLUDE', None)
    BOOT_CN_NETWORK = get_bool_env('BOOT_CN_NETWORK', False)
    BOOT_INIT_NODE = get_bool_env('BOOT_INIT_NODE', False)
    BOOT_INIT_MODEL = get_bool_env('BOOT_INIT_MODEL', False)

    # check if comfyui path exists
    if not COMFYUI_PATH.is_dir():
        logger.error(f"❌ Invalid ComfyUI path: {COMFYUI_PATH}")
        exit(1)

    # chinese mainland network settings
    if BOOT_CN_NETWORK:
        logger.info(f"🌐 Using CN network optimization")
        # pip source to ustc mirror
        os.environ['PIP_INDEX_URL'] = 'https://mirrors.ustc.edu.cn/pypi/web/simple'
        # huggingface endpoint to hf-mirror.com
        os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
        if HF_API_TOKEN:
            logger.warning(f"⚠️ HF_API_TOKEN will be sent to hf-mirror.com")
        if CIVITAI_API_TOKEN:
            logger.warning(f"⚠️ CIVITAIAPI_TOKEN will be sent to civitai.work")

    cli_config_manager = ConfigManager()
    if HF_API_TOKEN:
        cli_config_manager.set(cli_constants.HF_API_TOKEN_KEY, HF_API_TOKEN)
    if CIVITAI_API_TOKEN:
        cli_config_manager.set(cli_constants.CIVITAI_API_TOKEN_KEY, CIVITAI_API_TOKEN)

    app = ComfyUIInitializer(BOOT_CONFIG_DIR, COMFYUI_PATH)
    logger.info(f"Initializing ComfyUI...")
    app.run()
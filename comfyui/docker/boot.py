import os
import subprocess
import shutil
import re
from pathlib import Path
import tomllib
import urllib
import json
import git
import giturlparse
from comfy_cli.config_manager import ConfigManager
import comfy_cli.constants as cli_constants
import logging
from rich.console import Console
from rich.logging import RichHandler
from collections import defaultdict


console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False)]
)
logger = logging.getLogger("boot")

def exec_command(command: list[str], cwd: str = None) -> int:
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd) as proc:
        for line in proc.stdout:
            logger.info(line.strip())
        for line in proc.stderr:
            logger.error(line.strip())
        return proc.returncode

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

class ConfigLoader:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.node_exclude = ["ComfyUI-Manager", "comfyui-manager"]

    def _drop_duplicates_config(self, config: list[dict], cond_key: list[str]) -> tuple[list[dict], int]:
        unique_items = []
        unique_kv_tracker = set()
        for item in config:
            unique_kv = tuple(item[key] for key in cond_key)
            if unique_kv not in unique_kv_tracker:
                unique_items.append(item)
                unique_kv_tracker.add(unique_kv)
        duplicates_count = len(config) - len(unique_items)
        if duplicates_count:
            logger.warning(f"⚠️ Found {duplicates_count} duplicate items")
        return unique_items, duplicates_count

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
        logger.info(f"📂 Loading boot config from {self.config_path}")

        config_path = Path(self.config_path)
        if not config_path.is_dir():
            if config_path.is_file():
                logger.warning(f"⚠️ Invalid boot config detected, removing")
                shutil.rmtree(config_path)
            logger.info(f"ℹ️ No boot config found, using default settings")
            return {}
        
        config_files = list(config_path.rglob("*.toml"))
        
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
            logger.info(f"      └─ {file}")
        
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
        nodes_config, duplicates_count = self._drop_duplicates_config(nodes_config, ['name'])
        if duplicates_count:
            logger.warning(f"⚠️ Found {duplicates_count} duplicate nodes")

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
        models_config, duplicates_count = self._drop_duplicates_config(models_config, ['path'])
        if duplicates_count:
            logger.warning(f"⚠️ Found {duplicates_count} duplicate models")

        return models_config


class NodeManager:
    def __init__(self, comfyui_path: Path):
        self.comfyui_path = comfyui_path
        self.progress = BootProgress()
        self.node_exclude = ["ComfyUI-Manager", "comfyui-manager"]

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
                self.progress.advance(msg=f"⚠️ Cannot install node: {node_name}", style="warning")
                return False
            if self.is_node_exists(config):
                self.progress.advance(msg=f"ℹ️ {node_name} already exists, skip.", style="info")
                return False
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
                self.progress.advance(msg=f"⚠️ Cannot uninstall node: {node_name}", style="warning")
                return False
            if not self.is_node_exists(config):
                self.progress.advance(msg=f"ℹ️ {node_name} not found, skip.", style="info")
                return False
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
                logger.info(f"      └─ {node['url']}")
            self.progress.start(install_count)
            for node in install_nodes:
                self.install_node(node)
        if uninstall_nodes:
            uninstall_count = len(uninstall_nodes)
            logger.info(f"🗑️ Uninstalling {uninstall_count} nodes:")
            for node in uninstall_nodes:
                logger.info(f"      └─ {node['name']}")
            self.progress.start(uninstall_count)
            for node in uninstall_nodes:
                self.uninstall_node(node)
        return True

class ModelManager:
    def __init__(self, comfyui_path: Path):
        self.comfyui_path = comfyui_path
        self.progress = BootProgress()

    def is_model_exists(self, config: dict) -> bool:
        model_path = Path(config['path'])
        model_filename = config['filename']
        if model_path.exists():
            if model_path.is_file():
                return True
            else:
                logger.warning(f"⚠️ {model_filename} invalid, removing: {model_path}")
                shutil.rmtree(model_path)
        return False

    def download_model(self, config: dict) -> bool:
        try:
            model_url = config['url']
            model_dir = config['dir']
            model_filename = config['filename']
            if self.is_model_exists(config):
                self.progress.advance(msg=f"ℹ️ {model_filename} already exists in {model_dir}, skip.", style="info")
                return False
            self.progress.advance(msg=f"⬇️ Downloading model: {model_filename} -> {model_dir}", style="info")
            exec_command(["comfy", "model", "download", "--url", model_url, "--relative-path", model_dir, "--filename", model_filename])
            return True
        except Exception as e:
            logger.error(f"❌ Failed to download model {model_filename}: {str(e)}")
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
                return False
            self.progress.advance(msg=f"🗑️ Removing model: {model_filename}", style="info")
            model_path.unlink()
            logger.info(f"✅ Removed model: {model_filename}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to remove model {model_filename}: {str(e)}")
            return False

    def init_models(self, current_config: list, prev_config: list = None):
        if not current_config:
            logger.info(f"📦 No models in config")
            return False
        
        if not prev_config:
            models_to_download = current_config
            models_to_move = []
            models_to_remove = []

        else:
            models_to_download = []
            models_to_move = []
            for model in current_config:
                if model not in prev_config:
                    models_to_download.append(model)
                else:
                    prev_model = next((m for m in prev_config if m['url'] == model['url']), None)
                    prev_path = Path(prev_model['path'])
                    current_path = Path(model['path'])
                    if current_path != prev_path:
                        models_to_move.append({"src": prev_path, "dst": current_path})
            models_to_remove = []
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
                logger.info(f"      └─ {model['filename']}")
            self.progress.start(download_count)
            for model in models_to_download:
                self.download_model(model)
        if models_to_move:
            move_count = len(models_to_move)
            logger.info(f"📦 Moving {move_count} models:")
            for model in models_to_move:
                logger.info(f"      └─ {model['src']} -> {model['dst']}")
            self.progress.start(move_count)
            for file in models_to_move:
                self.move_model(file['src'], file['dst'])
        if models_to_remove:
            remove_count = len(models_to_remove)
            logger.info(f"🗑️ Removing {remove_count} models:")
            for model in models_to_remove:
                logger.info(f"      └─ {model['filename']}")
            self.progress.start(remove_count)
            for model in models_to_remove:
                self.remove_model(model)
        return True


class ComfyUIInitializer:
    def __init__(self):
        self.config_loader = ConfigLoader(BOOT_CONFIG_DIR)
        self.node_manager = NodeManager(COMFYUI_PATH)
        self.model_manager = ModelManager(COMFYUI_PATH)

    def run(self):
        current_boot_config = self.config_loader.load_boot_config()
        prev_boot_config = self.config_loader.load_prev_config(BOOT_CONFIG_PREV_PATH)
        if current_boot_config and BOOT_INIT_NODE:
            current_nodes = self.config_loader.load_nodes_config(current_boot_config)
            prev_nodes = prev_boot_config.get('custom_nodes', [])
            self.node_manager.init_nodes(current_nodes, prev_nodes)
        if current_boot_config and BOOT_INIT_MODEL:
            current_models = self.config_loader.load_models_config(current_boot_config)
            prev_models = prev_boot_config.get('models', [])
            self.model_manager.init_models(current_models, prev_models)
        self.config_loader.write_config_cache(BOOT_CONFIG_PREV_PATH, current_boot_config)
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

    cli_config_manager = ConfigManager()
    if HF_API_TOKEN:
        cli_config_manager.set(cli_constants.HF_API_TOKEN_KEY, HF_API_TOKEN)
    if CIVITAI_API_TOKEN:
        cli_config_manager.set(cli_constants.CIVITAI_API_TOKEN_KEY, CIVITAI_API_TOKEN)

    app = ComfyUIInitializer()
    logger.info(f"Initializing ComfyUI...")
    app.run()
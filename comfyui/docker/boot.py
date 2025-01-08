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
from rich.console import Console

class BootProgress:
    def __init__(self):
        self.total_steps = 0
        self.current_step = 0

    def start(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0

    def advance(self, msg: str = None, style: str = None):
        self.current_step += 1
        if msg:
            self.print(msg, style)

    def print(self, msg: str = None, style: str = None):
        overall_progress = f"[{self.current_step}/{self.total_steps}]"
        if msg:
            console.print(f"{overall_progress}: {msg}", style=style)

def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError

def compile_pattern(pattern_str: str) -> re.Pattern:
    if not pattern_str:
        return None
    try:
        return re.compile(pattern_str)
    except re.error as e:
        console.print(f"[ERROR] ❌ Invalid regex pattern: {pattern_str}\n{str(e)}", style="red")
        return None

def preprocess_url(url: str) -> str:
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


def drop_duplicates_config(config: list[dict], cond_key: list[str]) -> tuple[list[dict], int]:
    unique_items = []
    unique_kv_tracker = set()
    
    for item in config:
        # Create tuple of values for comparison
        unique_kv = tuple(item[key] for key in cond_key)
        
        if unique_kv not in unique_kv_tracker:
            unique_items.append(item)
            unique_kv_tracker.add(unique_kv)
    
    duplicates_count = len(config) - len(unique_items)
    return unique_items, duplicates_count


def load_models_config(boot_config: dict) -> list[dict]:
    models_config = boot_config.get('models', [])
    if not models_config:
        return []

    for model in models_config.copy():
        try:
            model['url'] = preprocess_url(model['url'])
            model['path'] = str(COMFYUI_PATH / model['dir'] / model['filename'])
        except KeyError as e:
            console.print(f"[WARN] ⚠️ Invalid model config: {model}\n{str(e)}", style="yellow")
            continue

    # drop duplicates
    models_config, duplicates_count = drop_duplicates_config(models_config, ['path'])
    if duplicates_count:
        console.print(f"[WARN] ⚠️ Found {duplicates_count} duplicate models", style="yellow")

    return models_config

def load_nodes_config(boot_config: dict) -> list[dict]:
    nodes_config = boot_config.get('custom_nodes', [])
    if not nodes_config:
        return []

    for node in nodes_config.copy():
        try:
            node['url'] = preprocess_url(node['url'])
            node_repo = giturlparse.parse(node['url'])
            # validate git url
            if not node_repo.valid:
                raise Exception(f"Invalid git URL: {node['url']}")
            # parse custom node name from URL
            node['name'] = node.get('name', node_repo.name)
            node['alt_name'] = node.get('alt_name', node['name'].lower())
            should_exclude = (
                node['name'] in BOOT_INIT_NODE_EXCLUDE
                or node['alt_name'] in BOOT_INIT_NODE_EXCLUDE
            )
            if should_exclude:
                console.print(f"[INFO] ℹ️ Skip excluded node: {node['name']}", style="blue")
                nodes_config.remove(node)
                continue
        except KeyError as e:
            console.print(f"[WARN] ⚠️ Invalid node config: {node}\n{str(e)}", style="yellow")
            continue

    # drop duplicates
    nodes_config, duplicates_count = drop_duplicates_config(nodes_config, ['name'])
    if duplicates_count:
        console.print(f"[WARN] ⚠️ Found {duplicates_count} duplicate nodes", style="yellow")

    return nodes_config


def load_boot_config(path: Path) -> dict:
    console.print(f"[INFO] 📂 Loading boot config from {path}", style="blue")

    config_path = Path(path)
    if not config_path.is_dir():
        if config_path.is_file():
            console.print(f"[WARN] ⚠️ Invalid boot config detected, removing", style="yellow")
            shutil.rmtree(config_path)
        console.print("[INFO] ℹ️ No boot config found, using default settings", style="blue")
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
            console.print(f"[INFO] ⚡ Include filter: {BOOT_CONFIG_INCLUDE}", style="blue")
        if exclude_pattern:
            console.print(f"[INFO] ⚡ Exclude filter: {BOOT_CONFIG_EXCLUDE}", style="blue")
        config_files = filtered_files

    console.print(f"[INFO] 📄 Found {len(config_files)} config files:", style="blue")
    for file in config_files:
        console.print(f"      └─ {file}", style="blue")
    
    boot_config = {}
    try:
        for file in config_files:
            boot_config.update(tomllib.loads(file.read_text()))
    except Exception as e:
        console.print(f"[ERROR] ❌ Failed to load boot config: {str(e)}", style="red")
        exit(1)
    return boot_config

def load_prev_config(path: Path) -> dict:
    if path.is_file():
        console.print(f"[INFO] 📂 Loading previous config: {path}", style="blue")
    elif path.is_dir():
        console.print(f"[WARN] ⚠️ Invalid previous config detected, removing", style="yellow")
        shutil.rmtree(path)
    else:
        console.print("[INFO] ℹ️ No previous config found", style="blue")
        return {}
    return json.loads(path.read_text())

def save_config_cache(path: Path, config: dict) -> bool:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(config,default=json_default,indent=4))
        console.print(f"[INFO] ✅ Config saved to {path}", style="green")
        return True
    except Exception as e:
        console.print(f"[ERROR] ❌ Failed to save config: {str(e)}", style="red")
        return False

def is_valid_git_repo(path: str) -> bool:        
    try:
        _ = git.Repo(path).git_dir
        return True
    except Exception as e:
        return False

def is_node_exists(config: dict) -> bool:
    node_name = config['name']
    node_alt_name = config.get('alt_name', node_name.lower())
    possible_paths = { COMFYUI_PATH / "custom_nodes" / name for name in [node_name, node_alt_name] } 

    for p in possible_paths:
        if p.exists() and is_valid_git_repo(p):
            return True
        elif p.is_dir():
            console.print(f"[WARN] ⚠️ {node_name} invalid, removing: {p}", style="yellow")
            shutil.rmtree(p)
        elif p.is_file():
            console.print(f"[WARN] ⚠️ {node_name} invalid, removing: {p}", style="yellow")
            p.unlink()
    return False

def install_node(config: dict, progress: BootProgress = None) -> bool:
    try:
        node_name = config['name']
        node_url = config['url']
        if node_name in BOOT_INIT_NODE_EXCLUDE:
            console.print(f"[WARN] ⚠️ Cannot install node: {node_name}", style="yellow")
            return False
        if is_node_exists(config):
            console.print(f"[INFO] ℹ️ {node_name} already exists, skip.", style="blue")
            return False
        msg = f"[INFO] 📦 Installing node: {node_name}"
        if progress:
            progress.advance(msg=msg, style="blue")
        else:
            console.print(msg, style="blue")
        subprocess.run(["comfy", "node", "install", node_url], check=True)
        if 'script' in config:
            exec_script(config['script'])
        return True
    except Exception as e:
        console.print(f"[ERROR] ❌ Failed to install node {node_name}: {str(e)}", style="red")
        return False

def uninstall_node(config: dict, progress: BootProgress = None) -> bool:
    try:
        node_name = config['name']
        node_alt_name = config.get('alt_name', node_name.lower())
        if node_name in BOOT_INIT_NODE_EXCLUDE or node_alt_name in BOOT_INIT_NODE_EXCLUDE:
            console.print(f"[WARN] ⚠️ Cannot uninstall node: {node_name}", style="yellow")
            return False
        if not is_node_exists(config):
            console.print(f"[INFO] ℹ️ {node_name} not found, skip.", style="blue")
            return False
        possible_paths = { COMFYUI_PATH / "custom_nodes" / name for name in [node_name, node_alt_name] } 
        msg = f"[INFO] 🗑️ Uninstalling node: {node_name}"
        if progress:
            progress.advance(msg=msg, style="blue")
        else:
            console.print(msg, style="blue")
        for node_path in possible_paths:
            if node_path.exists():
                shutil.rmtree(node_path)
        console.print(f"[INFO] ✅ Uninstalled node: {node_name}", style="green")
        return True
    except Exception as e:
        console.print(f"[ERROR] ❌ Failed to uninstall node {node_name}: {str(e)}", style="red")
        return False

def init_nodes(current_config: list[dict], prev_config: list[dict] = None) -> bool:

    if not current_config:
        console.print("[INFO] 📦 No nodes in config", style="blue")
        return False

    if not prev_config:
        install_nodes = current_config
        uninstall_nodes = []
    else:
        install_nodes = [node for node in current_config if node not in prev_config]
        uninstall_nodes = [node for node in prev_config if node not in current_config]

    if not install_nodes and not uninstall_nodes:
        console.print("[INFO] ℹ️ No changes in nodes", style="blue")
        return False
    if install_nodes:
        install_count = len(install_nodes)
        console.print(f"[INFO] 📦 Installing {install_count} nodes:", style="blue")
        for node in install_nodes:
            console.print(f"      └─ {node['url']}", style="cyan")
        install_progress = BootProgress()
        install_progress.start(install_count)
        for node in install_nodes:
            install_node(node, install_progress)
    if uninstall_nodes:
        uninstall_count = len(uninstall_nodes)
        console.print(f"[INFO] 🗑️ Uninstalling {uninstall_count} nodes:", style="blue")
        for node in uninstall_nodes:
            console.print(f"      └─ {node['name']}", style="cyan")
        uninstall_progress = BootProgress()
        uninstall_progress.start(uninstall_count)
        for node in uninstall_nodes:
            uninstall_node(node, uninstall_progress)
    return True

def is_model_exists(config: dict) -> bool:
    model_path = Path(config['path'])
    model_filename = config['filename']
    if model_path.exists():
        if model_path.is_file():
            return True
        else:
            console.print(f"[WARN] ⚠️ {model_filename} invalid, removing: {model_path}", style="yellow")
            shutil.rmtree(model_path)
    return False

def download_model(config: dict, progress: BootProgress = None) -> bool:
    try:
        model_url = config['url']
        model_dir = config['dir']
        model_filename = config['filename']
        if is_model_exists(config):
            console.print(f"[INFO] ℹ️ {model_filename} already exists in {model_dir}, skip.", style="blue")
            return False
        msg = f"[INFO] ⬇️ Downloading model: {model_filename} -> {model_dir}"
        if progress:
            progress.advance(msg=msg, style="blue")
        else:
            console.print(msg, style="blue")
        subprocess.run(["comfy", "model", "download", "--url", model_url, "--relative-path", model_dir, "--filename", model_filename], check=True)
        return True
    except Exception as e:
        console.print(f"[ERROR] ❌ Failed to download model {model_filename}: {str(e)}", style="red")
        return False

def move_files(src: Path, dst: Path, progress: BootProgress = None) -> bool:
    try:
        msg = f"[INFO] 📦 Moving: {src} -> {dst}"
        if progress:
            progress.advance(msg=msg, style="blue")
        else:
            console.print(msg, style="blue")
        src.rename(dst)
        return True
    except Exception as e:
        console.print(f"[ERROR] ❌ Failed to move: {src} -> {dst}\n{str(e)}", style="red")
        return False

def remove_model(config: dict, progress: BootProgress = None) -> bool:
    try:
        model_path = Path(config['path'])
        model_filename = config['filename']
        if not is_model_exists(config):
            console.print(f"[INFO] ℹ️ {model_filename} not found in path: {model_path}, skip.", style="blue")
            return False
        msg = f"[INFO] 🗑️ Removing model: {model_filename}"
        if progress:
            progress.advance(msg=msg, style="blue")
        else:
            console.print(msg, style="blue")
        model_path.unlink()
        console.print(f"[INFO] ✅ Removed model: {model_filename}", style="green")
        return True
    except Exception as e:
        console.print(f"[ERROR] ❌ Failed to remove model {model_filename}: {str(e)}", style="red")
        return False


def init_models(current_config: list, prev_config: list = None):
    if not current_config:
        console.print("[INFO] 📦 No models in config", style="blue")
        return False
    
    if not prev_config:
        download_models = current_config
        move_models = []
        remove_models = []

    else:
        download_models = []
        move_models = []
        for model in current_config:
            if model not in prev_config:
                download_models.append(model)
            else:
                prev_model = next((m for m in prev_config if m['url'] == model['url']), None)
                prev_path = Path(prev_model['path'])
                current_path = Path(model['path'])
                if current_path != prev_path:
                    move_models.append({"src": prev_path, "dst": current_path})
        remove_models = []
        for prev_model in prev_config:
            if not any(model['url'] == prev_model['url'] for model in current_config):
                remove_models.append(prev_model)
    
    if not download_models and not move_models and not remove_models:
        console.print("[INFO] ℹ️ No changes in models", style="blue")
        return False
    if download_models:
        download_count = len(download_models)
        console.print(f"[INFO] ⬇️ Downloading {download_count} models:", style="blue")
        for model in download_models:
            console.print(f"      └─ {model['filename']}", style="cyan")
        download_progress = BootProgress()
        download_progress.start(download_count)
        for model in download_models:
            download_model(model, download_progress)
    if move_models:
        move_count = len(move_models)
        console.print(f"[INFO] 📦 Moving {move_count} models:", style="blue")
        for model in move_models:
            console.print(f"      └─ {model['src']} -> {model['dst']}", style="cyan")
        move_progress = BootProgress()
        move_progress.start(move_count)
        for file in move_models:
            move_files(file['src'], file['dst'], move_progress)
    if remove_models:
        remove_count = len(remove_models)
        console.print(f"[INFO] 🗑️ Removing {remove_count} models:", style="blue")
        for model in remove_models:
            console.print(f"      └─ {model['filename']}", style="cyan")
        remove_progress = BootProgress()
        remove_progress.start(remove_count)
        for model in remove_models:
            remove_model(model, remove_progress)
    return True

def exec_script(path: str) -> bool:
    script = Path(path)
    if not script.is_file():
        console.print(f"[WARN] ⚠️ Invalid script path: {path}", style="yellow")
        return False
        
    try:
        console.print(f"[INFO] 🛠️ Executing: {path}", style="blue")
        result = subprocess.run(str(script), shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"[ERROR] ❌ Script failed: {path}\n{result.stderr}", style="red")
            return False
        console.print(f"[INFO] ✅ Script completed: {path}", style="green")
        return True
    except Exception as e:
        console.print(f"[ERROR] ❌ Script error: {path}\n{str(e)}", style="red")
        return False

if __name__ == '__main__':
    console = Console()

    # Environment variables
    HF_API_TOKEN = os.environ.get('HF_API_TOKEN', None)
    CIVITAI_API_TOKEN = os.environ.get('CIVITAI_API_TOKEN', None)
    COMFYUI_PATH = Path(os.environ.get('COMFYUI_PATH', "/workspace/ComfyUI"))
    COMFYUI_EXTRA_ARGS = os.environ.get('COMFYUI_EXTRA_ARGS', None)
    BOOT_CN_NETWORK = os.environ.get('BOOT_CN_NETWORK', False)
    BOOT_CONFIG_DIR = Path(os.environ.get('BOOT_CONFIG_DIR', None))
    BOOT_CONFIG_PREV_PATH = Path.home() / ".cache" / "comfyui" / "boot_config.prev.json"
    BOOT_CONFIG_INCLUDE = os.environ.get('BOOT_CONFIG_INCLUDE', None)
    BOOT_CONFIG_EXCLUDE = os.environ.get('BOOT_CONFIG_EXCLUDE', None)
    BOOT_INIT_NODE = os.environ.get('BOOT_INIT_NODE', False)
    BOOT_INIT_MODEL = os.environ.get('BOOT_INIT_MODEL', False)
    BOOT_INIT_NODE_EXCLUDE = ["ComfyUI-Manager", "comfyui-manager"] 

    # check if comfyui path exists
    if not COMFYUI_PATH.is_dir():
        console.print(f"[ERROR] ❌ Invalid ComfyUI path: {COMFYUI_PATH}", style="red")
        exit(1)

    # chinese mainland network settings
    if BOOT_CN_NETWORK:
        console.print("[INFO] 🌐 Using CN network optimization", style="blue")
        # pip source to ustc mirror
        os.environ['PIP_INDEX_URL'] = 'https://mirrors.ustc.edu.cn/pypi/web/simple'
        # huggingface endpoint to hf-mirror.com
        os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
        if HF_API_TOKEN:
            console.print("[WARN] ⚠️ HF_API_TOKEN will be sent to hf-mirror.com", style="yellow")

    cli_config_manager = ConfigManager()
    if HF_API_TOKEN:
        cli_config_manager.set(cli_constants.HF_API_TOKEN_KEY, HF_API_TOKEN)
    if CIVITAI_API_TOKEN:
        cli_config_manager.set(cli_constants.CIVITAI_API_TOKEN_KEY, CIVITAI_API_TOKEN)

    current_boot_config = load_boot_config(BOOT_CONFIG_DIR)
    prev_boot_config = load_prev_config(BOOT_CONFIG_PREV_PATH)
    if current_boot_config:
        if BOOT_INIT_NODE:
            current_nodes_config = load_nodes_config(current_boot_config)
            prev_nodes_config = prev_boot_config.get('custom_nodes', [])
            init_nodes(current_nodes_config, prev_nodes_config)
        if BOOT_INIT_MODEL:
            current_models_config = load_models_config(current_boot_config)
            prev_models_config = prev_boot_config.get('models', [])
            init_models(current_models_config, prev_models_config)
        save_config_cache(BOOT_CONFIG_PREV_PATH, current_boot_config)

    launch_args_list = ["--listen", "0.0.0.0,::", "--port", "8188"] + (COMFYUI_EXTRA_ARGS.split() if COMFYUI_EXTRA_ARGS else [])
    launch_args_str = " ".join(launch_args_list).strip()
    cli_config_manager.set(cli_constants.CONFIG_KEY_DEFAULT_LAUNCH_EXTRAS, launch_args_str)
    console.print("[INFO] 🚀 Initialization complete, launching ComfyUI...", style="green")
    subprocess.run(["comfy", "env"], check=True)
    subprocess.run(["comfy", "launch", "--"] + launch_args_list, check=True)
import os
import subprocess
import shutil
import re
from pathlib import Path
import tomllib
import git
import giturlparse
from comfy_cli.config_manager import ConfigManager
import comfy_cli.constants as cli_constants
from rich.console import Console

WORKDIR = Path(os.environ.get('WORKDIR', "/workspace"))
HF_API_TOKEN = os.environ.get('HF_API_TOKEN', None)
CIVITAI_API_TOKEN = os.environ.get('CIVITAI_API_TOKEN', None)
BOOT_CN_NETWORK = os.environ.get('BOOT_CN_NETWORK', False)
BOOT_CONFIG_DIR = Path(os.environ.get('BOOT_CONFIG_DIR', None))
BOOT_CONFIG_INCLUDE = os.environ.get('BOOT_CONFIG_INCLUDE', None)
BOOT_CONFIG_EXCLUDE = os.environ.get('BOOT_CONFIG_EXCLUDE', None)
BOOT_INIT_NODE = os.environ.get('BOOT_INIT_NODE', False)
BOOT_INIT_MODEL = os.environ.get('BOOT_INIT_MODEL', False)
COMFYUI_PATH = Path(os.environ.get('COMFYUI_PATH', "/workspace/ComfyUI"))
COMFYUI_EXTRA_ARGS = os.environ.get('COMFYUI_EXTRA_ARGS', None)

console = Console()


class BootProgress:
    def __init__(self):
        self.total_steps = 0
        self.current_step = 0

    def start(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0

    def advance(self, msg: str = None):
        self.current_step += 1
        if msg:
            self.print_progress(msg)

    def print_progress(self, msg: str = None):
        overall_progress = f"[{self.current_step}/{self.total_steps}]"
        if msg:
            console.print(f"[yellow]{overall_progress}[/yellow]: {msg}")

def compile_pattern(pattern_str: str) -> re.Pattern:
    if not pattern_str:
        return None
    try:
        return re.compile(pattern_str)
    except re.error as e:
        console.print(f"Invalid regex pattern '{pattern_str}': {str(e)}", style="red")
        return None

def preprocess_url(url: str) -> str:
    # check if url is valid
    valid_pattern = re.compile(r'^https?://')
    if not bool(valid_pattern.match(url)):
        raise Exception(f"Invalid URL: {url}")
    # chinese mainland network settings
    if BOOT_CN_NETWORK:
        if 'huggingface.co' in url:
            url = url.replace('https://huggingface.co', os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com'))
        if 'civitai.com' in url:
            url = url.replace('https://civitai.com', 'https://civitai.work')
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


def load_models_config(boot_config: dict) -> dict:
    models_config = boot_config.get('models', [])
    if not models_config:
        return []

    for model in models_config:
        try:
            model['url'] = preprocess_url(model['url'])
            model['path'] = COMFYUI_PATH / model['dir'] / model['filename']
        except KeyError as e:
            console.print(f"Invalid model config: {model}\n{str(e)}", style="yellow")
            continue

    # drop duplicates
    models_config, duplicates_count = drop_duplicates_config(models_config, ['path'])
    if duplicates_count:
        console.print(f"Ignoring {duplicates_count} duplicate models in boot config", style="yellow")
    return models_config

def load_nodes_config(boot_config: dict) -> dict:
    nodes_config = boot_config.get('custom_nodes', [])
    if not nodes_config:
        return []
    
    for node in nodes_config:
        try: 
            # lowercase urls
            node['url'] = preprocess_url(node['url']).lower()
            git_url = giturlparse.parse(node['url'])
            # validate git url
            if not git_url.valid:
                raise Exception(f"Invalid git URL: {node['url']}")
            # pharse custome node name from url
            node['name'] = git_url.name
            node['path'] = COMFYUI_PATH / "custom_nodes" / node['name']            
        except KeyError as e:
            console.print(f"Invalid node config: {node}\n{str(e)}", style="yellow")
            continue

    # drop duplicates
    nodes_config, duplicates_count = drop_duplicates_config(nodes_config, ['name'])
    if duplicates_count:
        console.print(f"Ignoring {duplicates_count} duplicate nodes in boot config", style="yellow")

    return nodes_config


def load_boot_config(path: Path) -> dict:

    console.print(f"🔍 Loading boot config from {path}...", style="blue")

    config_path = Path(path)
    if not config_path.is_dir():
        console.print(f"Invalid config path: {path}", style="yellow")
        if config_path.is_file():
            config_path.unlink()
        config_path.mkdir(parents=True, exist_ok=True)
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
            console.print(f"Include filter enabled: {BOOT_CONFIG_INCLUDE}", style="yellow")
        if exclude_pattern:
            console.print(f"Exclude filter enabled: {BOOT_CONFIG_EXCLUDE}", style="yellow")
        config_files = filtered_files

    console.print(f"Using {len(config_files)} config files in {path}:", style="blue")
    for file in config_files:
        console.print(f"  📄 {file}", style="blue")
    
    boot_config = {}
    try:
        for file in config_files:
            boot_config.update(tomllib.loads(file.read_text()))
    except Exception as e:
        console.print(f"Error loading boot config:\n{str(e)}", style="red")
        exit(1)
    return boot_config

def is_valid_git_repo(path: str) -> bool:        
    try:
        _ = git.Repo(path).git_dir
        return True
    except Exception as e:
        return False

def init_nodes(node_config: dict):
    all_count = len(node_config)
    if not all_count:
        console.print("No custom nodes found in boot config, skip.", style="blue")
        return False
    console.print(f"Installing {all_count} custom nodes in boot config:", style="blue")
    console.print("\n".join([f"  📦 {node['url']}" for node in node_config]), style="blue")
    nodes_progress = BootProgress()
    nodes_progress.start(all_count)
    for node in node_config:
        try:
            node_url = node['url']
            node_name = node['name']
            node_path = node['path']
            nodes_progress.advance(msg=f"Processing custom nodes: {node_name}")

            if node_path.exists():
                if node_path.is_dir() and is_valid_git_repo(node_path):
                    console.print(f"ℹ️ [cyan]{node_name}[/cyan] already exists, skipping...")
                    continue
                else:
                    console.print(f"⚠️ [yellow]{node_name}[/yellow] is corrupted, removing...")
                    shutil.rmtree(node_path)

            console.print(f"🔧 Installing custom nodes [blue]{node_name}[/blue]...")
            subprocess.run(["comfy", "node", "install", node_url], check=True)

            node_script = node.get('script', '')
            if node_script:
                exec_script(node_script)

        except Exception as e:
            console.print(f"❌ Error processing custom nodes [red]{node_name}[/red]:\n{str(e)}", style="red")
    return True


def init_models(model_config: dict):
    all_count = len(model_config)
    if not all_count:
        console.print("No models found in boot config, skip.", style="blue")
        return False
    console.print(f"Downloading {all_count} models in boot config:", style="blue")
    console.print("\n".join([f"  📦 {model['filename']}" for model in model_config]), style="blue")
    model_progress = BootProgress()
    model_progress.start(all_count)
    for model in model_config:
        try:
            model_url = model['url']
            model_dir = model['dir']
            model_filename = model['filename']
            model_path = model['path']
            model_progress.advance(msg=f"Processing model: {model_filename}")

            if model_path.exists():
                if model_path.is_file():
                    console.print(f"ℹ️ [cyan]{model_filename}[/cyan] already exists, skipping...")
                    continue
                else:
                    console.print(f"⚠️ [yellow]{model_filename}[/yellow] is corrupted, removing...")
                    shutil.rmtree(model_path)

            console.print(f"⬇️ Downloading model [blue]{model_filename}[/blue] -> [blue]{model_dir}[/blue]")
            download_model(model_url, model_dir, model_filename)

        except Exception as e:
            console.print(f"❌ Error processing model [red]{model_filename}[/red]:\n{str(e)}", style="red")
    return True


def exec_script(path: str) -> bool:
    script = Path(path)
    if not script.is_file():
        console.print(f"⚠️ Invalid script path: [yellow]{path}[/yellow]", style="yellow")
        return False
        
    try:
        console.print(f"🛠️ Executing script: [blue]{path}[/blue]...")
        result = subprocess.run(str(script), shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"❌ Error running script '[red]{path}[/red]':\n{result.stderr}", style="red")
            return False
        console.print(f"✅ Successfully executed: [green]{path}[/green]")
        return True
    except Exception as e:
        console.print(f"❌ Exception while running script '[red]{path}[/red]': {str(e)}", style="red")
        return False

def download_model(url: str, dir: str, filename: str) -> bool:
    try:
        subprocess.run(["comfy", "model", "download", "--url", url, "--relative-path", dir, "--filename", filename], check=True)
    except Exception as e:
        console.print(f"Error downloading model {filename}:\n{str(e)}", style="red")
        return False
    return True

if __name__ == '__main__':
    # check if comfyui path exists
    if not COMFYUI_PATH.is_dir():
        console.print(f"ERROR: Invalid ComfyUI path \"{COMFYUI_PATH}\"", style="red")
        raise Exception("Invalid ComfyUI path")

    # chinese mainland network settings
    if BOOT_CN_NETWORK:
        console.print("🌐 Optimizing for Chinese Mainland network...", style="blue")
        # pip source to ustc mirror
        os.environ['PIP_INDEX_URL'] = 'https://mirrors.ustc.edu.cn/pypi/web/simple'
        # huggingface endpoint to hf-mirror.com
        os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
        if HF_API_TOKEN:
            console.print("⚠️ Your [yellow]HF_API_TOKEN[/yellow] will be sent to a third-party server 'https://hf-mirror.com' when downloading authorized models", style="yellow")

    cli_config_manager = ConfigManager()
    if HF_API_TOKEN:
        cli_config_manager.set(cli_constants.HF_API_TOKEN_KEY, HF_API_TOKEN)
    if CIVITAI_API_TOKEN:
        cli_config_manager.set(cli_constants.CIVITAI_API_TOKEN_KEY, CIVITAI_API_TOKEN)

    boot_config = load_boot_config(BOOT_CONFIG_DIR)
    if not boot_config:
        console.print("🔍 No boot config found, continuing with default settings...", style="blue")
    else:
        if BOOT_INIT_NODE:
            nodes_config = load_nodes_config(boot_config)
            init_nodes(nodes_config)
        if BOOT_INIT_MODEL:
            models_config = load_models_config(boot_config)
            init_models(models_config)

    launch_args_list = ["--listen", "0.0.0.0,::", "--port", "8188"] + (COMFYUI_EXTRA_ARGS.split() if COMFYUI_EXTRA_ARGS else [])
    launch_args_str = " ".join(launch_args_list).strip()
    cli_config_manager.set(cli_constants.CONFIG_KEY_DEFAULT_LAUNCH_EXTRAS, launch_args_str)
    console.print("✅ Initialization completed, now launching ComfyUI...", style="green")
    subprocess.run(["comfy", "env"], check=True)
    subprocess.run(["comfy", "launch", "--"] + launch_args_list, check=True)
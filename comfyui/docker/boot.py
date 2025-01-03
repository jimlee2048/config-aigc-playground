import os
import subprocess
import shutil
from pathlib import Path
import tomllib
import git
import giturlparse
from comfy_cli.config_manager import ConfigManager
import comfy_cli.constants as cli_constants
from rich.console import Console

DEBUG = os.environ.get('DEBUG', False)
HF_API_TOKEN = os.environ.get('HF_API_TOKEN', None)
CIVITAI_API_TOKEN = os.environ.get('CIVITAI_API_TOKEN', None)
COMFYUI_EXTRA_ARGS = os.environ.get('COMFYUI_EXTRA_ARGS', None)
BOOT_CN_NETWORK = os.environ.get('BOOT_CN_NETWORK', False)
BOOT_CONFIG_DIR = os.environ.get('BOOT_CONFIG_DIR', None)
BOOT_INIT_NODE = os.environ.get('BOOT_INIT_NODE', False)
BOOT_INIT_MODEL = os.environ.get('BOOT_INIT_MODEL', False)
COMFYUI_PATH = Path(os.environ.get('COMFYUI_PATH', "/workspace/ComfyUI"))

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

boot_progress = BootProgress()

def load_boot_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.is_dir():
        console.print(f"Invalid config path: {path}", style="yellow")
        if config_path.is_file():
            config_path.unlink()
        config_path.mkdir(parents=True, exist_ok=True)
        return {}
    
    config_files = list(config_path.rglob("*.toml"))
    console.print(f"Found {len(config_files)} config files in {path}:")
    for file in config_files:
        console.print(f"  📄 {file}", style="cyan")
    
    boot_config = {}
    try:
        for file in config_files:
            boot_config.update(tomllib.loads(file.read_text()))

        # ignore duplicate models in config
        if 'models' in boot_config:
            unique_keys = set()
            unique_models = [
                model for model in boot_config['models']
                if (model['filename'], model['url'], model['dir']) not in unique_keys 
                and not unique_keys.add((model['filename'], model['url'], model['dir']))
            ]
            duplicates_count = len(boot_config['models']) - len(unique_models)
            boot_config['models'] = unique_models
            if duplicates_count:
                console.print("⚠️ Ignoring duplicate models in boot config", style="yellow")

        # ignore duplicate nodes in config
        if 'custom_nodes' in boot_config:
            unique_keys = set()
            unique_nodes = [
                node for node in boot_config['custom_nodes']
                if (node['url']) not in unique_keys 
                and not unique_keys.add((node['url']))
            ]
            duplicates_count = len(boot_config['custom_nodes']) - len(unique_nodes)
            boot_config['custom_nodes'] = unique_nodes
            if duplicates_count:
                console.print(f"[yellow]Ignoring {duplicates_count} duplicate nodes in boot config[/yellow]")

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

def process_node(node):
    try:
        node_url = node['url']
        node_name = giturlparse.parse(node_url).name
        if not giturlparse.parse(node_url).valid:
            raise Exception("Invalid git URL")
        
        boot_progress.advance(msg=f"Processing node: {node_name}")
        
        node_path = COMFYUI_PATH / "custom_nodes" / node_name
        if node_path.exists():
            if node_path.is_dir() and is_valid_git_repo(node_path):
                console.print(f"ℹ️ [cyan]{node_name}[/cyan] already exists, skipping...")
                return
            else:
                console.print(f"⚠️ [yellow]{node_name}[/yellow] is corrupted, removing...")
                shutil.rmtree(node_path)

        console.print(f"🔧 Installing [blue]{node_name}[/blue]...")
        subprocess.run(["comfy", "node", "install", node_url], check=True)

        node_script = node.get('script', '')
        if node_script:
            exec_script(node_script)
            
    except Exception as e:
        console.print(f"❌ Error processing [red]{node_name}[/red]:\n{str(e)}", style="red")

def process_model(model):
    try:
        model_url = model['url']
        model_dir = model['dir']
        model_filename = model['filename']
        
        boot_progress.advance(msg=f"Processing model: {model_filename}")
        
        model_path = COMFYUI_PATH / model_dir / model_filename
        if model_path.exists():
            if model_path.is_file():
                console.print(f"ℹ️ [cyan]{model_filename}[/cyan] already exists, skipping...")
                return
            else:
                console.print(f"⚠️ [yellow]{model_filename}[/yellow] is corrupted, removing...")
                shutil.rmtree(model_path)

        if BOOT_CN_NETWORK:
            if 'huggingface.co' in model_url:
                model_url = model_url.replace('https://huggingface.co', os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com'))
            if 'civitai.com' in model_url:
                model_url = model_url.replace('https://civitai.com', 'https://civitai.work')
                
        console.print(f"⬇️ Downloading [blue]{model_filename}[/blue] -> [blue]{model_dir}[/blue]")
        download_model(model_url, model_dir, model_filename)
            
    except Exception as e:
        console.print(f"❌ Error processing model [red]{model_filename}[/red]:\n{str(e)}", style="red")

def init_nodes(boot_config: dict):
    node_config = boot_config.get('custom_nodes', [])
    all_count = len(node_config)
    console.print(f"Found {all_count} custom nodes in boot config", style="blue")
    
    if all_count > 0:
        boot_progress.start(all_count)
        for node in node_config:
            process_node(node)
    return all_count

def init_models(boot_config: dict):
    model_config = boot_config.get('models', [])
    all_count = len(model_config)
    console.print(f"Found {all_count} models in boot config", style="blue")
    
    if all_count > 0:
        boot_progress.start(all_count)
        for model in model_config:
            process_model(model)
    return all_count

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
        total_steps = 0
        if BOOT_INIT_NODE:
            total_steps += len(boot_config.get('custom_nodes', []))
        if BOOT_INIT_MODEL:
            total_steps += len(boot_config.get('models', []))
            
        boot_progress.start(total_steps)
        if BOOT_INIT_NODE:
            init_nodes(boot_config)
        if BOOT_INIT_MODEL:
            init_models(boot_config)


    launch_args_list = ["--listen", "0.0.0.0,::", "--port", "8188"] + (COMFYUI_EXTRA_ARGS.split() if COMFYUI_EXTRA_ARGS else [])
    launch_args_str = " ".join(launch_args_list).strip()
    cli_config_manager.set(cli_constants.CONFIG_KEY_DEFAULT_LAUNCH_EXTRAS, launch_args_str)
    subprocess.run(["comfy", "env"], check=True)
    console.print("✅ Initialization completed, now launching ComfyUI...", style="green")
    subprocess.run(["comfy", "launch", "--"] + launch_args_list, check=True)